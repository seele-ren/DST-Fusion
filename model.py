"""
DDCN 模型实现：
- WeekTransformer：周内序列编码（Transformer）
- ATLSTM：带注意力的 LSTM 聚合
- DDCN：单塔时序模型
- DualTowerDDCN：动态时序 + 静态特征双塔模型
"""
import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn


class WeekTransformer(nn.Module):
    """把每周 7 天的特征序列编码为周级向量。"""
    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        use_cls_token: bool = False,
        post_softmax: bool = False,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.use_cls_token = use_cls_token
        self.post_softmax = post_softmax

        # 用 1D 卷积做投影：输入 (B, 7, F) -> (B, 7, d_model)
        self.input_proj = nn.Conv1d(input_dim, d_model, kernel_size=1)
        self.pos_embed = nn.Parameter(torch.zeros(1, 7 + int(use_cls_token), d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        else:
            self.cls_token = None
        # 嵌入投影层：全连接 + 激活/归一化（或 softmax）
        self.embed_fc = nn.Linear(d_model, d_model)
        self.embed_act = nn.GELU()
        self.embed_dropout = nn.Dropout(dropout)
        self.embed_norm = nn.LayerNorm(d_model)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x_week: torch.Tensor, day_mask: Optional[torch.Tensor]) -> torch.Tensor:
        # x_week: (B, W, 7, F) 按周切片的日序列
        # 先把每个周片段当成一个小序列送入 Transformer
        bsz, weeks, days, feats = x_week.shape
        x = x_week.view(bsz * weeks, days, feats)
        # Conv1d 期望 (B, C, L)
        x = self.input_proj(x.transpose(1, 2)).transpose(1, 2)
        if self.use_cls_token:
            cls = self.cls_token.expand(bsz * weeks, -1, -1)
            x = torch.cat([cls, x], dim=1)

        x = x + self.pos_embed

        if day_mask is not None:
            # day_mask: (B, W, 7) -> (B*W, 7)，CLS 视为有效
            mask = day_mask.view(bsz * weeks, days)
            if self.use_cls_token:
                cls_mask = torch.ones((bsz * weeks, 1), device=mask.device, dtype=mask.dtype)
                mask = torch.cat([cls_mask, mask], dim=1)
            key_padding_mask = ~mask.bool()
        else:
            key_padding_mask = None

        # Transformer 编码每周序列
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)

        # 用全局最大池化聚合序列（把一周变成一个向量）
        week_feat = torch.amax(x, dim=1)
        # 线性投影 + 激活/归一化 或 softmax
        week_feat = self.embed_fc(week_feat)
        if self.post_softmax:
            week_feat = torch.softmax(week_feat, dim=-1)
            week_feat = self.embed_dropout(week_feat)
        else:
            week_feat = self.embed_act(week_feat)
            week_feat = self.embed_dropout(week_feat)
            week_feat = self.embed_norm(week_feat)
        return week_feat.view(bsz, weeks, self.d_model)


class ATLSTM(nn.Module):
    """LSTM + 注意力汇聚模块，用于从序列中提取上下文向量。"""
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = False,
        attn_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True,
        )
        attn_dim = hidden_dim * (2 if bidirectional else 1)
        self.attn_proj = nn.Linear(attn_dim, attn_dim)
        self.attn_score = nn.Linear(attn_dim, 1, bias=False)
        self.attn_dropout = nn.Dropout(attn_dropout)

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        hx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # x: (B, L, D), lengths: (B,) 变长序列用 pack/pad 处理
        # lengths 是每个样本的有效长度（去掉补齐部分）
        lengths = lengths.clamp(min=1)
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, (h, c) = self.lstm(packed, hx)
        out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True, total_length=x.size(1)
        )

        # 注意力对时间维加权
        scores = self.attn_score(torch.tanh(self.attn_proj(out))).squeeze(-1)
        mask = torch.arange(out.size(1), device=out.device).unsqueeze(0) < lengths.unsqueeze(1)
        scores = scores.masked_fill(~mask, -1e9)
        attn = torch.softmax(scores, dim=1)
        attn = self.attn_dropout(attn)
        context = torch.sum(attn.unsqueeze(-1) * out, dim=1)
        return context, attn, (h, c)


class DDCN(nn.Module):
    """单塔时序模型：日 -> 周 -> 阶段 -> 季节，最后回归产量。"""
    def __init__(
        self,
        feature_dim: int = 9,
        week_dim: int = 128,
        transformer_heads: int = 4,
        stage_hidden: int = 256,
        season_hidden: int = 256,
        lstm_layers: int = 1,
        stage_lengths: Optional[List[int]] = None,
        stage_ratio: Optional[List[int]] = None,
        auto_extend_last_stage: bool = True,
        dropout: float = 0.1,
        sif_week_idx: Optional[int] = None,
        sif_missing_idx: Optional[int] = None,
        sif_stage_indices: Optional[List[int]] = None,
        use_week_extra_concat: bool = False,
        week_extra_dim: int = 0,
    ) -> None:
        super().__init__()
        if stage_lengths is None:
            stage_lengths = [10, 11]
        if stage_ratio is not None and len(stage_ratio) == 0:
            stage_ratio = None
        self.stage_lengths = stage_lengths
        self.stage_ratio = stage_ratio
        self.auto_extend_last_stage = auto_extend_last_stage
        self.sif_week_idx = sif_week_idx
        self.sif_missing_idx = sif_missing_idx
        self.sif_stage_indices = (
            sorted(set(int(i) for i in sif_stage_indices)) if sif_stage_indices else None
        )
        self.use_week_extra_concat = bool(use_week_extra_concat)
        self.week_extra_dim = week_extra_dim

        self.week_encoder = WeekTransformer(
            feature_dim,
            d_model=week_dim,
            n_heads=transformer_heads,
            dropout=dropout,
            post_softmax=True,
        )
        self.stage_lstm = ATLSTM(
            week_dim,
            stage_hidden,
            num_layers=lstm_layers,
            dropout=dropout,
            attn_dropout=dropout,
        )
        self.season_lstm = ATLSTM(
            stage_hidden,
            season_hidden,
            num_layers=lstm_layers,
            dropout=dropout,
            attn_dropout=dropout,
        )
        self.regressor = nn.Sequential(
            nn.Linear(season_hidden, season_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(season_hidden // 2, 1),
        )
        self.sif_film = None
        if sif_week_idx is not None:
            self.sif_film = nn.Linear(1, stage_hidden * 2)
        self.week_extra_proj = None
        if week_extra_dim > 0:
            self.week_extra_proj = nn.Linear(week_dim + week_extra_dim, week_dim)

    def _pad_or_trim(self, x: torch.Tensor, target_days: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # 返回补齐/裁剪后的序列与有效天掩码
        # 统一长度：太长就截断，太短就补 0
        bsz, t_days, feats = x.shape
        if t_days >= target_days:
            x = x[:, :target_days, :]
            mask = torch.ones((bsz, target_days), device=x.device, dtype=torch.bool)
            return x, mask
        pad = target_days - t_days
        padding = torch.zeros((bsz, pad, feats), device=x.device, dtype=x.dtype)
        x = torch.cat([x, padding], dim=1)
        mask = torch.cat(
            [torch.ones((bsz, t_days), device=x.device, dtype=torch.bool),
             torch.zeros((bsz, pad), device=x.device, dtype=torch.bool)],
            dim=1,
        )
        return x, mask

    def _ratio_to_lengths(self, total_weeks: int, ratios: List[int]) -> List[int]:
        # 将比例分配到具体周数，保证总和为 total_weeks
        total_ratio = float(sum(ratios))
        exact = [total_weeks * r / total_ratio for r in ratios]
        floors = [int(math.floor(v)) for v in exact]
        remainder = total_weeks - sum(floors)
        if remainder > 0:
            frac = sorted(
                range(len(ratios)),
                key=lambda i: exact[i] - floors[i],
                reverse=True,
            )
            for i in range(remainder):
                floors[frac[i % len(frac)]] += 1
        return floors

    def _compute_week_sif_stats(
        self, week_extra: torch.Tensor, week_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sif_vals = week_extra[..., self.sif_week_idx]
        sif_vals = torch.nan_to_num(sif_vals, nan=0.0)
        if self.sif_missing_idx is not None:
            miss_vals = week_extra[..., self.sif_missing_idx]
            miss_vals = torch.nan_to_num(miss_vals, nan=1.0)
            valid = week_mask & (miss_vals <= 0)
            miss_ratio = (miss_vals > 0).float() * week_mask.float()
        else:
            valid = week_mask
            miss_ratio = torch.zeros_like(sif_vals)
        sif_week = sif_vals
        week_valid = valid
        return sif_week, week_valid, miss_ratio

    def _merge_week_extra(
        self, week_feat: torch.Tensor, week_extra: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.week_extra_proj is None:
            return week_feat
        merged = torch.cat([week_feat, week_extra], dim=-1)
        return self.week_extra_proj(merged)

    def _apply_sif_film(self, stage_vec: torch.Tensor, sif_stage: torch.Tensor) -> torch.Tensor:
        film = self.sif_film(sif_stage.unsqueeze(1))
        gamma, beta = film.chunk(2, dim=1)
        return stage_vec * (1.0 + gamma) + beta

    def forward(self, x: torch.Tensor, week_extra: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (B, T_days, feature_dim)
        # 先把天数换算成周数（不足一周则补齐到整周）
        total_weeks = int(math.ceil(x.size(1) / 7))
        if self.stage_ratio is not None:
            # 按比例分配每个阶段的周数
            stage_lengths = self._ratio_to_lengths(total_weeks, self.stage_ratio)
        else:
            stage_lengths = list(self.stage_lengths)
            stage_sum = sum(stage_lengths)
            if self.auto_extend_last_stage and stage_sum < total_weeks:
                stage_lengths[-1] += total_weeks - stage_sum
            elif stage_sum > total_weeks:
                stage_lengths[-1] = max(1, stage_lengths[-1] - (stage_sum - total_weeks))
        target_days = total_weeks * 7
        # 对齐到整周，得到有效天掩码
        x_days, day_mask = self._pad_or_trim(x, target_days)
        bsz = x_days.size(0)

        # (B, total_weeks, 7, feature_dim)
        x_week = x_days.view(bsz, total_weeks, 7, -1)
        day_mask = day_mask.view(bsz, total_weeks, 7)
        week_mask = day_mask.any(dim=2)

        # 周内 Transformer 聚合 -> 得到每周向量
        week_feat = self.week_encoder(x_week, day_mask)
        if week_extra is not None:
            week_extra = week_extra.to(week_feat.device)
            if week_extra.size(1) != week_feat.size(1):
                raise ValueError("week_extra length does not match week features")
            if self.use_week_extra_concat:
                week_feat = self._merge_week_extra(week_feat, week_extra)
        use_sif_film = (
            self.sif_film is not None
            and self.sif_week_idx is not None
            and week_extra is not None
        )
        if use_sif_film:
            sif_week, week_valid, miss_ratio_week = self._compute_week_sif_stats(
                week_extra, week_mask
            )

        # 逐阶段聚合（阶段内 LSTM + 注意力）
        stage_feats = []
        stage_valid = []
        start = 0
        stage_state = None
        for stage_idx, length in enumerate(stage_lengths):
            end = start + length
            stage_x = week_feat[:, start:end, :]
            stage_len = week_mask[:, start:end].sum(dim=1)
            # 阶段内 LSTM + 注意力
            stage_vec, _, stage_state = self.stage_lstm(stage_x, stage_len, stage_state)
            use_stage_sif = use_sif_film and (
                self.sif_stage_indices is None or stage_idx in self.sif_stage_indices
            )
            if use_stage_sif:
                stage_week_mask = week_mask[:, start:end]
                stage_valid_mask = week_valid[:, start:end] & stage_week_mask
                valid_count = stage_valid_mask.sum(dim=1).clamp(min=1)
                sif_stage = (
                    sif_week[:, start:end] * stage_valid_mask.float()
                ).sum(dim=1) / valid_count
                _ = (
                    miss_ratio_week[:, start:end] * stage_week_mask.float()
                ).sum(dim=1) / stage_week_mask.sum(dim=1).clamp(min=1)
                stage_vec = self._apply_sif_film(stage_vec, sif_stage)
            valid = (stage_len > 0).float().unsqueeze(1)
            stage_feats.append(stage_vec * valid)
            stage_valid.append(valid.squeeze(1))
            start = end

        # 组装阶段序列，再做季节级别 LSTM 聚合
        stage_feat = torch.stack(stage_feats, dim=1)
        stage_valid = torch.stack(stage_valid, dim=1)
        season_lengths = stage_valid.sum(dim=1).clamp(min=1).long()
        # 生长季级别聚合，再回归产量
        season_vec, _, _ = self.season_lstm(stage_feat, season_lengths)

        # 回归到产量
        yield_pred = self.regressor(season_vec).squeeze(-1)
        return yield_pred


class ContextTower(nn.Module):
    """
    静态特征塔：将物候特征（SOS, VGS, RGS）编码为向量。
    """
    def __init__(
        self,
        static_dim: int = 3,
        output_dim: int = 128,
    ) -> None:
        super().__init__()
        self.static_dim = static_dim
        self.fc_net = nn.Linear(static_dim, output_dim)

    def forward(self, static: torch.Tensor) -> torch.Tensor:
        """
        Args:
            static: (B, 3) - [SOS, VGS, RGS]
        Returns:
            (B, output_dim)
        """
        # 静态特征线性映射
        return self.fc_net(static)

class DynamicTower(nn.Module):
    """
    动态时序塔：处理时间序列数据（DDCN结构）
    """
    def __init__(
        self,
        feature_dim: int = 9,
        week_dim: int = 128,
        transformer_heads: int = 4,
        stage_hidden: int = 256,
        season_hidden: int = 256,
        output_dim: int = 128,
        lstm_layers: int = 1,
        stage_lengths: Optional[List[int]] = None,
        stage_ratio: Optional[List[int]] = None,
        auto_extend_last_stage: bool = True,
        dropout: float = 0.1,
        sif_week_idx: Optional[int] = None,
        sif_missing_idx: Optional[int] = None,
        sif_stage_indices: Optional[List[int]] = None,
        use_week_extra_concat: bool = False,
        week_extra_dim: int = 0,
    ) -> None:
        super().__init__()
        if stage_lengths is None:
            stage_lengths = [10, 11]
        if stage_ratio is not None and len(stage_ratio) == 0:
            stage_ratio = None
        self.stage_lengths = stage_lengths
        self.stage_ratio = stage_ratio
        self.auto_extend_last_stage = auto_extend_last_stage
        self.sif_week_idx = sif_week_idx
        self.sif_missing_idx = sif_missing_idx
        self.sif_stage_indices = (
            sorted(set(int(i) for i in sif_stage_indices)) if sif_stage_indices else None
        )
        self.use_week_extra_concat = bool(use_week_extra_concat)
        self.week_extra_dim = week_extra_dim

        self.week_encoder = WeekTransformer(
            feature_dim, d_model=week_dim, n_heads=transformer_heads, dropout=dropout
        )
        self.stage_lstm = ATLSTM(
            week_dim,
            stage_hidden,
            num_layers=lstm_layers,
            dropout=dropout,
            attn_dropout=dropout,
        )
        self.season_lstm = ATLSTM(
            stage_hidden,
            season_hidden,
            num_layers=lstm_layers,
            dropout=dropout,
            attn_dropout=dropout,
        )
        # 输出投影层
        self.output_proj = nn.Sequential(
            nn.Linear(season_hidden, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.sif_film = None
        if sif_week_idx is not None:
            self.sif_film = nn.Linear(1, stage_hidden * 2)
        self.week_extra_proj = None
        if week_extra_dim > 0 and self.use_week_extra_concat:
            self.week_extra_proj = nn.Linear(week_dim + week_extra_dim, week_dim)

    def _pad_or_trim(self, x: torch.Tensor, target_days: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # 返回补齐/裁剪后的序列与有效天掩码
        bsz, t_days, feats = x.shape
        if t_days >= target_days:
            x = x[:, :target_days, :]
            mask = torch.ones((bsz, target_days), device=x.device, dtype=torch.bool)
            return x, mask
        pad = target_days - t_days
        padding = torch.zeros((bsz, pad, feats), device=x.device, dtype=x.dtype)
        x = torch.cat([x, padding], dim=1)
        mask = torch.cat(
            [torch.ones((bsz, t_days), device=x.device, dtype=torch.bool),
             torch.zeros((bsz, pad), device=x.device, dtype=torch.bool)],
            dim=1,
        )
        return x, mask

    def _ratio_to_lengths(self, total_weeks: int, ratios: List[int]) -> List[int]:
        # 将比例分配到具体周数，保证总和为 total_weeks
        total_ratio = float(sum(ratios))
        exact = [total_weeks * r / total_ratio for r in ratios]
        floors = [int(math.floor(v)) for v in exact]
        remainder = total_weeks - sum(floors)
        if remainder > 0:
            frac = sorted(
                range(len(ratios)),
                key=lambda i: exact[i] - floors[i],
                reverse=True,
            )
            for i in range(remainder):
                floors[frac[i % len(frac)]] += 1
        return floors

    def _compute_week_sif_stats(
        self, week_extra: torch.Tensor, week_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sif_vals = week_extra[..., self.sif_week_idx]
        sif_vals = torch.nan_to_num(sif_vals, nan=0.0)
        if self.sif_missing_idx is not None:
            miss_vals = week_extra[..., self.sif_missing_idx]
            miss_vals = torch.nan_to_num(miss_vals, nan=1.0)
            valid = week_mask & (miss_vals <= 0)
            miss_ratio = (miss_vals > 0).float() * week_mask.float()
        else:
            valid = week_mask
            miss_ratio = torch.zeros_like(sif_vals)
        sif_week = sif_vals
        week_valid = valid
        return sif_week, week_valid, miss_ratio

    def _merge_week_extra(
        self, week_feat: torch.Tensor, week_extra: torch.Tensor
    ) -> torch.Tensor:
        if self.week_extra_proj is None:
            return week_feat
        merged = torch.cat([week_feat, week_extra], dim=-1)
        return self.week_extra_proj(merged)

    def _apply_sif_film(self, stage_vec: torch.Tensor, sif_stage: torch.Tensor) -> torch.Tensor:
        film = self.sif_film(sif_stage.unsqueeze(1))
        gamma, beta = film.chunk(2, dim=1)
        return stage_vec * (1.0 + gamma) + beta

    def forward(self, x: torch.Tensor, week_extra: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (B, T_days, feature_dim)
        Returns:
            (B, output_dim) - 动态特征向量
        """
        # 把天数换算成周数，便于后续分阶段
        total_weeks = int(math.ceil(x.size(1) / 7))
        if self.stage_ratio is not None:
            stage_lengths = self._ratio_to_lengths(total_weeks, self.stage_ratio)
        else:
            stage_lengths = list(self.stage_lengths)
            stage_sum = sum(stage_lengths)
            if self.auto_extend_last_stage and stage_sum < total_weeks:
                stage_lengths[-1] += total_weeks - stage_sum
            elif stage_sum > total_weeks:
                stage_lengths[-1] = max(1, stage_lengths[-1] - (stage_sum - total_weeks))
        
        target_days = total_weeks * 7
        # 对齐到整周，并得到有效天掩码
        x_days, day_mask = self._pad_or_trim(x, target_days)
        bsz = x_days.size(0)

        # (B, total_weeks, 7, feature_dim)
        x_week = x_days.view(bsz, total_weeks, 7, -1)
        day_mask = day_mask.view(bsz, total_weeks, 7)
        week_mask = day_mask.any(dim=2)

        # 周内 Transformer 聚合
        week_feat = self.week_encoder(x_week, day_mask)
        if week_extra is not None:
            week_extra = week_extra.to(week_feat.device)
            if week_extra.size(1) != week_feat.size(1):
                raise ValueError("week_extra length does not match week features")
            if self.use_week_extra_concat:
                week_feat = self._merge_week_extra(week_feat, week_extra)
        use_sif_film = (
            self.sif_film is not None
            and self.sif_week_idx is not None
            and week_extra is not None
        )
        if use_sif_film:
            sif_week, week_valid, miss_ratio_week = self._compute_week_sif_stats(
                week_extra, week_mask
            )

        # 分阶段 LSTM + 注意力
        stage_feats = []
        stage_valid = []
        start = 0
        stage_state = None
        for stage_idx, length in enumerate(stage_lengths):
            end = start + length
            stage_x = week_feat[:, start:end, :]
            stage_len = week_mask[:, start:end].sum(dim=1)
            stage_vec, _, stage_state = self.stage_lstm(stage_x, stage_len, stage_state)
            use_stage_sif = use_sif_film and (
                self.sif_stage_indices is None or stage_idx in self.sif_stage_indices
            )
            if use_stage_sif:
                stage_week_mask = week_mask[:, start:end]
                stage_valid_mask = week_valid[:, start:end] & stage_week_mask
                valid_count = stage_valid_mask.sum(dim=1).clamp(min=1)
                sif_stage = (
                    sif_week[:, start:end] * stage_valid_mask.float()
                ).sum(dim=1) / valid_count
                _ = (
                    miss_ratio_week[:, start:end] * stage_week_mask.float()
                ).sum(dim=1) / stage_week_mask.sum(dim=1).clamp(min=1)
                stage_vec = self._apply_sif_film(stage_vec, sif_stage)
            valid = (stage_len > 0).float().unsqueeze(1)
            stage_feats.append(stage_vec * valid)
            stage_valid.append(valid.squeeze(1))
            start = end

        # 阶段序列 -> 季节级别向量
        stage_feat = torch.stack(stage_feats, dim=1)
        stage_valid = torch.stack(stage_valid, dim=1)
        season_lengths = stage_valid.sum(dim=1).clamp(min=1).long()
        season_vec, _, _ = self.season_lstm(stage_feat, season_lengths)

        # 输出投影成固定维度的动态特征向量
        return self.output_proj(season_vec)


class GraphConv(nn.Module):
    """Simple GCN layer with normalized adjacency."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        if adj.dim() == 2:
            adj = adj.unsqueeze(0)
        if adj.device != x.device:
            adj = adj.to(x.device)
        bsz, n_nodes, _ = x.shape
        eye = torch.eye(n_nodes, device=adj.device, dtype=adj.dtype).unsqueeze(0)
        adj_hat = adj + eye
        deg = adj_hat.sum(dim=-1).clamp(min=1.0)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt.unsqueeze(-1) * adj_hat * deg_inv_sqrt.unsqueeze(-2)
        out = torch.bmm(norm, x)
        out = self.lin(out)
        out = torch.relu(out)
        return self.dropout(out)


class DSTfusion(nn.Module):
    """
    双塔 + 图卷积：在季节级别向量后进行空间邻接传播。
    """

    def __init__(
        self,
        feature_dim: int = 9,
        static_dim: int = 3,
        meteo_idx: Optional[List[int]] = None,
        sif_week_idx: Optional[int] = None,
        sif_missing_idx: Optional[int] = None,
        sif_stage_indices: Optional[List[int]] = None,
        use_week_extra_concat: bool = False,
        week_extra_dim: int = 0,
        week_dim: int = 128,
        transformer_heads: int = 4,
        stage_hidden: int = 256,
        season_hidden: int = 256,
        tower_output_dim: int = 128,
        lstm_layers: int = 1,
        stage_lengths: Optional[List[int]] = None,
        stage_ratio: Optional[List[int]] = None,
        auto_extend_last_stage: bool = True,
        fusion_hidden: int = 256,
        gnn_hidden: int = 128,
        gnn_layers: int = 2,
        gnn_edge_dropout: float = 0.0,
        gnn_alpha0: float = 0.1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.meteo_idx = sorted(set(meteo_idx or []))
        self.register_buffer(
            "_meteo_idx_t",
            torch.tensor(self.meteo_idx, dtype=torch.long),
            persistent=False,
        )

        self.dynamic_tower = DynamicTower(
            feature_dim=feature_dim,
            week_dim=week_dim,
            transformer_heads=transformer_heads,
            stage_hidden=stage_hidden,
            season_hidden=season_hidden,
            output_dim=tower_output_dim,
            lstm_layers=lstm_layers,
            stage_lengths=stage_lengths,
            stage_ratio=stage_ratio,
            auto_extend_last_stage=auto_extend_last_stage,
            dropout=dropout,
            sif_week_idx=sif_week_idx,
            sif_missing_idx=sif_missing_idx,
            sif_stage_indices=sif_stage_indices,
            use_week_extra_concat=use_week_extra_concat,
            week_extra_dim=week_extra_dim,
        )

        self.context_tower = ContextTower(
            static_dim=static_dim,
            output_dim=tower_output_dim,
        )

        self.film_net = nn.Linear(tower_output_dim, tower_output_dim * 2)

        gnn_layers = max(1, gnn_layers)
        self.gnn_in_proj = nn.Linear(tower_output_dim, gnn_hidden)
        gnn_blocks = []
        gnn_in = gnn_hidden
        for _ in range(gnn_layers):
            gnn_blocks.append(GraphConv(gnn_in, gnn_hidden, dropout=dropout))
        self.gnn_layers = nn.ModuleList(gnn_blocks)
        self.gnn_out_proj = nn.Linear(gnn_hidden, tower_output_dim)
        alpha0 = min(max(gnn_alpha0, 1e-4), 1 - 1e-4)
        self.gnn_alpha = nn.Parameter(torch.tensor(math.log(alpha0 / (1.0 - alpha0))))
        self.delta_norm = nn.LayerNorm(gnn_hidden)
        self.gnn_edge_dropout = float(gnn_edge_dropout)

        self.graph_head = nn.Sequential(
            nn.Linear(tower_output_dim, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, fusion_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden // 2, 1),
        )
        self.province_head = nn.Sequential(
            nn.Linear(tower_output_dim, fusion_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden // 2, 1),
        )

    def forward(
        self,
        x_dynamic: torch.Tensor,
        x_static: torch.Tensor,
        adj: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        week_extra: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x_dynamic: (B, N, T_days, feature_dim)
            x_static: (B, N, static_dim)
            adj: (N, N) or (B, N, N)
        Returns:
            (B, N) - 产量预测
        """
        bsz, n_nodes, t_days, _ = x_dynamic.shape
        if mask is None:
            mask = torch.ones((bsz, n_nodes), device=x_dynamic.device, dtype=torch.float32)
        else:
            mask = mask.to(x_dynamic.device).float()

        x_dynamic_flat = x_dynamic.view(bsz * n_nodes, t_days, -1)
        x_static_flat = x_static.view(bsz * n_nodes, -1)
        week_extra_flat = None
        if week_extra is not None:
            week_extra_flat = week_extra.view(bsz * n_nodes, week_extra.size(2), -1)

        season_vec = self.dynamic_tower(x_dynamic_flat, week_extra_flat).view(bsz, n_nodes, -1)
        ctx_vec = self.context_tower(x_static_flat).view(bsz, n_nodes, -1)

        z = self.gnn_in_proj(season_vec)
        z = z * mask.unsqueeze(-1)
        if adj.dim() == 2:
            adj = adj.unsqueeze(0)
        if adj.device != x_dynamic.device:
            adj = adj.to(x_dynamic.device)
        adj = adj * mask.unsqueeze(1) * mask.unsqueeze(2)
        if self.training and self.gnn_edge_dropout > 0.0:
            keep = torch.rand_like(adj) > self.gnn_edge_dropout
            keep = keep & keep.transpose(-1, -2)
            eye = torch.eye(adj.size(-1), device=adj.device, dtype=torch.bool)
            keep = (keep & ~eye) | eye
            adj = adj * keep.to(adj.dtype)
        for layer in self.gnn_layers:
            out = layer(z, adj)
            delta = self.delta_norm(out - z)
            alpha = torch.sigmoid(self.gnn_alpha)
            z = z + alpha * delta
            z = z * mask.unsqueeze(-1)

        h = self.gnn_out_proj(z)
        h = h * mask.unsqueeze(-1)

        gamma, beta = self.film_net(ctx_vec).chunk(2, dim=2)
        season_fused = h * (1.0 + gamma) + beta
        county_out = self.graph_head(season_fused).squeeze(-1)
        pooled = (season_fused * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        province_out = self.province_head(pooled).squeeze(-1)
        return county_out, province_out


class DualTowerDDCN(nn.Module):
    """
    双塔模型：动态塔（时序数据）+ 静态塔（物候特征）
    两个塔的输出进行融合后回归产量
    """
    def __init__(
        self,
        feature_dim: int = 9,
        static_dim: int = 3,
        meteo_idx: Optional[List[int]] = None,
        sif_week_idx: Optional[int] = None,
        sif_missing_idx: Optional[int] = None,
        sif_stage_indices: Optional[List[int]] = None,
        use_week_extra_concat: bool = False,
        week_extra_dim: int = 0,
        week_dim: int = 128,
        transformer_heads: int = 4,
        stage_hidden: int = 256,
        season_hidden: int = 256,
        tower_output_dim: int = 128,
        lstm_layers: int = 1,
        stage_lengths: Optional[List[int]] = None,
        stage_ratio: Optional[List[int]] = None,
        auto_extend_last_stage: bool = True,
        fusion_hidden: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.meteo_idx = sorted(set(meteo_idx or []))
        self.other_idx = [i for i in range(feature_dim) if i not in self.meteo_idx]
        self.register_buffer(
            "_meteo_idx_t",
            torch.tensor(self.meteo_idx, dtype=torch.long),
            persistent=False,
        )
        
        # 动态塔（时序特征）
        self.dynamic_tower = DynamicTower(
            feature_dim=feature_dim,
            week_dim=week_dim,
            transformer_heads=transformer_heads,
            stage_hidden=stage_hidden,
            season_hidden=season_hidden,
            output_dim=tower_output_dim,
            lstm_layers=lstm_layers,
            stage_lengths=stage_lengths,
            stage_ratio=stage_ratio,
            auto_extend_last_stage=auto_extend_last_stage,
            dropout=dropout,
            sif_week_idx=sif_week_idx,
            sif_missing_idx=sif_missing_idx,
            sif_stage_indices=sif_stage_indices,
            use_week_extra_concat=use_week_extra_concat,
            week_extra_dim=week_extra_dim,
        )
        
        # 静态塔（物候特征）
        self.context_tower = ContextTower(
            static_dim=static_dim,
            output_dim=tower_output_dim,
        )

        self.film_net = nn.Linear(tower_output_dim, tower_output_dim * 2)
        
        # 融合层：拼接动态/静态/门控/交互特征，然后回归
        self.fusion_net = nn.Sequential(
            nn.Linear(tower_output_dim, fusion_hidden),
            nn.BatchNorm1d(fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, fusion_hidden // 2),
            nn.BatchNorm1d(fusion_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden // 2, 1),
        )

    def forward(
        self,
        x_dynamic: torch.Tensor,
        x_static: torch.Tensor,
        week_extra: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x_dynamic: (B, T_days, feature_dim) - 动态时序数据
            x_static: (B, 3) - 静态物候数据 [SOS, VGS, RGS]
        Returns:
            (B,) - 产量预测
        """
        # 用 SIF summary 生成 gate，先调制气象通道
        # 动态塔与静态塔各自编码（SIF 不参与静态塔输入）
        season_vec = self.dynamic_tower(x_dynamic, week_extra)   # (B, tower_output_dim)
        ctx_vec = self.context_tower(x_static)   # (B, tower_output_dim)

        # 用非 SIF 静态特征生成门控，调制动态向量
        gamma, beta = self.film_net(ctx_vec).chunk(2, dim=1)
        season_fused = season_vec * (1.0 + gamma) + beta

        yield_pred = self.fusion_net(season_fused).squeeze(-1)
        return yield_pred
