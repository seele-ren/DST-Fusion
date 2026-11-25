"""DDCN 单塔。使用 147 日窗口和 4、5、4、4、4 周阶段划分。"""


import math
from typing import List, Optional, Tuple

import torch
from torch import nn

from _shared.components import ATLSTM, WeekTransformer


MODEL_ID = 'single'


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


class CountyBaseline(nn.Module):
    """Adapt the existing county baselines to the shared annual batches."""

    def __init__(self):
        super().__init__()
        self.variant = 'single'
        self.network = DDCN(transformer_heads=2, stage_lengths=[4, 5, 4, 4, 4], dropout=0.39662019990393316)

    def forward(self, x, static, weekly, mask, adj):
        output = x.new_zeros(mask.shape)
        valid = mask.bool()
        daily = x[..., :9]
        days = daily[valid]
        days = days[:, :147].contiguous()
        output[valid] = self.network(days)
        return output


def build_model():
    return CountyBaseline()


if __name__ == '__main__':
    from _shared.train import main

    main(MODEL_ID)
