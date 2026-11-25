import math
from typing import Optional, Tuple

import torch
from torch import nn


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
