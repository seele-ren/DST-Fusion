"""日尺度 Transformer。使用 175 日序列和 9 项动态变量。"""


import math

import torch
from torch import nn


MODEL_ID = 'transformer_daily'


class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class DailyTransformerBaseline(torch.nn.Module):
    """A small Transformer encoder regressor over daily feature sequences."""

    def __init__(
        self,
        input_dim: int,
        model_dim: int,
        num_heads: int,
        num_layers: int,
        ff_dim: int,
        dropout: float,
        max_len: int,
    ) -> None:
        super().__init__()
        self.input_proj = torch.nn.Linear(input_dim, model_dim)
        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, model_dim))
        self.pos_encoder = PositionalEncoding(model_dim, max_len=max_len + 1)
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = torch.nn.Sequential(
            torch.nn.LayerNorm(model_dim),
            torch.nn.Linear(model_dim, model_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(model_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.pos_encoder(x)
        x = self.encoder(x)
        return self.head(x[:, 0]).squeeze(-1)


class CountyBaseline(nn.Module):
    """Adapt the existing county baselines to the shared annual batches."""

    def __init__(self):
        super().__init__()
        self.variant = 'transformer_daily'
        self.network = DailyTransformerBaseline(9, 128, 2, 2, 256, 0.2, 175)

    def forward(self, x, static, weekly, mask, adj):
        output = x.new_zeros(mask.shape)
        valid = mask.bool()
        daily = x[..., :9]
        days = daily[valid]
        output[valid] = self.network(days)
        return output


def build_model():
    return CountyBaseline()


if __name__ == '__main__':
    from _shared.train import main

    main(MODEL_ID)
