"""日尺度 LSTM。使用 175 日序列和 9 项动态变量。"""


import torch
from torch import nn


MODEL_ID = 'lstm_daily'


class DailyLSTMBaseline(torch.nn.Module):
    """A small LSTM regressor over daily feature sequences."""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        lstm_dropout = float(dropout) if num_layers > 1 else 0.0
        self.lstm = torch.nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.head = torch.nn.Sequential(
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(float(dropout)),
            torch.nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x)
        return self.head(h_n[-1]).squeeze(-1)


class CountyBaseline(nn.Module):
    """Adapt the existing county baselines to the shared annual batches."""

    def __init__(self):
        super().__init__()
        self.variant = 'lstm_daily'
        self.network = DailyLSTMBaseline(9, 256, 2, 0.3)

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
