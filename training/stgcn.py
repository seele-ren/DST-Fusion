"""时空图卷积网络。使用县域邻接图和日序列。"""


import torch
from torch import nn


MODEL_ID = 'stgcn'


class TemporalConvBlock(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=(1, kernel_size), padding=(0, padding)),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
        )
        self.residual = (
            torch.nn.Identity()
            if in_channels == out_channels
            else torch.nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) + self.residual(x)


class GraphConvBlock(torch.nn.Module):
    def __init__(self, channels: int, dropout: float) -> None:
        super().__init__()
        self.proj = torch.nn.Conv2d(channels, channels, kernel_size=1)
        self.bn = torch.nn.BatchNorm2d(channels)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        agg = torch.einsum("ij,bcjt->bcit", adj, x)
        out = self.proj(agg)
        out = self.bn(out)
        out = torch.relu(out)
        return self.dropout(out)


class STGCNBlock(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        self.temporal1 = TemporalConvBlock(in_channels, out_channels, kernel_size, dropout)
        self.graph = GraphConvBlock(out_channels, dropout)
        self.temporal2 = TemporalConvBlock(out_channels, out_channels, kernel_size, dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        x = self.temporal1(x)
        x = self.graph(x, adj)
        x = self.temporal2(x)
        return x


class STGCNBaseline(torch.nn.Module):
    """A standard spatio-temporal graph baseline over county-year daily sequences."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        kernel_size: int,
        num_blocks: int,
        dropout: float,
    ) -> None:
        super().__init__()
        blocks = []
        in_channels = input_dim
        for _ in range(num_blocks):
            blocks.append(STGCNBlock(in_channels, hidden_dim, kernel_size, dropout))
            in_channels = hidden_dim
        self.blocks = torch.nn.ModuleList(blocks)
        self.head = torch.nn.Sequential(
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # x: [B, N, T, F] -> [B, F, N, T]
        x = x.permute(0, 3, 1, 2)
        for block in self.blocks:
            x = block(x, adj)
        x = x.mean(dim=-1).permute(0, 2, 1)
        return self.head(x).squeeze(-1)


class CountyBaseline(nn.Module):
    """Adapt the existing county baselines to the shared annual batches."""

    def __init__(self):
        super().__init__()
        self.variant = 'stgcn'
        self.network = STGCNBaseline(9, 64, 5, 1, 0.3)

    def forward(self, x, static, weekly, mask, adj):
        output = x.new_zeros(mask.shape)
        valid = mask.bool()
        daily = x[..., :9]
        for index, (days, nodes) in enumerate(zip(daily, valid)):
            edges = adj[nodes][:, nodes]
            edges = edges + torch.eye(edges.shape[0], device=edges.device, dtype=edges.dtype)
            scale = edges.sum(dim=1).rsqrt()
            output[index, nodes] = self.network(days[nodes].unsqueeze(0), edges * scale[:, None] * scale)[0]
        return output


def build_model():
    return CountyBaseline()


if __name__ == '__main__':
    from _shared.train import main

    main(MODEL_ID)
