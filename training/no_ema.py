"""DST-Fusion，移除历史产量 EMA 特征及其缺失标记。"""


import math

import torch
from torch import nn

from _shared.components import ATLSTM, WeekTransformer, GraphConv


MODEL_ID = 'no_ema'
EMA_FEATURE = None


class StageEncoder(nn.Module):

    def __init__(self, dropout):
        super().__init__()
        self.variant = 'no_ema'
        self.lengths = [5, 8, 3, 3, 6]
        self.week = WeekTransformer(19, 128, 2, dropout=dropout)
        self.stage = ATLSTM(128, 256, 1, dropout=dropout, attn_dropout=dropout)
        self.season = ATLSTM(256, 256, 1, dropout=dropout, attn_dropout=dropout)
        self.film = nn.Linear(1, 512)
        self.concat = None
        self.proj = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout))

    def forward(self, x, w):
        n = x.shape[0]
        h = self.week(x.reshape(n, 25, 7, 19), torch.ones(n, 25, 7, dtype=torch.bool, device=x.device))
        start = 0
        state = None
        features = []
        for length in self.lengths:
            end = start + length
            v, _, state = self.stage(h[:, start:end], torch.full((n,), length, device=x.device), state)
            valid = (w[:, start:end, 1] <= 0) & torch.isfinite(w[:, start:end, 0])
            count = valid.sum(1)
            available = (count > 0).float().unsqueeze(1)
            sif = (torch.nan_to_num(w[:, start:end, 0]) * valid).sum(1) / count.clamp_min(1)
            if self.film is not None:
                gamma, beta = self.film(sif[:, None]).chunk(2, 1)
                v = v * (1 + gamma * available) + beta * available
            elif self.concat is not None:
                v = self.concat(torch.cat([v, sif[:, None], available], 1))
            features.append(v)
            start = end
        season, _, _ = self.season(torch.stack(features, 1), torch.full((n,), 5, device=x.device))
        return self.proj(season)


class RevisionModel(nn.Module):

    def __init__(self, dropout=0.2112762832922651):
        super().__init__()
        self.variant = 'no_ema'
        self.is_transformer = False
        self.has_graph = True
        self.temporal = StageEncoder(dropout)
        use_static = True
        self.context = nn.Linear(8, 128) if use_static else None
        self.static_film = nn.Linear(128, 256) if use_static and True else None
        self.static_cat = None
        self.in_proj = nn.Linear(128, 128)
        self.gcn = GraphConv(128, 128, dropout)
        self.out_proj = nn.Linear(128, 128)
        self.delta_norm = nn.LayerNorm(128)
        self.alpha = nn.Parameter(torch.tensor(math.log(0.005 / 0.995)), requires_grad=True)
        self.head = nn.Sequential(nn.Linear(128, 256), nn.ReLU(), nn.Dropout(dropout), nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, 1))

    def forward(self, x, s, w, mask, adj):
        b, n, t, f = x.shape
        ids = mask.reshape(-1).bool()
        xf = x.reshape(-1, t, f)[ids]
        wf = w.reshape(-1, 25, 2)[ids]
        v = self.temporal(xf, wf)
        z = x.new_zeros((b * n, 128))
        z[ids] = v
        z = z.reshape(b, n, 128)
        z = self.in_proj(z) * mask[..., None]
        a = adj.unsqueeze(0) * mask[:, :, None] * mask[:, None, :]
        out = self.gcn(z, a)
        z = z + torch.sigmoid(self.alpha) * self.delta_norm(out - z)
        z = self.out_proj(z * mask[..., None]) * mask[..., None]
        if self.context is not None:
            ctx = self.context(s)
            if self.static_cat is not None:
                z = self.static_cat(torch.cat([z, ctx], -1))
            else:
                gamma, beta = self.static_film(ctx).chunk(2, -1)
                z = z * (1 + gamma) + beta
        return self.head(z).squeeze(-1)


def build_model():
    return RevisionModel()


if __name__ == '__main__':
    from _shared.train import main

    main(MODEL_ID)
