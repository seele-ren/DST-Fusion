"""匹配输入 Transformer，移除静态上下文。"""


import torch
from torch import nn


MODEL_ID = 'transformer_matched_no_context'


class RevisionModel(nn.Module):

    def __init__(self, dropout=0.2112762832922651):
        super().__init__()
        self.variant = 'transformer_matched_no_context'
        self.is_transformer = True
        self.has_graph = False
        channels = 21
        self.input = nn.Linear(channels, 128)
        self.cls = nn.Parameter(torch.zeros(1, 1, 128))
        self.position = nn.Parameter(torch.zeros(1, 176, 128))
        nn.init.trunc_normal_(self.position, std=0.02)
        layer = nn.TransformerEncoderLayer(128, 4, 256, dropout, batch_first=True, norm_first=True, activation='gelu')
        self.temporal = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        use_static = True
        self.context = nn.Linear(8, 128) if use_static else None
        self.static_film = nn.Linear(128, 256) if use_static and True else None
        self.static_cat = None
        self.head = nn.Sequential(nn.Linear(128, 256), nn.ReLU(), nn.Dropout(dropout), nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, 1))
        self.context = None
        self.static_film = None

    def forward(self, x, s, w, mask, adj):
        b, n, t, f = x.shape
        ids = mask.reshape(-1).bool()
        xf = x.reshape(-1, t, f)[ids]
        wf = w.reshape(-1, 25, 2)[ids]
        xf = torch.cat([xf, wf.repeat_interleave(7, dim=1)], -1)
        v = self.input(xf)
        v = torch.cat([self.cls.expand(v.size(0), -1, -1), v], 1) + self.position
        v = self.temporal(v)[:, 0]
        z = x.new_zeros((b * n, 128))
        z[ids] = v
        z = z.reshape(b, n, 128)
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
