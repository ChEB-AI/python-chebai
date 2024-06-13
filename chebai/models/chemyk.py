import logging
import os
import pickle
import sys

import networkx as nx
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.functional import pad

from chebai.models.base import ChebaiBaseNet

logging.getLogger("pysmiles").setLevel(logging.CRITICAL)


class ChemYK(ChebaiBaseNet):
    NAME = "ChemYK"

    def __init__(self, in_d, out_d, num_classes, **kwargs):
        super().__init__(num_classes, **kwargs)
        d_internal = in_d
        self.d_internal = d_internal
        self.embedding = nn.Embedding(800, d_internal)
        self.s = nn.Linear(d_internal, 1)
        self.a_l = nn.Linear(d_internal, 1)
        self.a_r = nn.Linear(d_internal, 1)
        self.w_l = nn.Linear(d_internal, d_internal)
        self.w_r = nn.Linear(d_internal, d_internal)
        self.norm = nn.LayerNorm(d_internal)
        self.output = nn.Sequential(
            nn.Linear(in_d, in_d),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(in_d, num_classes),
        )

    def forward(self, data, *args, **kwargs):
        m = self.embedding(data.x)
        max_width = m.shape[1]
        h = [m]  # torch.zeros(emb.shape[0], max_width, *emb.shape[1:])
        # h[:, 0] = emb
        for width in range(1, max_width):
            l = torch.stack(tuple(h[i][:, : (max_width - width)] for i in range(width)))
            r = torch.stack(
                tuple(h[i][:, (width - i) :] for i in range(0, width))
            ).flip(0)
            m = self.merge(l, r)
            h.append(m)
        return self.output(m).squeeze(1)

    def merge(self, l, r):
        x = torch.stack([self.a_l(l), self.a_r(r)])
        beta = torch.softmax(x, 0)
        return F.leaky_relu(
            self.attention(
                torch.sum(beta * torch.stack([self.w_l(l), self.w_r(r)]), dim=0)
            )
        )

    def attention(self, parts):
        at = torch.softmax(self.s(parts), 1)
        return torch.sum(at * parts, dim=0)
