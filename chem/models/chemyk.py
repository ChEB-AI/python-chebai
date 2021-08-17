import os
import sys
import pickle

import networkx as nx
from torch import nn

from torch.nn.functional import pad
import torch
from torch import functional as F
import logging
from chem.models.base import JCIBaseNet

logging.getLogger('pysmiles').setLevel(logging.CRITICAL)


class ChemYK(JCIBaseNet):
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
        self.output = nn.Sequential(nn.Linear(in_d, in_d), nn.ReLU(), nn.Dropout(0.2), nn.Linear(in_d, num_classes))

    def forward(self, data, *args, **kwargs):
        m = self.embedding(data.x)
        max_width = m.shape[1]
        h = m.unsqueeze(1) #torch.zeros(emb.shape[0], max_width, *emb.shape[1:])
        #h[:, 0] = emb
        for width in range(1, max_width):
            l = h[:, :, :-width]
            r = torch.stack(tuple(torch.diagonal(h[:, :, off:], dim1=1, dim2=2).transpose(1,2) for off in
             range(1, max_width-(width-1)))).permute(1,2,0,3).flip(1)
            m0 = self.merge(l,r)
            m = pad(m0,(0,0,0,width))
            h = torch.cat((m, h), dim=1)
        return self.output(m[:,0,0]).squeeze(1)

    def get_all_merges(self, h, left, width):
        "Returns all possible merges for the substring [left, left+width]"
        # merge_point is the offset at which a specific split is done. Note that this splits the interval in a part of
        # length `merge_point` and a part of length `width - merge_point`
        s = torch.stack([self.merge(h[merge_point-1][:, left], h[width-merge_point-1][:, left+merge_point]) for merge_point in range(1,width)])
        return self.attention(s).unsqueeze(1)

    def merge(self, l, r):
        x = torch.stack([self.a_l(l), self.a_r(r)])
        beta = torch.softmax(x, 0)
        return self.attention(torch.sum(beta*torch.stack([self.w_l(l), self.w_r(r)]), dim=0)).unsqueeze(1)

    def attention(self, parts):
        at = torch.softmax(self.s(parts), 1)
        return torch.sum(at*parts, dim=1)
