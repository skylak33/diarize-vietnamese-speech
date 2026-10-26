"""
AAM-Softmax loss.
Based on: https://github.com/clovaai/voxceleb_trainer/blob/master/loss/aamsoftmax.py
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.metrics import accuracy


class AAMsoftmax(nn.Module):
    def __init__(self, n_class, m, s):
        super().__init__()
        self.m = m
        self.s = s
        self.weight = nn.Parameter(torch.FloatTensor(n_class, 192))
        self.ce = nn.CrossEntropyLoss()
        nn.init.xavier_normal_(self.weight, gain=1)
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, x, label=None):
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.mul(cosine, cosine)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.s
        loss = self.ce(output, label)
        prec = accuracy(output.detach(), label.detach(), topk=(1,))[0]
        return loss, prec
