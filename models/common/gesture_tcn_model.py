from .constants import *

import torch
import torch.nn as nn
import torch.optim as optim

class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, ks, dilation=1):
        super().__init__()
        self.pad = (ks - 1) * dilation
        self.conv = nn.Conv1d(
            in_ch, out_ch, ks, padding=self.pad, dilation=dilation, bias=False
        )

    def forward(self, x):
        o = self.conv(x)
        if self.pad > 0:
            o = o[:, :, : -self.pad]
        return o


class ResBlock(nn.Module):
    def __init__(self, ch, ks=3, dilation=1, dropout=0.15):
        super().__init__()
        self.net = nn.Sequential(
            CausalConv1d(ch, ch, ks, dilation),
            nn.BatchNorm1d(ch),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            CausalConv1d(ch, ch, ks, dilation),
            nn.BatchNorm1d(ch),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.net(x) + x)


class ChannelBlock(nn.Module):
    def __init__(self, in_ch, out_ch, ks=3, dilation=1, dropout=0.15):
        super().__init__()
        self.net = nn.Sequential(
            CausalConv1d(in_ch, out_ch, ks, dilation),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            CausalConv1d(out_ch, out_ch, ks, dilation),
            nn.BatchNorm1d(out_ch),
        )
        self.skip = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm1d(out_ch),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.net(x) + self.skip(x))


class GestureTCN(nn.Module):
    def __init__(
        self, num_classes=NUM_CLASSES, feat_dim=FEATURE_DIM, dropout=0.15, channels=None
    ):
        super().__init__()

        # Support pruned channel configuration
        c = channels or {"stem": 48, "mid": 48, "out": 64, "head": 32}
        stem_ch = c["stem"]
        out_ch = c["out"]
        head_ch = c["head"]

        self.stem = nn.Sequential(
            nn.Conv1d(feat_dim, stem_ch, 1, bias=False),
            nn.BatchNorm1d(stem_ch),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            ResBlock(stem_ch, 3, 1, dropout),
            ResBlock(stem_ch, 3, 2, dropout),
            ChannelBlock(stem_ch, out_ch, 3, 4, dropout),
            ResBlock(out_ch, 3, 1, dropout),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(out_ch, head_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(head_ch, num_classes),
        )
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x).squeeze(-1)
        return self.head(x)
