"""
GestureTCN model definition.
"""

import torch
import torch.nn as nn

from .constants import NUM_CLASSES, FEATURE_DIM


class CausalConv1d(nn.Module):
    """Causal 1D convolution with proper padding."""

    def __init__(self, in_ch: int, out_ch: int, ks: int, dilation: int = 1):
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
    """Residual block with dilated causal convolutions."""

    def __init__(self, ch: int, ks: int = 3, dilation: int = 1, dropout: float = 0.15):
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
    """Channel-changing block with residual connection."""

    def __init__(self, in_ch: int, out_ch: int, ks: int = 3, dilation: int = 1, dropout: float = 0.15):
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
    """
    Temporal Convolutional Network for gesture recognition.

    Architecture:
    - Stem: 1x1 conv to expand channels
    - TCN blocks with increasing dilation
    - Global average pooling
    - Classification head
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        feat_dim: int = FEATURE_DIM,
        dropout: float = 0.15,
        channels: dict = None,
    ):
        """
        Initialize GestureTCN.

        Args:
            num_classes: Number of output classes
            feat_dim: Input feature dimension
            dropout: Dropout rate
            channels: Custom channel configuration for pruned models
        """
        super().__init__()

        # Default or custom channel configuration
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
        """Initialize weights."""
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
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, feat_dim, seq_len)

        Returns:
            Logits of shape (batch, num_classes)
        """
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x).squeeze(-1)
        return self.head(x)


def count_parameters(model) -> int:
    """
    Count the number of trainable parameters.

    Args:
        model: PyTorch model

    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size_mb(model) -> float:
    """
    Get model size in megabytes.

    Args:
        model: PyTorch model

    Returns:
        Size in MB
    """
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / (1024 ** 2)
