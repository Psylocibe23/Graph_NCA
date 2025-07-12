import torch
import torch.nn as nn


class GateConv(nn.Module):
    """
    Applies a 3x3 convolutions and a sigmoid to compute a per-channel, per-position gate
    input: (B, 1, H, W) affinity score
    output: (B, C, H, W) gate mask
    """
    def __init__(self, out_channels, kernel_size=3):
        super().__init__()
        padding = kernel_size//2
        self.conv = nn.Conv2d(1, out_channels=out_channels, kernel_size=kernel_size, padding=padding)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        gate = self.conv(x)
        gate = self.sigmoid(gate)
        return gate
    