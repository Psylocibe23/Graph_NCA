import torch
import torch.nn as nn
import torch.nn.functional as F


class SobelFilter(nn.Module):
    def __init__(self, in_channels=3, as_grayscale=False):
        super().__init__()
        # Sobel kernels
        kernel_x = torch.tensor([[1,0,-1], [2,0,-2], [1,0,-1]], dtype=torch.float32)
        kernel_y = torch.tensor([[1,2,1], [0,0,0], [-1,-2,-1]], dtype=torch.float32)
        kx = kernel_x[None, None, :, :].repeat(in_channels, 1, 1, 1)
        ky = kernel_y[None, None, :, :].repeat(in_channels, 1, 1, 1)
        self.register_buffer('weight_x', kx)
        self.register_buffer('weight_y', ky)
        self.in_channels = in_channels
        self.as_grayscale = as_grayscale

    def forward(self, x):
        # x: (B, C, H, W)
        gx = F.conv2d(x, self.weight_x, padding=1, groups=self.in_channels)
        gy = F.conv2d(x, self.weight_y, padding=1, groups=self.in_channels)
        edge = (gx**2 + gy**2).sqrt()
        if self.as_grayscale:
            edge = edge.mean(dim=1, keepdim=True)
        return edge 
