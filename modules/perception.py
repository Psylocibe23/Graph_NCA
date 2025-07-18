import torch
import torch.nn as nn
import torch.nn.functional as F


class Perception(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        # Four fixed convolutional filters: identity, Sobel-x, Sobel-y, Laplace
        identity = torch.zeros(1, 1, 3, 3)
        identity[0, 0, 1, 1] = 1.0
        sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32).reshape(1, 1, 3, 3)
        sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).reshape(1, 1, 3, 3)
        laplace = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32).reshape(1, 1, 3, 3)
        # Stack and repeat for each channel
        kernel = torch.cat([identity, sobel_x, sobel_y, laplace], dim=0)
        kernel = kernel.repeat(n_channels, 1, 1, 1)  # shape: (4*n_channels, 1, 3, 3)
        self.register_buffer('weight', kernel)
        self.n_channels = n_channels

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        # Apply fixed kernels to each channel independently (groups=C)
        x_perc = F.conv2d(x, self.weight, padding=1, groups=C)
        # Output shape: (B, 4*C, H, W)
        return x_perc


class NCAUpdate(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.perception = Perception(n_channels)
        self.update_mlp = nn.Sequential(
            nn.Conv2d(n_channels*5, 128, 1),  # 4 filters + 1 state = 5
            nn.ReLU(),
            nn.Conv2d(128, n_channels, 1, bias=False),
        )

    def forward(self, x):
        percept = self.perception(x)
        stack = torch.cat([x, percept], dim=1)
        dx = self.update_mlp(stack)
        return x + dx