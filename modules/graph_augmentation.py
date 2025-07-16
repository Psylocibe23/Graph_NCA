import torch
import torch.nn as nn
import torch.nn.functional as F

class PixelGraphMessagePassing(nn.Module):
    """
    Efficient pixel-level message passing via 3x3 convolution (local neighbor sum)
    Mimic NCA information flow and produces graph-level information (short-mid-range)
    """
    def __init__(self, channels):
        super().__init__()
        # 3x3 kernel with all ones except center zero
        kernel = torch.ones((channels, 1, 3, 3), dtype=torch.float32)
        kernel[:,:,1,1] = 0.0  # No self-loop
        self.register_buffer('kernel', kernel)
        self.msg_proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        device = x.device
        kernel = torch.ones((C, 1, 3, 3), device=device)
        kernel[:, :, 1, 1] = 0
        neighbor_sum = F.conv2d(x, kernel, bias=None, stride=1, padding=1, groups=C)
        msg = self.msg_proj(neighbor_sum)
        return msg

    