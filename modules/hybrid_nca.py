import torch
import torch.nn as nn
from modules.graph_augmentation import PixelGraphMessagePassing
from modules.perception import Perception, NCAUpdate

class HybridPixelGraphNca(nn.Module):
    """
    Pixelwise NCA with explicit channel separation:
    - Channel 0: alive
    - Channels 1-3: RGB (only decoded from hidden at each step)
    - Channels 4+: hidden (evolves over time)
    """
    def __init__(self, channels, edge_index, alpha=1.0, beta=1.0, radius=1):
        super().__init__()
        self.C = channels
        self.edge_index = edge_index
        self.alpha = alpha
        self.beta = beta
        self.d = self.C - 4  # 1 alive, 3 rgb, rest hidden

        assert self.d > 0, "Need at least 5 channels: 1 alive, 3 rgb, >=1 hidden"

        # Local NCA-style update 
        self.local_nca = NCAUpdate(self.C) 

        # Graph message passing: only update hidden channels
        self.graph_mp = nn.Conv2d(self.d, self.d, kernel_size=2*radius+1, padding=radius)
        # To use a GNN, swap with PixelGraphMessagePassing

        self.elu = nn.ELU()
        self.alive_update = nn.Conv2d(self.d, 1, 1)
        self.rgb_decode = nn.Sequential(
            nn.Conv2d(self.d + 1, self.d, 1), nn.ReLU(), nn.Conv2d(self.d, 3, 1)
        )

    def forward(self, x, steps=8):
        for t in range(steps):
            # 1. Local NCA update on all channels
            dx = self.local_nca(x)  # (B, C, H, W)

            # 2. Graph MP on hidden only
            hidden = x[:, 4:]  # (B, D, H, W)
            h_graph = self.graph_mp(hidden)

            # 3. Combine updates for hidden
            dx_hidden = dx[:, 4:] + self.beta * h_graph  # (B, D, H, W)
            dx_other = dx[:, :4]  # alive + RGB (B, 4, H, W)

            # 4. Update state 
            new_hidden = self.elu(x[:, 4:] + self.alpha * dx_hidden)  # (B, D, H, W)
            other = x[:, :4] + self.alpha * dx_other  # (B, 4, H, W)

            # 5. Alive + RGB decode from hidden 
            alive_now = torch.sigmoid(self.alive_update(new_hidden))  # (B, 1, H, W)
            rgb_now = torch.sigmoid(self.rgb_decode(torch.cat([new_hidden, alive_now], dim=1)))
            rgb_now = rgb_now * alive_now  # (B, 3, H, W)

            # 6. Rebuild the full state
            x = torch.cat([
                alive_now,  # (B, 1, H, W)
                rgb_now,  # (B, 3, H, W)
                new_hidden  # (B, D, H, W)
            ], dim=1)

        return x