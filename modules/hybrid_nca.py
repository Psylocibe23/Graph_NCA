import torch
import torch.nn as nn
from modules.graph_augmentation import PixelGraphMessagePassing

class HybridPixelGraphNca(nn.Module):
    """
    Pixelwise NCA with explicit channel separation:
    - Channel 0: alive
    - Channels 1-3: RGB (only decoded from hidden at each step)
    - Channels 4+: hidden (evolves over time)
    """
    def __init__(self, channels, edge_index, alpha=1.0, beta=1.0):
        super().__init__()
        self.C = channels
        self.edge_index = edge_index
        self.alpha = alpha
        self.beta = beta
        self.d = self.C - 4  # Do not consider alive and RGB channels

        hidden_dim = channels - 4
        assert hidden_dim > 0, "Need at least 5 channels: 1 alive, 3 rgb, >=1 hidden"

        # Only operate on hidden channels for state evolution
        self.hidden_update = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.graph_mp = PixelGraphMessagePassing(hidden_dim)  # Only for hidden
        # allow alive mask update from hidden
        self.alive_update = nn.Conv2d(hidden_dim, 1, 1)
        # RGB decode (from hidden + alive), only 1x1
        self.rgb_decode = nn.Sequential(
            nn.Conv2d(self.d+1, self.d, 1), nn.ReLU(),
            nn.Conv2d(self.d, 3, 1)
        )
        self.elu = nn.ELU()

    def forward(self, x, steps=8):
        B, C, H, W = x.shape
        for t in range(steps):
            alive = x[:, 0:1]  # [B, 1, H, W]
            rgb   = x[:, 1:4]  # [B, 3, H, W]
            hidden = x[:, 4:]  # [B, D, H, W]

            # 1. Hidden update: local + graph, NO RGB
            h_local = self.hidden_update(hidden)
            h_graph = self.graph_mp(hidden)
            hidden_new = hidden + self.alpha * h_local + self.beta * h_graph
            hidden_new = self.elu(hidden_new)

            # 2. Alive update 
            alive_new = torch.sigmoid(self.alive_update(hidden_new))

            # 3. RGB decode: only from hidden+alive
            rgb_new = torch.sigmoid(self.rgb_decode(torch.cat([hidden_new, alive_new], dim=1)))
            # Multiply by alive to mask dead cells
            rgb_new = rgb_new * alive_new

            # 4. Reconstruct state
            x = torch.cat([alive_new, rgb_new, hidden_new], dim=1)

        return x
