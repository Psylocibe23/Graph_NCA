import torch


def create_touching_edges(P):
    """
    Returns a list of directed edges (i -> j) for a PxP grid
    each node i connects with its 8 touching neighbors j
    """
    edges = []
    for row in range(P):
        for col in range(P):
            i = row*P + col  # Node index

            # Consider all the 8 neighbors (d_row, d_col)
            for d_row in (-1, 0, 1):
                for d_col in (-1, 0, 1):
                    # Skip (0,0) which is node i itself
                    if d_row == 0 and d_col == 0:
                        continue
                    n_row = row + d_row
                    n_col = col + d_col 

                    # Stay within grid boundary
                    if 0 <= n_row < P and 0 <= n_col < P:
                        j = n_row*P + n_col  # Neighbor index
                        edges.append((i,j))
    return edges


import matplotlib.pyplot as plt
import os

def save_ca_channels(state, epoch, save_dir, vmax=None, vmin=None):
    # state: (1, C, H, W) or (C, H, W)
    state = state.detach().cpu().squeeze(0)  # (C, H, W)
    C = state.shape[0]
    os.makedirs(save_dir, exist_ok=True)
    for ch in range(C):
        channel_img = state[ch].numpy()
        # Optionally normalize to 0-1 for visualization
        channel_img = (channel_img - channel_img.min()) / (channel_img.max() - channel_img.min() + 1e-8)
        plt.imsave(f"{save_dir}/epoch{epoch:04d}_ch{ch}.png", channel_img, cmap="gray", vmin=0, vmax=1)
        plt.imsave(f"{save_dir}/epoch{epoch:04d}_alive_ch0.png", channel_img, cmap="gray", vmin=0, vmax=1)



import numpy as np

def save_channel_grid(state, epoch, save_dir):
    state = state.detach().cpu().squeeze(0)  # (C, H, W)
    C, H, W = state.shape
    grid_size = int(np.ceil(C ** 0.5))
    # Pad with zeros if not a perfect square
    pad = grid_size ** 2 - C
    if pad > 0:
        state = torch.cat([state, torch.zeros(pad, H, W)], dim=0)
    # Normalize for visualization
    state = (state - state.min()) / (state.max() - state.min() + 1e-8)
    # Arrange into grid
    grid = torch.cat([torch.cat([state[i*grid_size + j] for j in range(grid_size)], dim=1) for i in range(grid_size)], dim=0)
    plt.imsave(f"{save_dir}/epoch{epoch:04d}_channel_grid.png", grid.numpy(), cmap="gray", vmin=0, vmax=1)

