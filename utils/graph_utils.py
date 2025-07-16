import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import numpy as np



def create_touching_edges(P):
    """
    Returns a list of directed edges (i -> j) for a PxP grid
    each node i connects with its 8 touching neighbors j (nodes are patches)
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


def create_pixel_graph_edges(H, W):
    """
    Returns a list of directed edges (j -> i) connecting pixel j
    with its 8 neighbors
    """
    edges = []
    for row in range(H):
        for col in range(W):
            i = row * W + col
            for d_row in [-1, 0, 1]:
                for d_col in [-1, 0, 1]:
                    if d_row == 0 and d_col == 0: continue
                    n_row = row + d_row
                    n_col = col + d_col
                    if 0 <= n_row < H and 0 <= n_col < W:
                        j = n_row * W + n_col
                        edges.append((i, j))
    return edges
