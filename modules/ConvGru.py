import torch
import torch.nn as nn


class ConvGRUCell(nn.Module):
    """
    Convolutional GRU cell for patch‐level state updates in GraphNCA
    This implements the standard GRU gating mechanism using 2D convolutions to
    preserve spatial structure:

    1. conv_gates: a single convolution producing 2 * hidden_channels outputs
       - The first half (r) after sigmoid is the reset gate (introduce memory)
       - The second half (z) after sigmoid is the update gate (introduce new information)

    2. conv_cand: a convolution producing hidden_channels outputs for the candidate new state

    On each forward pass:
    a) Concatenate the input x (aggregated messages) and previous hidden state h_prev along channels
    b) Apply conv_gates, split into r and z, apply sigmoid to get gate maps
    c) Compute the candidate state c by applying conv_cand to the concatenation of x and r*h_prev, then tanh
    d) Combine old state and candidate via h_new = (1 - z)*h_prev + z*c
    """
    def __init__(self, input_channels, hidden_channels, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        # Combined convolution for reset and update gates
        self.conv_gates = nn.Conv2d(
            input_channels + hidden_channels,
            2 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=True
        )
        # Convolution for candidate hidden state
        self.conv_cand = nn.Conv2d(
            input_channels + hidden_channels,
            hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=True
        )
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x, h_prev):
        """
        Args:
            x: (B, input_channels, H, W) input tensor (aggregated messages)
            h_prev: (B, hidden_channels, H, W) previous hidden state
        Returns:
            h_new: (B, hidden_channels, H, W) updated hidden state
        """
        # Concatenate input and previous state
        combined = torch.cat([x, h_prev], dim=1)
        # Compute reset and update gates in one convolution
        gates = self.conv_gates(combined)  # (B, 2*hidden_channels, H, W)
        r, z = gates.chunk(2, dim=1)  # each (B, hidden_channels, H, W)
        r = self.sigmoid(r)
        z = self.sigmoid(z)

        # Compute candidate hidden state
        combined_cand = torch.cat([x, r * h_prev], dim=1)
        c = self.tanh(self.conv_cand(combined_cand))  # (B, hidden_channels, H, W)

        # Final hidden state update
        h_new = (1 - z) * h_prev + z * c
        return h_new
