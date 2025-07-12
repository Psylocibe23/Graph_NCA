import torch
import torch.nn as nn
from modules.attention import ConcatMLP
from modules.gating import GateConv
from modules.ConvGru import ConvGRUCell


class GraphNCA(nn.Module):
    """
    Graph Attention Neural Cellular Automata network:
    - Split the canvas into P*P nodes (patches)
    - Each node is a NCA with internal state (C, H, W)
    - Performs message passing, attention, gating, and convolutional GRU state updates
    """
    def __init__(self, C, H, W, d, K, attention_hidden, attention_layers):
        super().__init__()
        self.C = C  # Hidden channels
        self.H = H
        self.W = W
        self.d = d  # queries/keys space dimension
        self.K = K  # Number of iterations

        self.f_Q = nn.Conv2d(C, d, kernel_size=1)  # Projection onto queries space
        self.f_K = nn.Conv2d(C, d, kernel_size=1)  # Projection onto keys space
        self.f_M = nn.Conv2d(C, C, kernel_size=3, padding=1)  # Projection onto messages space

        # Attention
        self.attention = ConcatMLP(d, hidden=attention_hidden, layers=attention_layers)
        # Gating
        self.gate = GateConv(out_channels=C, kernel_size=3)
        # ConvGRU
        self.cell = ConvGRUCell(input_channels=C, hidden_channels=C, kernel_size=3)


    def forward(self, seed, edge_index):
        """
        seed: (B, C, PH, PW) input canvas
        edge_index: list of edges (i, j)
        returns: final canvas after K iterations (B, C, PH, PW)
        """
        B, C, PH, PW = seed.shape
        assert C == self.C and PH % self.H == 0 and PW % self.W == 0
        P = PH // self.H
        N = P*P  # nodes

        # Split canvas into pathces of dimension (B, C, H, W)
        patches = []
        for p in range(P):
            for q in range(P):
                patch = seed[:, :, p*self.H:(p+1)*self.H, q*self.W:(q+1)*self.W]
                patches.append(patch)
        X = torch.stack(patches, dim=1)  # (B, N, C, H, W)

        # Initialize node state
        for t in range(self.K):
            # Project (B, N, C, H, W) --> (B, N, d, H, W)
            Q = self.f_Q(X.view(B*N, C, self.H, self.W)).view(B, N, self.d, self.H, self.W)
            K_proj = self.f_K(X.view(B*N, C, self.H, self.W)).view(B, N, self.d, self.H, self.W)
            # Message projection
            M = self.f_M(X.view(B*N, C, self.H, self.W)).view(B, N, C, self.H, self.W)

            # Spatial pooling for queries and keys (obtain global summary)
            q_vec = Q.mean([-2,-1])  # (B, N, d), average over last two dimension (-2) for each patch (-1)
            k_vec = K_proj.mean([-2,-1])  # (B, N, d)

            # Message passing
            msgs = [torch.zeros_like(X)] * N
            for i in range(N):
                m_agg = torch.zeros(B, C, self.H, self.W, device=X.device)
                for e, (i_, j_) in enumerate(edge_index):
                    if i_ != i:
                        continue
                    # Compute affinity scalar for this edge
                    e_ij = self.attention(q_vec[:, i, :], k_vec[:, j_, :])  # (B, 1)
                    e_map = e_ij.view(B, 1, 1, 1).expand(-1, 1, self.H, self.W)  # (B, 1, H, W)
                    G_ij = self.gate(e_map)  # (B, C, H, W)
                    M_j = M[:, j_, :, :, :]  # (B, C, H, W)
                    # Gated message
                    msg = G_ij * M_j
                    m_agg += msg
                msgs[i] = m_agg
            M_agg = torch.stack(msgs, dim=1)  # (B, N, C, H, W)

            # ConvGRU updates
            X_new = []
            for i in range(N):
                h = self.cell(M_agg[:, i], X[:, i])
                X_new.append(h)
            X = torch.stack(X_new, dim=1)

        # After K iterations ressemble the canvas for the red-out
        out = torch.zeros(B, C, PH, PW, device=X.device)
        for n in range(N):
            p, q = divmod(n, P)  # n=p*p+q
            out[:, :, p*self.H:(p+1)*self.H, q*self.W:(q+1)*self.W] = X[:, n, :, :, :]
        return out
    