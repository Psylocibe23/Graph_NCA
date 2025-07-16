import torch
import torch.nn as nn
from modules.attention import ConcatMLP
from modules.gating import GateConv
from modules.ConvGru import ConvGRUCell



class LocalCA(nn.Module):
    """Local pixel wise update rule for the NCAs"""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True)
        self.act = nn.ELU(inplace=False)

    def forward(self, x):
        return self.act(self.conv(x))



class GraphNCA(nn.Module):
    """
    Graph Attention Neural Cellular Automata network:
    - Split the canvas into P*P nodes (patches)
    - Each node is a NCA with internal state (C, H, W)
    - Performs local update, message passing, attention, gating, and convolutional GRU graph-level state update
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
        # Local update rule
        self.local_ca = LocalCA(C)
        # 1x1 convolution to mix local and graph level information
        self.mix = nn.Conv2d(2*C, C, kernel_size=1)
        print('LocalCA conv weight stats:', self.local_ca.conv.weight.data.min().item(), self.local_ca.conv.weight.data.max().item(), self.local_ca.conv.weight.data.mean().item())
        print('LocalCA conv bias stats:', self.local_ca.conv.bias.data.min().item(), self.local_ca.conv.bias.data.max().item(), self.local_ca.conv.bias.data.mean().item())




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
        X = torch.clamp(X, -10, 10)
        
        # Initialize node state
        for t in range(self.K):
            X_prev = X  # Avoid updating X until the end of the loop
            # Local CA update (pixel-wise) within each patch
            X_local = []
            for i in range(N):
                h_local = self.local_ca(X_prev[:, i])  # Local convolution update
                X_local.append(h_local)
            X_local = torch.stack(X_local, dim=1)  # (B, N, C, H, W)


            if torch.isnan(X_local).any() or torch.isinf(X_local).any():
                print(f'NaN/Inf detected at CA state update (t={t})')
                raise RuntimeError('Runtime error') 

            # Project (B, N, C, H, W) --> (B, N, d, H, W)
            Q = self.f_Q(X.view(B*N, C, self.H, self.W)).view(B, N, self.d, self.H, self.W)
            K_proj = self.f_K(X.view(B*N, C, self.H, self.W)).view(B, N, self.d, self.H, self.W)
            # Message projection
            M = self.f_M(X.view(B*N, C, self.H, self.W)).view(B, N, C, self.H, self.W)

            # Spatial pooling for queries and keys (obtain global summary)
            q_vec = Q.mean([-2,-1])  # (B, N, d), average over last two dimension (-2) for each patch (-1)
            k_vec = K_proj.mean([-2,-1])  # (B, N, d)

            # Message passing
            msgs = [None] * N
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
                    if (t in [0, self.K-1]) and (i in [0, 1]) and (j_ in [0, 1]):
                        print(f"[t={t}] Edge {i}->{j_}: e_ij={e_ij.mean().item():.4f}, "
                  f"msg mean={msg.mean().item():.4f}, m_agg mean={m_agg.mean().item():.4f}")
                if (t in [0, self.K-1]) and (i in [0, 1]):
                    print(f"[t={t}] Node {i} m_agg mean after all messages: {m_agg.mean().item():.4f}, std: {m_agg.std().item():.4f}")
                msgs[i] = m_agg
            M_agg = torch.stack(msgs, dim=1)  # (B, N, C, H, W)

            # ConvGRU update (graph-level)
            X_graph = []
            for i in range(N):
                h = self.cell(M_agg[:, i], X_prev[:, i])
                X_graph.append(h)
            X_graph = torch.stack(X_graph, dim=1)

            if torch.isnan(X_graph).any() or torch.isinf(X_graph).any():
                print(f'NaN/Inf detected at graph state update (t={t})')
                raise RuntimeError('Runtime error')

            # 4. Concatenate local and graph, then mix
            X_new = []
            for i in range(N):
                concat = torch.cat([X_local[:, i], X_graph[:, i]], dim=1)  # (B, 2C, H, W)
                mixed = self.mix(concat)  # (B, C, H, W)
                X_new.append(mixed)
            X = torch.stack(X_new, dim=1)
            if torch.isnan(X).any() or torch.isinf(X).any():
                print(f'NaN/Inf detected at ConvGRU state update (t={t})')
                raise RuntimeError('Runtime error') 

        # After K iterations reassemble the canvas for the read-out (autograd-safe, no inplace ops)
        X = X.view(B, P, P, C, self.H, self.W)
        X = X.permute(0, 3, 1, 4, 2, 5)
        out = X.contiguous().view(B, C, PH, PW)
        # Enforce alive mask on RGB channels (channels 1-3) for the whole batch
        alive_mask = out[:, 0:1, :, :].clamp(0, 1)
        out_rgb = out[:, 1:4, :, :] * alive_mask
        if out.shape[1] > 4:
            out_rest = out[:, 4:, :, :]
            out = torch.cat([out[:, 0:1, :, :], out_rgb, out_rest], dim=1)
        else:
            out = torch.cat([out[:, 0:1, :, :], out_rgb], dim=1)
        return out
    