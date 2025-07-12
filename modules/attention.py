import torch
import torch.nn as nn


class ConcatMLP(nn.Module):
    """
    Concatenate + MLP to compute edge affinity score
    Given two embeddings q_i (queries) and k_j (keys) of dimension d, the MLP
    takes as input [q_i||k_j] and outputs a single score per edge e_ij
    """
    def __init__(self, d, hidden=64, layers=2):
        super().__init__()
        # For 2 layers (2*d -> hidden -> 1) 
        if layers == 2:
            self.mlp = nn.Sequential(
                nn.Linear(2*d, hidden),
                nn.ReLU(inplace=True),    
                nn.Linear(hidden, 1)                   
            )
        # For 3 layers (2*d -> hidden -> hidden -> 1)
        elif layers == 3:
            self.mlp = nn.Sequential(
                nn.Linear(2*d, hidden),
                nn.ReLU(inplace=True),    
                nn.Linear(hidden, hidden)                   ,
                nn.ReLU(inplace=True),
                nn.Linear(hidden, 1)
            )
        else:
            raise NotImplementedError("This version only accepts 2 or 3 layers")

    def forward(self, q, k):
        x = torch.cat([q, k], dim=-1)  # (batch, 2*d)
        return self.mlp(x)
    