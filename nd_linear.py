import torch
import torch.nn as nn
import numpy as np

class NdLinear(nn.Module):
    def __init__(self, in_features, out_features, rank=4):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        
        # Initialize low-rank matrices
        self.U = nn.Parameter(torch.randn(in_features, rank) / np.sqrt(rank))
        self.V = nn.Parameter(torch.randn(rank, out_features) / np.sqrt(rank))
        
        # Optional bias
        self.bias = nn.Parameter(torch.zeros(out_features))
        
    def forward(self, x):
        # Low-rank multiplication
        weight = torch.matmul(self.U, self.V)
        return torch.matmul(x, weight) + self.bias