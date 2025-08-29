import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv,  SAGEConv, Linear, GraphConv


class MicroEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        
        self.conv = SAGEConv(in_channels, hidden_channels)
        self.lin = Linear(hidden_channels, hidden_channels)

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = F.relu(x)

        x = self.lin(x)
        return x
