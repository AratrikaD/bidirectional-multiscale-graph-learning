import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GATv2Conv, GatedGraphConv, GCNConv, Linear


class MacroEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.gate_W = nn.Linear(hidden_channels, hidden_channels)
        self.gate_U = nn.Linear(hidden_channels, hidden_channels)

    def forward(self, x, edge_index, prev_embedding=None):
        h_t = F.relu(self.conv1(x, edge_index))
        h_t = self.conv2(h_t, edge_index)

        if prev_embedding is not None:
            alpha = torch.sigmoid(self.gate_W(h_t) + self.gate_U(prev_embedding))
            h_tilde = alpha * h_t + (1 - alpha) * prev_embedding
        else:
            h_tilde = h_t

        return h_tilde, h_t  # Return both gated and raw macro embeddings