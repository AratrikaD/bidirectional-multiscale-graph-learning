import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, SAGEConv, HeteroConv
import torch

class EventGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim=64):
        super().__init__()
        self.conv1 = GATConv(in_dim, hidden_dim, heads=2, concat=True)
        self.conv2 = GATConv(hidden_dim*2, hidden_dim, heads=2, concat=False)
    def forward(self, data):
        x = F.elu(self.conv1(data.x, data.edge_index))
        x = self.conv2(x, data.edge_index)
        return x

class CommunityGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim=64):
        super().__init__()
        self.convs = HeteroConv({
            ('neighborhood','geographical','neighborhood'): SAGEConv(in_dim, hidden_dim)
        }, aggr='sum')
    def forward(self, data):
        return self.convs(data.x_dict, data.edge_index_dict)['neighborhood']

class FusionMugRep(nn.Module):
    def __init__(self, event_dim, comm_dim, hidden_dim, num_districts):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(event_dim+comm_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU()
        )
        # If only one district → one head
        self.heads = nn.ModuleList([nn.Linear(hidden_dim//2, 1) for _ in range(num_districts)])

    def forward(self, h_e, h_c, district_idx, neigh_idx):
        h = torch.cat([h_e, h_c[neigh_idx]], dim=-1)
        h = self.mlp(h)
        if len(self.heads) == 1:
            return self.heads[0](h).squeeze(-1)
        else:
            out = torch.zeros(h.size(0), device=h.device)
            for d, head in enumerate(self.heads):
                mask = (district_idx==d)
                if mask.sum()>0:
                    out[mask] = head(h[mask]).squeeze(-1)
            return out


class MugRepModel(nn.Module):
    def __init__(self, event_in, comm_in, hidden_dim, num_districts):
        super().__init__()
        self.event_gnn = EventGNN(event_in, hidden_dim)
        self.comm_gnn = CommunityGNN(comm_in, hidden_dim)
        self.fusion = FusionMugRep(hidden_dim, hidden_dim, hidden_dim, num_districts)
    def forward(self, event_graph, comm_graph):
        h_e = self.event_gnn(event_graph)
        h_c = self.comm_gnn(comm_graph)
        return self.fusion(h_e, h_c, event_graph.district, event_graph.neighborhood)
