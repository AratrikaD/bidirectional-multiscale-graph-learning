import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np

from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv, GATConv, Linear
from torch_geometric.loader import DataLoader
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

def create_hetero_graph(tx_df, nh_df, year, month, tx_features, nh_features):
    data = HeteroData()

    # Filter window
    window_tx = tx_df[(tx_df['YEAR'] == year) & (tx_df['MONTH'] == month)].copy()
    window_nh = nh_df[nh_df['YEAR'] == year].copy()

    # Scaling (per window)
    tx_scaler = StandardScaler()
    nh_scaler = StandardScaler()

    data['transaction'].x = torch.tensor(tx_scaler.fit_transform(window_tx[tx_features]), dtype=torch.float)
    data['neighborhood'].x = torch.tensor(nh_scaler.fit_transform(window_nh[nh_features]), dtype=torch.float)

    # Map BUURTCODEs to node IDs
    buurtcode_to_txid = dict(enumerate(window_tx['BUURTCODE']))
    buurtcode_to_nhid = {code: i for i, code in enumerate(window_nh['BUURTCODE'])}
    tx_to_nh_idx = [buurtcode_to_nhid[bc] for bc in window_tx['BUURTCODE']]

    # Transaction → transaction edge (kNN or dummy: fully connect neighborhood-local)
    tx_edge_index = get_transaction_edges(window_tx)  # User-defined (or spatial/temporal kNN)
    data['transaction', 'transacts_with', 'transaction'].edge_index = torch.tensor(tx_edge_index, dtype=torch.long)

    # Neighborhood → neighborhood edges (precomputed graph)
    nh_edge_index = get_neighborhood_edges(window_nh)  # User-defined (based on adjacency/spatial)
    data['neighborhood', 'connected_to', 'neighborhood'].edge_index = torch.tensor(nh_edge_index, dtype=torch.long)

    # Transaction → neighborhood (belongs_to)
    src = torch.arange(len(window_tx))
    tgt = torch.tensor(tx_to_nh_idx)
    data['transaction', 'belongs_to', 'neighborhood'].edge_index = torch.stack([src, tgt])

    # Neighborhood → transaction (influences)
    data['neighborhood', 'influences', 'transaction'].edge_index = torch.stack([tgt, src])

    # Targets
    data['transaction'].y = torch.tensor(window_tx['PRICE'].values, dtype=torch.float)

    return data

class SpatioTemporalGate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.spatial_proj = Linear(dim, dim)
        self.temporal_proj = Linear(dim, dim)
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )

    def forward(self, spatial_feat, temporal_feat):
        s_proj = self.spatial_proj(spatial_feat)
        t_proj = self.temporal_proj(temporal_feat)
        gate_weight = self.gate(torch.cat([s_proj, t_proj], dim=-1))
        return gate_weight * s_proj + (1 - gate_weight) * t_proj
class HierarchicalHeteroGNN(nn.Module):
    def __init__(self, tx_in, nh_in, hidden_dim):
        super().__init__()
        self.tx_conv = SAGEConv(tx_in, hidden_dim)

        self.nh_spatial_conv = SAGEConv(nh_in, hidden_dim)
        self.nh_temporal_conv = GATConv(hidden_dim, hidden_dim)
        self.st_gate = SpatioTemporalGate(hidden_dim)

        self.tx_to_nh = Linear(hidden_dim, hidden_dim)
        self.nh_to_tx = Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, data: HeteroData):
        tx_x = F.relu(self.tx_conv(data['transaction'].x, data['transaction', 'transacts_with', 'transaction'].edge_index))

        # Bottom-up
        tx_msg = self.tx_to_nh(tx_x)
        neigh_ids = data['transaction', 'belongs_to', 'neighborhood'].edge_index[1]
        nh_agg = torch.zeros(data['neighborhood'].num_nodes, tx_msg.shape[1], device=tx_x.device)
        nh_agg.index_add_(0, neigh_ids, tx_msg)

        # Neighborhood update
        nh_spatial = F.relu(self.nh_spatial_conv(data['neighborhood'].x, data['neighborhood', 'connected_to', 'neighborhood'].edge_index))
        nh_temporal = self.nh_temporal_conv(nh_spatial, data['neighborhood', 'connected_to', 'neighborhood'].edge_index)
        nh_x = F.relu(self.st_gate(nh_spatial, nh_temporal + nh_agg))

        # Top-down
        tx_idx = data['neighborhood', 'influences', 'transaction'].edge_index[1]
        nh_to_tx = self.nh_to_tx(nh_x)
        tx_enhanced = tx_x + nh_to_tx[tx_idx]

        return self.out(tx_enhanced).squeeze()
def train_model(data_list, model, epochs=20, batch_size=1, lr=1e-3):
    loader = DataLoader(data_list, batch_size=batch_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            loss = loss_fn(out, batch['transaction'].y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"[Epoch {epoch+1}] Loss: {total_loss / len(loader):.4f}")

def get_transaction_edges(df):
    # Example: connect every property in the same BUURTCODE (fully connected neighborhood subgraph)
    edges = []
    grouped = df.groupby('BUURTCODE').groups
    for nodes in grouped.values():
        for i in nodes:
            for j in nodes:
                if i != j:
                    edges.append((i, j))
    return np.array(edges).T

def get_neighborhood_edges(df):
    # Dummy: connect all neighborhoods to simulate adjacency (replace with spatial matrix)
    # N = len(df)
    # return np.array([(i, j) for i in range(N) for j in range(N) if i != j]).T
    return df.to_numpy

# 1. Load your tx_df, nh_df
# 2. Define feature columns
# def run_exp():
#     tx_features = ['AREA', 'ROOMS', 'AGE']
#     nh_features = ['POP_DENSITY', 'AVG_INCOME', 'CRIME_RATE']

#     # 3. Create graphs per time window
#     graph_list = []
#     for (year, month) in [(2021, 1), (2021, 2), (2021, 3)]:  # Sliding windows
#         g = create_hetero_graph(tx_df, nh_df, year, month, tx_features, nh_features)
#         graph_list.append(g)

#     # 4. Train
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model = HierarchicalHeteroGNN(tx_in=len(tx_features), nh_in=len(nh_features), hidden_dim=64).to(device)

#     train_model(graph_list, model, epochs=30, batch_size=1)
# Re-execute the previously shared code after environment reset
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import HeteroData
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict

# Step 1: Graph Construction
def construct_transaction_graph(transactions_df, k=5):
    data = HeteroData()

    # Add transaction nodes
    transaction_features = transactions_df.drop(columns=["transaction_id", "BUURTCODE", "YEAR", "MONTH", "price"])
    scaler = StandardScaler()
    transaction_x = scaler.fit_transform(transaction_features)
    data["transaction"].x = torch.tensor(transaction_x, dtype=torch.float)

    # KNN graph edges (e.g., based on coordinates or embeddings)
    coords = transactions_df[["lat", "lon"]].values
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(coords)
    indices = nbrs.kneighbors(coords, return_distance=False)

    edge_index = []
    for i, neighbors in enumerate(indices):
        for j in neighbors:
            if i != j:
                edge_index.append([i, j])
    data["transaction", "interacts", "transaction"].edge_index = torch.tensor(edge_index).t().contiguous()

    # Store metadata
    data["transaction"].year = torch.tensor(transactions_df["YEAR"].values, dtype=torch.long)
    data["transaction"].month = torch.tensor(transactions_df["MONTH"].values, dtype=torch.long)
    data["transaction"].price = torch.tensor(transactions_df["price"].values, dtype=torch.float)
    data["transaction"].BUURTCODE = transactions_df["BUURTCODE"].values

    return data, scaler

# Step 2: Create sliding windows
def split_sliding_window(df, window_years=3):
    min_year, max_year = df["YEAR"].min(), df["YEAR"].max()
    windows = []
    for start in range(min_year, max_year - window_years):
        train_years = list(range(start, start + window_years))
        test_year = start + window_years
        train_df = df[df["YEAR"].isin(train_years)]
        test_df = df[df["YEAR"] == test_year]
        windows.append((train_df, test_df))
    return windows

# Step 3: Construct heterogeneous graph with macro nodes
def add_macro_nodes(data, neighborhoods_df, year):
    macro_feats = neighborhoods_df[neighborhoods_df["YEAR"] == year]
    macro_x = macro_feats.drop(columns=["BUURTCODE", "YEAR"]).values
    data["neighborhood"].x = torch.tensor(macro_x, dtype=torch.float)

    # Map transactions to neighborhoods
    buurt_to_index = {b: i for i, b in enumerate(macro_feats["BUURTCODE"].values)}
    neighborhood_edge_index = [[], []]

    for i, buurt_code in enumerate(data["transaction"].BUURTCODE):
        if buurt_code in buurt_to_index:
            j = buurt_to_index[buurt_code]
            neighborhood_edge_index[0].append(i)  # transaction -> neighborhood
            neighborhood_edge_index[1].append(j)

    data["transaction", "in", "neighborhood"].edge_index = torch.tensor(neighborhood_edge_index, dtype=torch.long)

    return data
