import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.data import HeteroData
from sklearn.preprocessing import StandardScaler
from src.graphs.create_edges import get_transaction_edges

def load_graph_data(neighborhood_file, edge_file):
    neighborhood_df = pd.read_csv(neighborhood_file)
    edges = pd.read_csv(edge_file, index_col=0)
    edge_index = adjacency_matrix_to_edge_index(edges)
    node_features = {}
    for year, group in neighborhood_df.groupby('YEAR'):
        features = torch.tensor(group.drop(['BUURTCODE', 'YEAR'], axis=1).values, dtype=torch.float32)
        node_features[year] = features

    return node_features, edge_index

def load_transaction_data(transaction_file):
    df = pd.read_csv(transaction_file)
    df['DATUM'] = pd.to_datetime(df['DATUM'])
    df.sort_values('DATUM', inplace=True)

    df['YEAR'] = df['DATUM'].dt.year
    df['MONTH'] = df['DATUM'].dt.month
    df.drop([ "DATUM"], axis=1, inplace=True)

    return df

def adjacency_matrix_to_edge_index(adj_matrix):

    adj_matrix = adj_matrix.to_numpy()

    edge_list = torch.nonzero(torch.tensor(adj_matrix), as_tuple=False)

    edge_index = torch.cat([edge_list, edge_list.flip(0)], dim=0)

    edge_index = edge_index.unique(dim=0)

    return edge_index.T

def build_hetero_graph(transaction_df, neighborhood_df, test_mask, neighbor_edge_index):
    data = HeteroData()
    scaler = StandardScaler()

    trans_x = scaler.fit_transform(transaction_df.drop(columns=['LOG_KOOPSOM', 'BUURTCODE', "TRANSID"]).values)
    neigh_x = neighborhood_df.drop(columns=['BUURTCODE']).values

    data['transaction'].x = torch.tensor(trans_x, dtype=torch.float)
    data['transaction'].y = torch.tensor(transaction_df['LOG_KOOPSOM'].values, dtype=torch.float).unsqueeze(1)
    data['transaction'].predict_mask = test_mask
    data['neighborhood'].x = torch.tensor(neigh_x, dtype=torch.float)

    # Intra-level edges
    edge_index = get_transaction_edges(transaction_df)
    data['transaction', 'to', 'transaction'].edge_index = edge_index
    data['neighborhood', 'to', 'neighborhood'].edge_index = neighbor_edge_index

    neigh_id_map = {nid: idx for idx, nid in enumerate(neighborhood_df['BUURTCODE'])}
    trans_to_macro = [neigh_id_map[nid] for nid in transaction_df['BUURTCODE']]
    data['transaction'].neighborhood_index = torch.tensor(trans_to_macro, dtype=torch.long)

    # Inter-level edges
    # Build edges [2, num_edges] where each transaction node is connected to its neighborhood
    src = torch.tensor(trans_to_macro, dtype=torch.long)       # neighborhood indices
    dst = torch.arange(len(transaction_df), dtype=torch.long)  # transaction node indices

    # Create edges in both directions
    data['neighborhood', 'contains', 'transaction'].edge_index = torch.stack([src, dst], dim=0)
    data['transaction', 'belongs_to', 'neighborhood'].edge_index = torch.stack([dst, src], dim=0)

    return data