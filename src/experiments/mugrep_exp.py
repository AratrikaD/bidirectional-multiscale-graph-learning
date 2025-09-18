import os
import torch
import torch.nn as nn
from src.utils.train_mugrep import train_sliding_window_mugrep
from src.models.mugrep import MugRepModel

import os
import pandas as pd
import numpy as np 
import torch
from torch_geometric.data import Data, HeteroData
from sklearn.preprocessing import StandardScaler

# -------------------------------
# Load community (neighborhood) features + adjacency
# -------------------------------
def load_neighborhood_data(features_path, edge_path):
    nb = pd.read_csv(features_path)   # BUURTCODE, YEAR, feat...

    # Collapse to latest year snapshot per neighborhood
    nb_latest = nb.sort_values(['BUURTCODE','YEAR']).groupby('BUURTCODE', as_index=False).last()
    nb_latest = nb_latest.rename(columns={'BUURTCODE':'neighborhood_id'})

    # Normalize neighborhood IDs
    nb_latest = nb_latest.sort_values('neighborhood_id').reset_index(drop=True)
    idmap = {nid:i for i,nid in enumerate(nb_latest['neighborhood_id'])}
    idmap_rev = {i:nid for nid,i in idmap.items()}

    # === Assign dummy district ===
    nb_latest['district_id'] = 0
    dmap, dmap_rev = {0:0}, {0:0}

    # Feature matrix
    feat_cols = [c for c in nb_latest.columns if c not in ['neighborhood_id','district_id','YEAR']]
    scaler_nb = StandardScaler().fit(nb_latest[feat_cols].fillna(0.0))
    x_nb = torch.tensor(scaler_nb.transform(nb_latest[feat_cols].fillna(0.0)), dtype=torch.float)

    comm_graph = HeteroData()
    comm_graph['neighborhood'].x = x_nb
    comm_graph['neighborhood'].district = torch.tensor(nb_latest['district_id'].to_numpy(), dtype=torch.long)

    # Build geographical adjacency edges
    adj = pd.read_csv(edge_path, index_col=0).to_numpy()  # if CSV has header+index
    # or: adj = np.loadtxt(edge_path, delimiter=",")       # if plain numeric matrix

    src, dst = np.nonzero(adj)  # all pairs where adjacency > 0
    edge_index = torch.tensor(np.vstack([src, dst]), dtype=torch.long)

    comm_graph['neighborhood','geographical','neighborhood'].edge_index = edge_index

    return comm_graph, nb_latest, idmap, idmap_rev, dmap, dmap_rev



def run_mugrep_exp(data_path, hyperparameters):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    neighborhood_features_path = os.path.join(data_path, "all_neighborhood_features_rotterdam.csv")
    edge_path = os.path.join(data_path, "rotterdam_adj_2023.csv")
    transaction_path = os.path.join(data_path, "synthetic_transactions.csv")

    comm_graph, nb_latest, idmap, idmap_rev, dmap, dmap_rev = load_neighborhood_data(neighborhood_features_path, edge_path)
    comm_graph = comm_graph.to(device)

    transactions = pd.read_csv(transaction_path)

    event_feature_dim = transactions.drop(
        columns=["TRANSID","DATUM","LOG_KOOPSOM","BUURTCODE"]
    ).select_dtypes(include=[np.number]).shape[1]
    comm_feature_dim = comm_graph['neighborhood'].x.shape[1]
    num_districts = 1

    model = MugRepModel(event_feature_dim, comm_feature_dim,
                    hidden_dim=64,
                    num_districts=num_districts).to(device)


    optimizer = torch.optim.Adam(model.parameters(),
                                 lr= 0.0001,
                                 weight_decay= 1e-5)
    criterion = nn.MSELoss()

    return train_sliding_window_mugrep(
        model, optimizer, criterion,
        transactions, comm_graph,
        idmap_rev, dmap_rev, nb_latest,
        window_months=61,
        epochs=100,
        batch_size=64,
    )
