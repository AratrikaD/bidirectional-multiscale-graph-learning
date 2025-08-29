import pandas as pd
import networkx as nx
from sklearn.neighbors import NearestNeighbors
import torch.nn as nn
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import HeteroData
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict
import os
from src.utils.rolling_trainer import RollingTrainer
from src.utils.rolling_window_graph import RollingGraphWindowDataset
from src.models.hstgnn import HierarchicalSpatioTemporalGNN

def load_data(data_path):
    """
    Load all required data from a CSV file.
    """
    transactions_data_path = os.path.join(data_path, "rotterdam_transaction_data.csv")
    neighborhoods_data_path = os.path.join(data_path, "all_neighborhood_features_rotterdam.csv")
    neighborhood_adjacency_data_path = os.path.join(data_path, "rotterdam_adj_2023.csv")

    transaction_df = pd.read_csv(transactions_data_path)
    neighborhood_df = pd.read_csv(neighborhoods_data_path)
    neighborhood_adjacency_df = pd.read_csv(neighborhood_adjacency_data_path, index_col=0)

    # print("Transaction Data Shape:", transaction_df.head(5))
    # print("Neighborhood Data Shape:", neighborhood_df.head(5)) 
    # print("Neighborhood Adjacency Data Shape:", neighborhood_adjacency_df.head(5))  
    return transaction_df, neighborhood_df, neighborhood_adjacency_df

def process_transaction_data(df):
    df['DATUM'] = pd.to_datetime(df['DATUM'])
    df.sort_values('DATUM', inplace=True)

    df['YEAR'] = df['DATUM'].dt.year
    df['MONTH'] = df['DATUM'].dt.month
    df.drop(["TRANSID"], axis=1, inplace=True)

    return df

def neighborhood_adj_to_edge_index(neigh_adj_matrix):
    """
    Convert neigborhood adjacency numpy matrix to edge index format.
    """
    # neigh_adj_df = neigh_adj_df.set_index('BUURTCODE')
    # neigh_adj_df = neigh_adj_df.drop(columns=['BUURTCODE'])
    # neigh_adj_df = neigh_adj_df.fillna(0)
    # neigh_adj_matrix = neigh_adj_df.values

    # Convert to edge index format
    G = nx.from_numpy_array(neigh_adj_matrix, create_using=nx.Graph)
    edge_index = torch.tensor(list(G.edges), dtype=torch.long).t().contiguous()
    
    return edge_index


def run_exp_final(data_path):
    """
    Train the model using the loaded data.
    """
    # Load data
    tx_df, neigh_df, neigh_adj_df = load_data(data_path)

    tx_df = process_transaction_data(tx_df)

    
    neigh_edge_index = neighborhood_adj_to_edge_index(neigh_adj_df.values)
    # Step 1: Graph Construction
    trans_in_dim = len(tx_df.columns) - 3
    neigh_in_dim = len(neigh_df.columns) - 2
    model = HierarchicalSpatioTemporalGNN(trans_in_dim=trans_in_dim,
                                         neigh_in_dim=neigh_in_dim,hidden_dim_trans=64, hidden_dim_neigh=128) 
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)
    criterion = nn.MSELoss()

    min_date = tx_df["DATUM"].min()
    max_date = tx_df["DATUM"].max()
       
    trainer = RollingTrainer(
        model=model,
        dataset_class=RollingGraphWindowDataset,
        transactions_df=tx_df,
        neighborhood_df=neigh_df,
        trans_feat_cols= tx_df.drop(columns=['BUURTCODE', "LOG_KOOPSOM", "DATUM"]).columns.tolist(),
        neigh_feature_cols= neigh_df.drop(columns=['BUURTCODE', "YEAR"]).columns.tolist(),
        target_col='LOG_KOOPSOM',
        optimizer=optimizer,
        loss_fn=criterion,
        window_years=3,
        neighborhood_adj=neigh_edge_index,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        n_epochs=25  # Train for 5 epochs per month
    )
     

    trainer.train(
        start_date=min_date,
        end_date=max_date,
        sliding_window=True,
        test_months_ahead=1
    )





