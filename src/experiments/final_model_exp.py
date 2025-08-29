import os
import torch
import torch.nn as nn
from src.utils.rolling_trainer import train_sliding_window
from src.graphs.build_graph import load_graph_data, load_transaction_data
from src.models.hstgnn import HierarchicalHeteroGNN


def run_final_exp(data_path, hyperparameters):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    neighborhood_features_path = os.path.join(data_path,"all_neighborhood_features_rotterdam.csv")
    edge_path = os.path.join(data_path,"rotterdam_adj_2023.csv")
    transaction_path = os.path.join(data_path, "rotterdam_transaction_data.csv")

    node_features, edge_index = load_graph_data(neighborhood_features_path, edge_path)
    transactions = load_transaction_data(transaction_path)
    trans_feature_dim = transactions.shape[1] - 3 # Exclude 'BUURTCODE', 'YEAR', 'MONTH', 'LOG_KOOPSOM'
    node_feature_dim = next(iter(node_features.values())).shape[1] 

    model = HierarchicalHeteroGNN(trans_in=trans_feature_dim, macro_in=node_feature_dim, hidden_dim=64, out_dim=1)
    
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)
    criterion = nn.MSELoss()


    
    
    print("start training")
    train_sliding_window(model=model, optimizer=optimizer, criterion=criterion, transactions=transactions, node_features=node_features, edge_index=edge_index,
                        window_months=25, epochs=100, batch_size=64, base_patience=10)
    
