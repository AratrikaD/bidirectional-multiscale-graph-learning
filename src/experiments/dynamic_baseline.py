import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.nn import GCNConv, Linear, SAGEConv, GATConv
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, RobustScaler
from dateutil.relativedelta import relativedelta
import sys
import os
import mlflow
# from src.utils.azure_utils import parse_hyperparameter_args

### 1. Data Loading Functions
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

def scale_window_data(train_data, test_data, scaler):
    train_scaled = scaler.fit_transform(train_data)
    test_scaled = scaler.transform(test_data)

    return torch.tensor(train_scaled, dtype=torch.float32), torch.tensor(test_scaled, dtype=torch.float32)
### 2. Data Preparation for Sliding Windows

def prepare_window_data(transactions, node_features, start_year, window_size=60):
    data = []
    for month in range(window_size):
        year = start_year + (month // 12)
        month_of_year = (month % 12) + 1
        print(year)
        print(month)

        monthly_trans = transactions[(transactions['YEAR'] == year) & (transactions['MONTH'] == month_of_year)]
        if monthly_trans.empty:
            print("empty")
            continue

        current_node_features = node_features.get(year, node_features[max(node_features.keys())])
        # print(monthly_trans)
        node_idx = torch.tensor(monthly_trans['BUURTCODE'].values, dtype=torch.int32)

        trans_feats = torch.tensor(monthly_trans.drop(["BUURTCODE"], axis=1).iloc[:, :-3].values, dtype=torch.float32)
        # print(trans_feats.shape)
        time_feats = torch.tensor([(year, month_of_year)] * len(monthly_trans), dtype=torch.float32)
        prices = torch.tensor(monthly_trans['LOG_KOOPSOM'].values, dtype=torch.float32).unsqueeze(1)

        data.append((trans_feats, node_idx, time_feats, prices, current_node_features))

    print(len(data))
    return data

def split_train_test(data, test_ratio=0.2):
    split_idx = int(len(data) * (1 - test_ratio))
    print(split_idx)
    return data[:split_idx], data[split_idx:]

### 3. Utility for Mini-Batch Creation
def create_batches(trans_feats, node_idx, time_feats, prices, batch_size):
    dataset = TensorDataset(trans_feats, node_idx, time_feats, prices)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def adjacency_matrix_to_edge_index(adj_matrix):

    adj_matrix = adj_matrix.to_numpy()

    edge_list = torch.nonzero(torch.tensor(adj_matrix), as_tuple=False)

    edge_index = torch.cat([edge_list, edge_list.flip(0)], dim=0)

    edge_index = edge_index.unique(dim=0)

    return edge_index.T

### 4. Model Definitions
class NeighborhoodGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(NeighborhoodGNN, self).__init__()
        self.conv1 = GATConv(in_dim, hidden_dim)
        self.conv2 = GATConv(hidden_dim, out_dim)

    def forward(self, x, edge_index):
        # print(x.shape)
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x,edge_index))
        return x

class TransactionPredictor(nn.Module):
    def __init__(self, trans_dim, emb_dim, time_dim, hidden_dim):
        super(TransactionPredictor, self).__init__()
        self.fc1 = nn.Linear(trans_dim + emb_dim + time_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, trans_feats, node_emb, time_feats):
        x = torch.cat((trans_feats, node_emb, time_feats), dim=1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class IntegratedModel(nn.Module):
    def __init__(self, gnn, predictor):
        super(IntegratedModel, self).__init__()
        self.gnn = gnn
        self.predictor = predictor

    def forward(self, trans_feats, node_features, edge_index, node_idx, time_feats):
        # print(node_features.shape)
        # print(edge_index.shape)
        emd = self.gnn(node_features, edge_index)
        # print("Embeding shape" ,emd.shape)
        node_emb = emd[node_idx]
        return self.predictor(trans_feats, node_emb, time_feats)

### 5. Metrics Calculation
def mean_absolute_percentage_error(y_true, y_pred):
    return torch.mean(torch.abs((np.exp(y_true)- np.exp(y_pred)) / (np.exp(y_true) + 1e-8))) * 100

### 6. Evaluation Function (for Train and Test)
def evaluate(model, data_loader, edge_index, node_features_year, criterion):
    model.eval()
    total_loss, total_mse, total_mape, num_samples = 0.0, 0.0, 0.0, 0

    with torch.no_grad():
        for batch in data_loader:
            trans_feats, node_idx, time_feats, prices = batch

            # affected_nodes = list(set(node_idx.tolist()))
            # sub_nodes, sub_edge_index, _, _ = k_hop_subgraph(affected_nodes, 2, edge_index)

            preds = model(trans_feats, node_features_year, edge_index, node_idx, time_feats)

            loss = criterion(preds, prices)
            mse = F.mse_loss(preds, prices)
            mape = mean_absolute_percentage_error(prices, preds)

            total_loss += loss.item()
            total_mse += mse.item() * prices.size(0)
            total_mape += mape.item() * prices.size(0)
            num_samples += prices.size(0)

    return total_loss / len(data_loader), total_mse / num_samples, total_mape / num_samples


from dateutil.relativedelta import relativedelta

def train_sliding_window(model, optimizer, criterion, transactions, node_features, edge_index,
                         window_months=61, epochs=10, batch_size=128):

    transactions['DATUM'] = pd.to_datetime(dict(year=transactions["YEAR"], month=transactions["MONTH"], day=1))
    transactions.sort_values("DATUM", inplace=True)

    min_date = transactions["DATUM"].min()
    max_date = transactions["DATUM"].max()
    print(f"⏳ Detected date range in data: {min_date.date()} → {max_date.date()}")

    start = min_date.to_period("M").to_timestamp()
    end = max_date.to_period("M").to_timestamp()

    # To track all test predictions per window
    all_window_preds = []

    # To track relative errors per epoch
    epoch_stats = []

    while start + relativedelta(months=window_months + 1) <= end:
        train_start = start
        train_end = train_start + relativedelta(months=window_months)
        test_month = train_end

        print(f"\n🪟 Window: {train_start.date()} → {train_end.date()} (Test: {test_month.strftime('%Y-%m')})")

        # Filter window
        train_df = transactions[(transactions["DATUM"] >= train_start) & (transactions["DATUM"] < train_end)].copy()
        test_df = transactions[(transactions["DATUM"].dt.to_period("M") == test_month.to_period("M"))].copy()

        if train_df.empty or test_df.empty:
            print("⚠️ Skipping empty window.")
            start += relativedelta(months=1)
            continue

        # Drop unused
        train_df.drop(["DATUM"], inplace=True, axis=1)
        test_df.drop(["DATUM"], inplace=True, axis=1)

        # Feature scaling
        scaler = StandardScaler()
        train_feats = torch.tensor(scaler.fit_transform(
            train_df.drop(columns=["BUURTCODE", "TRANSID", "LOG_KOOPSOM", "YEAR", "MONTH"]).values), dtype=torch.float32)
        test_feats = torch.tensor(scaler.transform(
            test_df.drop(columns=["BUURTCODE", "TRANSID", "LOG_KOOPSOM", "YEAR", "MONTH"]).values), dtype=torch.float32)

        train_node_idx = torch.tensor(train_df["BUURTCODE"].values, dtype=torch.int64)
        test_node_idx = torch.tensor(test_df["BUURTCODE"].values, dtype=torch.int64)
        train_time = torch.tensor(train_df[["YEAR", "MONTH"]].values, dtype=torch.float32)
        test_time = torch.tensor(test_df[["YEAR", "MONTH"]].values, dtype=torch.float32)
        train_y = torch.tensor(train_df["LOG_KOOPSOM"].values, dtype=torch.float32).unsqueeze(1)
        test_y = torch.tensor(test_df["LOG_KOOPSOM"].values, dtype=torch.float32).unsqueeze(1)

        node_features_year = node_features.get(train_end.year, node_features[max(node_features.keys())])

        for epoch in range(epochs):
            model.train()
            total_loss = 0.0
            train_loader = create_batches(train_feats, train_node_idx, train_time, train_y, batch_size)

            for batch in train_loader:
                optimizer.zero_grad()
                trans_feats_b, node_idx_b, time_feats_b, y_b = batch
                preds = model(trans_feats_b, node_features_year, edge_index, node_idx_b, time_feats_b)
                loss = criterion(preds, y_b)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            # Evaluate
            train_loss, train_mse, train_mape = evaluate(model, train_loader, edge_index, node_features_year, criterion)
            test_loader = create_batches(test_feats, test_node_idx, test_time, test_y, batch_size)
            test_loss, test_mse, test_mape = evaluate(model, test_loader, edge_index, node_features_year, criterion)

            # Save epoch-level stats
            epoch_stats.append({
                "window_start": train_start.strftime('%Y-%m'),
                "epoch": epoch + 1,
                "train_mape": train_mape,
                "test_mape": test_mape,
                "train_mse": train_mse,
                "test_mse": test_mse
            })

            print(f"Epoch {epoch+1}/{epochs} | Train MSE: {train_mse:.4f} | MAPE: {train_mape:.2f}% | "
                  f"Test MSE: {test_mse:.4f} | MAPE: {test_mape:.2f}%")

            # Save predictions for the last epoch
            if epoch == epochs - 1:
                model.eval()
                preds_list = []
                with torch.no_grad():
                    for i in range(0, len(test_feats), batch_size):
                        batch_feats = test_feats[i:i+batch_size]
                        batch_idx = test_node_idx[i:i+batch_size]
                        batch_time = test_time[i:i+batch_size]
                        preds = model(batch_feats, node_features_year, edge_index, batch_idx, batch_time)
                        preds_list.append(preds)

                all_preds = torch.cat(preds_list).squeeze().numpy()
                all_true = test_y.squeeze().numpy()

                # Store metadata
                preds_df = pd.DataFrame({
                    "window_start": train_start.strftime('%Y-%m'),
                    "BUURTCODE": test_df["BUURTCODE"].values,
                    "YEAR": test_df["YEAR"].values,
                    "MONTH": test_df["MONTH"].values,
                    "TRANSID": test_df["TRANSID"].values,
                    "y_true": all_true,
                    "y_pred": all_preds
                })

                all_window_preds.append(preds_df)

        start += relativedelta(months=1)

    # Save all test predictions
    final_preds_df = pd.concat(all_window_preds, ignore_index=True)
    final_preds_df.to_csv("./outputs/all_test_predictions.csv", index=False)

    # Save epoch stats
    stats_df = pd.DataFrame(epoch_stats)
    stats_df.to_csv("./outputs/training_stats.csv", index=False)

    # Optional: plot learning curves
    fig = plot_learning_curves(epochs, 
                               [e['train_mse'] for e in epoch_stats],
                               [e['test_mse'] for e in epoch_stats],
                               [e['train_mape'] for e in epoch_stats],
                               [e['test_mape'] for e in epoch_stats])
    mlflow.log_figure(fig, "learning_curves.png")



def plot_learning_curves(num_epochs, train_losses, test_losses, train_mapes, test_mapes):
    # Visualize Results
    
    
    fig, axs = plt.subplots(2, 2)
    fig.set_size_inches(10, 10)
    axs[0, 0].plot( train_losses, label='Training Loss')
    axs[0, 0].plot( test_losses, label='Test Loss')
    axs[0, 0].set_title('Training and Test Losses (MSE)')
    axs[0, 1].plot( train_mapes, label='Training MAPE')
    axs[0, 1].plot( test_mapes, label='Test MAPE')
    axs[0, 1].set_title('Training and Test MAPE')
    axs[1, 0].plot( train_losses, label='Training Loss')
    axs[1, 0].plot( test_losses, label='Test Loss')
    axs[1, 0].set_yscale("log")
    axs[1, 0].set_title('Training and Test Log Losses (MSE)')
    axs[1, 1].plot( train_mapes, label='Training MAPE')
    axs[1, 1].plot( test_mapes, label='Test MAPE')
    axs[1, 1].set_yscale("log")
    axs[1, 1].set_title('Training and Test Log MAPE')

    for ax in axs.flat:
        ax.set(xlabel='x-label', ylabel='y-label')

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()
    plt.show()
    return fig


def run_exp(data_path):
    gnn = NeighborhoodGNN(in_dim=202, hidden_dim=128, out_dim=32)
    predictor = TransactionPredictor(trans_dim=12, emb_dim=32, time_dim=2, hidden_dim=100)
    model = IntegratedModel(gnn, predictor)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001, weight_decay=0.0001)
    criterion = nn.MSELoss()

    neighborhood_features_path = os.path.join(data_path,"all_neighborhood_features_rotterdam.csv")

    # r"C:\Users\AratrikaD\rdlabs-gnns-for-property-valuation\gnns-for-property-valuation\housing-data\all_neighborhood_features_rotterdam.csv"

    edge_path = os.path.join(data_path,"rotterdam_adj_2023.csv")
    # r"C:\Users\AratrikaD\rdlabs-gnns-for-property-valuation\gnns-for-property-valuation\housing-data\rotterdam_adj_2023.csv" 
    node_features, edge_index = load_graph_data(neighborhood_features_path, edge_path)
    transaction_path = os.path.join(data_path, "rotterdam_transaction_data.csv")
    transactions = load_transaction_data(transaction_path)
    print("start training")
    train_sliding_window(model, optimizer, criterion, transactions, node_features, edge_index,
                        window_months=61, epochs=100, batch_size=64)
    
    