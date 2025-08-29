import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.nn import GCNConv, SAGEConv
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from dateutil.relativedelta import relativedelta
import os
import mlflow
from torch_geometric.data import HeteroData
# from src.utils.azure_utils import parse_hyperparameter_args
from dateutil.relativedelta import relativedelta
from collections import defaultdict
from torch_geometric.utils import subgraph

### 1. Data Loading Functions
def load_graph_data(neighborhood_file, edge_file):
    neighborhood_df = pd.read_csv(neighborhood_file)
    edges = pd.read_csv(edge_file, index_col=0)
    edge_index = adjacency_matrix_to_edge_index(edges)
    # print(neighborhood_df.columns.values)
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
# ### 2. Data Preparation for Sliding Windows

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



# --- Micro-Level Encoder (Transaction Graph) ---
class TransactionEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)


# --- Macro-Level Encoder (Neighborhood Graph) ---
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


# --- Cross-Level Interaction Module ---
class CrossLevelInteraction(nn.Module):
    def __init__(self, trans_dim, macro_dim, hidden_dim):
        super().__init__()
        self.trans_proj = nn.Linear(trans_dim, hidden_dim)
        self.macro_proj = nn.Linear(macro_dim, hidden_dim)

        # Gating for top-down fusion
        self.W_gamma = nn.Linear(hidden_dim, hidden_dim)
        self.U_gamma = nn.Linear(hidden_dim, hidden_dim)

        self.W_fuse = nn.Linear(hidden_dim, hidden_dim)
        self.U_fuse = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, trans_embed, macro_embed, trans_to_neigh):
        # Project embeddings to shared space
        h_micro_proj = self.trans_proj(trans_embed)
        h_macro_proj = self.macro_proj(macro_embed)

        num_neigh = macro_embed.size(0)

        # Bottom-up aggregation
        macro_agg = torch.zeros((num_neigh, h_micro_proj.size(1)), device=trans_embed.device)
        macro_agg = macro_agg.index_add(0, trans_to_neigh, h_micro_proj)
        counts = torch.bincount(trans_to_neigh, minlength=num_neigh).unsqueeze(1).clamp(min=1)
        bottom_up_macro = macro_agg / counts

        # Fuse macro GNN and bottom-up signals
        fuse_gate = torch.sigmoid(self.W_fuse(bottom_up_macro) + self.U_fuse(h_macro_proj))
        fused_macro_embed = fuse_gate * h_macro_proj + (1 - fuse_gate) * bottom_up_macro

        # Top-down injection
        aligned_macro = fused_macro_embed[trans_to_neigh]
        gamma = torch.sigmoid(self.W_gamma(h_micro_proj) + self.U_gamma(aligned_macro))
        trans_embed_final = gamma * h_micro_proj + (1 - gamma) * aligned_macro

        return trans_embed_final, fused_macro_embed


# --- Full Hierarchical Heterogeneous GNN Model ---
class HierarchicalHeteroGNN(nn.Module):
    def __init__(self, trans_in, macro_in, hidden_dim, out_dim):
        super().__init__()
        self.micro_encoder = TransactionEncoder(trans_in, hidden_dim)
        self.macro_encoder = MacroEncoder(macro_in, hidden_dim)
        self.cross_layer = CrossLevelInteraction(hidden_dim, hidden_dim, hidden_dim)
        self.predictor = nn.Linear(hidden_dim, out_dim)
        self.prev_macro_state = None

    def forward(self, data):
        trans_x = data['transaction'].x
        macro_x = data['neighborhood'].x
        trans_edge_index = data['transaction', 'to', 'transaction'].edge_index
        macro_edge_index = data['neighborhood', 'to', 'neighborhood'].edge_index

        trans_to_neigh = data['transaction'].neighborhood_index

        h_micro = self.micro_encoder(trans_x, trans_edge_index)
        h_macro_gated, h_macro_raw = self.macro_encoder(macro_x, macro_edge_index, self.prev_macro_state)

        h_trans_final, h_macro_final = self.cross_layer(h_micro, h_macro_gated, trans_to_neigh)

        self.prev_macro_state = h_macro_final.detach()

        out = self.predictor(h_trans_final).squeeze(-1)
        return out


def get_transaction_edges(transaction_df):
    """
    Generate edge_index for transactions: all transactions in the same neighborhood are fully connected.
    Returns edge_index (2, num_edges) as torch.LongTensor.
    """
    idx = transaction_df.index.to_numpy()
    buurtcodes = transaction_df['BUURTCODE'].to_numpy()
    edges = []
    # Group by neighborhood
    buurt_to_indices = defaultdict(list)
    for i, buurt in zip(idx, buurtcodes):
        buurt_to_indices[buurt].append(i)
    # For each neighborhood, connect all pairs (fully connected)
    for indices in buurt_to_indices.values():
        if len(indices) > 1:
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    edges.append([indices[i], indices[j]])
                    edges.append([indices[j], indices[i]])
    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    return edge_index
        
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
    data['transaction', 'to', 'transaction'].edge_index = get_transaction_edges(transaction_df)
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


def train_sliding_window(model, optimizer, criterion, transactions, node_features, edge_index,
                         window_months=61, epochs=10, batch_size=128):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    transactions['DATUM'] = pd.to_datetime(dict(year=transactions["YEAR"], month=transactions["MONTH"], day=1))
    transactions.sort_values("DATUM", inplace=True)

    min_date = transactions["DATUM"].min()
    max_date = transactions["DATUM"].max()

    start = min_date.to_period("M").to_timestamp()
    end = max_date.to_period("M").to_timestamp()

    # To track all test predictions per window
    all_window_preds = []

    # To track relative errors per epoch
    epoch_stats = []

    train_mape_hist, test_mape_hist = [], []
    train_mse_hist, test_mse_hist = [], []

    while start + relativedelta(months=window_months + 1) <= end:
        train_start = start
        train_end = train_start + relativedelta(months=window_months)
        test_month = train_end

        print(f"\n🪟 Window: {train_start.date()} → {train_end.date()} (Test: {test_month.strftime('%Y-%m')})")

        train_df = transactions[(transactions["DATUM"] >= train_start) & (transactions["DATUM"] < train_end)].copy()
        test_df = transactions[(transactions["DATUM"].dt.to_period("M") == test_month.to_period("M"))].copy()

        train_df.drop(["DATUM"], inplace=True, axis=1)
        test_df.drop(["DATUM"], inplace=True, axis=1)
        if train_df.empty or test_df.empty:
            print("⚠️ Skipping empty window.")
            start += relativedelta(months=1)
            continue

        combined_df = pd.concat([train_df, test_df], ignore_index=True)
        is_test_mask = torch.zeros(len(combined_df), dtype=torch.bool, device=device)
        is_test_mask[-len(test_df):] = True

        node_features_year = node_features.get(train_end.year, node_features[max(node_features.keys())])
        neighborhood_df = pd.DataFrame(node_features_year.numpy())
        neighborhood_df['BUURTCODE'] = range(node_features_year.shape[0])

        hetero_graph = build_hetero_graph(
            transaction_df=combined_df.reset_index(drop=True),
            neighborhood_df=neighborhood_df,
            test_mask=is_test_mask,
            neighbor_edge_index=edge_index
        ).to(device)


        train_indices = (~is_test_mask).nonzero(as_tuple=True)[0].to(device)
        test_indices = (is_test_mask).nonzero(as_tuple=True)[0].to(device)

        train_dataset = torch.utils.data.TensorDataset(train_indices)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

        print("start training")

        for epoch in range(epochs):
            model.train()
            batch_losses = []

            for batch in train_loader:
                batch_idx = batch[0]
                optimizer.zero_grad()
                first_year = combined_df.iloc[batch_idx[0].item()]["YEAR"]
                batch_neigh_feats = node_features.get(first_year, node_features[max(node_features.keys())]).to(device)

                # Subgraph creation
                edge_index_full = hetero_graph['transaction', 'to', 'transaction'].edge_index.to(device)
                sub_edge_index, mapping = subgraph(
                    batch_idx, edge_index_full, relabel_nodes=True,
                    num_nodes=hetero_graph['transaction'].x.size(0)
                )

                batch_hetero_graph = HeteroData()
                batch_hetero_graph['transaction'].x = hetero_graph['transaction'].x[batch_idx].to(device)
                batch_hetero_graph['transaction'].y = hetero_graph['transaction'].y[batch_idx].to(device)
                batch_hetero_graph['transaction'].neighborhood_index = hetero_graph['transaction'].neighborhood_index[batch_idx].to(device)
                batch_hetero_graph['transaction', 'to', 'transaction'].edge_index = sub_edge_index.to(device)

                # Add neighborhood data
                batch_hetero_graph['neighborhood'].x = batch_neigh_feats.to(device)
                batch_hetero_graph['neighborhood', 'to', 'neighborhood'].edge_index = hetero_graph['neighborhood', 'to', 'neighborhood'].edge_index.to(device)

                trans_to_neigh = batch_hetero_graph['transaction'].neighborhood_index  # shape: [batch_size]
                dst = torch.arange(len(batch_idx), dtype=torch.long, device=device)
                src = trans_to_neigh

                # Cross-type edges
                batch_hetero_graph['transaction', 'belongs_to', 'neighborhood'].edge_index = torch.stack([dst, src], dim=0).to(device)
                batch_hetero_graph['neighborhood', 'contains', 'transaction'].edge_index = torch.stack([src, dst], dim=0).to(device)

                # Forward and backward pass
                out = model(batch_hetero_graph).squeeze(-1)
                y_true = batch_hetero_graph["transaction"].y.squeeze(-1)
                loss = criterion(out, y_true)
                loss.backward()
                optimizer.step()
                batch_losses.append(loss.item())

            print("evaluate")
            model.eval()
            with torch.no_grad():
                out_eval = model(hetero_graph).squeeze(-1)
                y_true_eval = hetero_graph["transaction"].y.squeeze(-1)

                train_mse = criterion(out_eval[train_indices], y_true_eval[train_indices]).item()
                train_mape = mean_absolute_percentage_error(
                    y_true_eval[train_indices].cpu(), out_eval[train_indices].cpu()).item()

                test_mse = criterion(out_eval[test_indices], y_true_eval[test_indices]).item()
                test_mape = mean_absolute_percentage_error(
                    y_true_eval[test_indices].cpu(), out_eval[test_indices].cpu()).item()

            epoch_stats.append({
                "window_start": train_start.strftime('%Y-%m'),
                "epoch": epoch + 1,
                "train_mape": train_mape,
                "test_mape": test_mape,
                "train_mse": train_mse,
                "test_mse": test_mse
            })
            train_mape_hist.append(train_mape)
            test_mape_hist.append(test_mape)
            train_mse_hist.append(train_mse)
            test_mse_hist.append(test_mse)

            print(f"Epoch {epoch+1}/{epochs} | Train MSE: {train_mse:.4f} | MAPE: {train_mape:.2f}% | "
                  f"Test MSE: {test_mse:.4f} | MAPE: {test_mape:.2f}%")

            if epoch == epochs - 1:
                # Store metadata
                preds_df = pd.DataFrame({
                    "window_start": train_start.strftime('%Y-%m'),
                    "BUURTCODE": test_df["BUURTCODE"].values,
                    "YEAR": test_df["YEAR"].values,
                    "MONTH": test_df["MONTH"].values,
                    "TRANSID": test_df["TRANSID"].values,
                    "y_true": y_true_eval[test_indices].cpu().numpy(),
                    "y_pred": out_eval[test_indices].cpu().numpy(),
                })

                all_window_preds.append(preds_df)
            del batch_hetero_graph, out, y_true, out_eval, y_true_eval
            torch.cuda.empty_cache()

        start += relativedelta(months=1)

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


        # fig = plot_learning_curves(epochs, train_mse_hist, test_mse_hist, train_mape_hist, test_mape_hist)
        # mlflow.log_figure(fig, "learning_curves.png")


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
    
    neighborhood_features_path = os.path.join(data_path,"all_neighborhood_features_rotterdam.csv")
    edge_path = os.path.join(data_path,"rotterdam_adj_2023.csv")
    transaction_path = os.path.join(data_path, "rotterdam_transaction_data.csv")
    node_features, edge_index = load_graph_data(neighborhood_features_path, edge_path)
    transactions = load_transaction_data(transaction_path)
    trans_feature_dim = transactions.shape[1] - 3 # Exclude 'BUURTCODE', 'YEAR', 'MONTH', 'LOG_KOOPSOM'
    node_feature_dim = next(iter(node_features.values())).shape[1] 
    model = HierarchicalHeteroGNN(trans_in=trans_feature_dim, macro_in=node_feature_dim, hidden_dim=128, out_dim=1)
    # model = HierarchicalSpatioTemporalGNN(hidden_dim_trans=64, hidden_dim_neigh=64, trans_in_dim=trans_feature_dim, neigh_in_dim=node_feature_dim)
    
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)
    criterion = nn.MSELoss()


    
    
    print("start training")
    train_sliding_window(model, optimizer, criterion, transactions, node_features, edge_index,
                        window_months=61, epochs=100, batch_size=64)
    
  