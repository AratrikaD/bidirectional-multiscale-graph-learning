from copy import deepcopy
import torch
from torch_geometric.loader import NeighborLoader
from dateutil.relativedelta import relativedelta
import pandas as pd
import torch.nn.functional as F
from src.utils.metrics import mean_absolute_percentage_error
from src.graphs.build_graph import build_hetero_graph
from torch_geometric.data import HeteroData
from torch_geometric.utils import subgraph
from torch.optim.lr_scheduler import LambdaLR



def evaluate(model, train_indices, test_indices, hetero_graph, criterion):
    model.eval()

    with torch.no_grad():
        # pred = model(hetero_graph).squeeze(-1)
        
        pred, embeddings = model.forward_with_embeddings(hetero_graph)

        pred = pred.squeeze(-1)
        embeddings = embeddings.cpu().numpy()
        y_true = hetero_graph['transaction'].y.squeeze(-1)

        mse_train = criterion(pred[train_indices], y_true[train_indices]).item()
        mape_train = mean_absolute_percentage_error(y_true[train_indices].cpu(), pred[train_indices].cpu()).item()

        mse_test = criterion(pred[test_indices], y_true[test_indices]).item()
        mape_test = mean_absolute_percentage_error(y_true[test_indices].cpu(), pred[test_indices].cpu()).item()

    return mse_train, mape_train, mse_test, mape_test, pred.cpu().numpy(), y_true.cpu().numpy(), embeddings



def get_batch_subgraph(batch_idx, hetero_graph, transactions_df, batch_neighborhood_features, mode='neighborhood_temporal'):
    """Builds a filtered subgraph from batch indices using neighborhood and temporal constraints."""
    # assert mode in ['batch', 'neighborhood_temporal', 'full'], "Unsupported mode"

    device = batch_idx.device
    full_edge_index = hetero_graph['transaction', 'to', 'transaction'].edge_index
    num_nodes = hetero_graph['transaction'].x.size(0)

    if mode == 'full':
        return hetero_graph, batch_idx

    if mode == 'batch':
        sub_edge_index, edge_mask, mapping = subgraph(
            batch_idx,
            full_edge_index,
            relabel_nodes=True,
            num_nodes=num_nodes,
            return_edge_mask=True
        )
        sub_nodes = batch_idx

    elif mode == 'neighborhood_temporal':
        # print("Building neighborhood-temporal subgraph")
        # Step 1: Find neighborhoods for the batch
        batch_neighs = hetero_graph['transaction'].neighborhood_index[batch_idx]

        # Step 2: Mask all transactions in those neighborhoods
        neighborhood_mask = torch.isin(
            hetero_graph['transaction'].neighborhood_index,
            batch_neighs.unique()
        )

        # Step 3: Temporal filter — only include transactions at or before latest batch month
        batch_times = transactions_df.iloc[batch_idx.tolist()][['YEAR', 'MONTH']]
        batch_cutoff = (batch_times['YEAR'] * 12 + batch_times['MONTH']).max()

        all_times = transactions_df[['YEAR', 'MONTH']]
        sub_time_vals = all_times['YEAR'] * 12 + all_times['MONTH']
        time_mask = torch.tensor((sub_time_vals <= batch_cutoff).values, device=device)

        # Combine masks
        combined_mask = neighborhood_mask & time_mask
        sub_nodes = combined_mask.nonzero(as_tuple=True)[0]

        # Step 4: Build subgraph from filtered nodes
        sub_edge_index, edge_attr = subgraph(
            subset=sub_nodes,
            edge_index=full_edge_index,
            relabel_nodes=True,
            num_nodes=num_nodes,
            return_edge_mask=False
        )

    # === Build HeteroData for the batch === #
    batch_subgraph = HeteroData()
    batch_subgraph['transaction'].x = hetero_graph['transaction'].x[sub_nodes]
    batch_subgraph['transaction'].y = hetero_graph['transaction'].y[sub_nodes]
    batch_subgraph['transaction'].neighborhood_index = hetero_graph['transaction'].neighborhood_index[sub_nodes]
    batch_subgraph['transaction', 'to', 'transaction'].edge_index = sub_edge_index
    
    # Neighborhood node and structure
    batch_subgraph['neighborhood'].x = batch_neighborhood_features
    batch_subgraph['neighborhood', 'to', 'neighborhood'].edge_index = hetero_graph['neighborhood', 'to', 'neighborhood'].edge_index

    # Transaction → Neighborhood bipartite connections
    dst = torch.arange(sub_nodes.size(0), device=device)
    src = batch_subgraph['transaction'].neighborhood_index
    batch_subgraph['transaction', 'belongs_to', 'neighborhood'].edge_index = torch.stack([dst, src], dim=0)
    batch_subgraph['neighborhood', 'contains', 'transaction'].edge_index = torch.stack([src, dst], dim=0)

    # Step 5: Map batch_idx to new positions in sub_nodes
    # This lets you index model output correctly
    batch_pos = (sub_nodes.unsqueeze(1) == batch_idx.unsqueeze(0)).nonzero(as_tuple=True)[0]

    return batch_subgraph.to(device), batch_pos



def train_sliding_window(model, optimizer, criterion, transactions, node_features, edge_index, mode='neighborhood_temporal',
                         window_months=61, epochs=10, batch_size=128, base_patience=3):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    transactions['DATUM'] = pd.to_datetime(dict(year=transactions["YEAR"], month=transactions["MONTH"], day=1))
    transactions.sort_values("DATUM", inplace=True)

    min_date = transactions["DATUM"].min()
    max_date = transactions["DATUM"].max()

    start = min_date.to_period("M").to_timestamp()
    end = max_date.to_period("M").to_timestamp()

    all_window_preds = []
    epoch_stats = []

    train_mape_hist, test_mape_hist = [], []
    train_mse_hist, test_mse_hist = [], []

    window_count = 0

    while start + relativedelta(months=window_months + 1) <= end:
        window_count += 1
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

        train_indices = (~is_test_mask).nonzero(as_tuple=True)[0].cpu()
        test_indices = (is_test_mask).nonzero(as_tuple=True)[0].to(device)

        train_dataset = torch.utils.data.TensorDataset(train_indices)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

        # scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: min(1.0, (epoch + 1) / 5))  # Warmup for first 5 epochs

        best_test_mse = float('inf')
        best_epoch = 0
        best_model_state = None
        patience_counter = 0
        patience = float('inf') if window_count <= 3 else base_patience

        print("start training")

        for epoch in range(epochs):
            model.train()
            batch_losses = []

            for batch in train_loader:
                batch_idx = batch[0].to(device)  # indices of nodes in this batch
                optimizer.zero_grad()

                first_year = combined_df.iloc[batch_idx[0].item()]["YEAR"]
                batch_neigh_feats = node_features.get(first_year, node_features[max(node_features.keys())]).to(device)

                # Get the batch subgraph
                batch_subgraph, batch_pos = get_batch_subgraph(
                    batch_idx = batch_idx,
                    hetero_graph= hetero_graph,
                    transactions_df=combined_df,
                    batch_neighborhood_features=batch_neigh_feats,
                    mode=mode
                )

                # Now forward pass on batch_subgraph
                out = model(batch_subgraph).squeeze(-1)

                # Compute loss only on batch nodes
                loss = criterion(out[batch_pos], batch_subgraph['transaction'].y[batch_pos].squeeze(-1))

                loss.backward()
                optimizer.step()

                batch_losses.append(loss.item())


            # scheduler.step()

            print("evaluate")
            train_mse, train_mape, test_mse, test_mape, out_eval, y_true_eval, embeddings = evaluate(
                model, train_indices, test_indices, hetero_graph, criterion
            )

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

            if test_mse < best_test_mse:
                best_test_mse = test_mse
                best_epoch = epoch
                best_model_state = deepcopy(model.state_dict())
                patience_counter = 0

                preds_df = pd.DataFrame({
                    "window_start": train_start.strftime('%Y-%m'),
                    "BUURTCODE": test_df["BUURTCODE"].values,
                    "YEAR": test_df["YEAR"].values,
                    "MONTH": test_df["MONTH"].values,
                    "TRANSID": test_df["TRANSID"].values,
                    "y_true": y_true_eval[test_indices.cpu()],
                    "y_pred": out_eval[test_indices.cpu()],
                    "embedding": embeddings[test_indices.cpu()].tolist(),
                })
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"⏹️ Early stopping at epoch {epoch+1} (best was {best_epoch+1})")
                del batch_subgraph, out, out_eval, y_true_eval
                torch.cuda.empty_cache()
                break

            del batch_subgraph, out, out_eval, y_true_eval
            torch.cuda.empty_cache()
        
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            
        all_window_preds.append(preds_df)
        start += relativedelta(months=1)

    final_preds_df = pd.concat(all_window_preds, ignore_index=True)
    final_preds_df.to_csv("./outputs/all_test_predictions.csv", index=False)

    stats_df = pd.DataFrame(epoch_stats)
    stats_df.to_csv("./outputs/training_stats.csv", index=False)
