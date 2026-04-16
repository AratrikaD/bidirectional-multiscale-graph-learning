from copy import deepcopy
from dateutil.relativedelta import relativedelta
from torch_geometric.nn import knn_graph
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data, HeteroData
from src.models.mugrep import MugRepModel
from torch_geometric.utils import subgraph
from torch.utils.data import DataLoader, TensorDataset
from src.utils.metrics import mean_absolute_percentage_error
import time

def evaluate_mugrep(model, event_graph, comm_graph, is_test, criterion):
    model.eval()
    with torch.no_grad():
        torch.cuda.synchronize()
        start = time.time()
        preds = model(event_graph, comm_graph).squeeze(-1)

        torch.cuda.synchronize()
        end = time.time()
        inference_time = end - start
        y_true = event_graph.y.squeeze(-1)

        mse_train = criterion(preds[~is_test], y_true[~is_test]).item()
        mape_train = mean_absolute_percentage_error(y_true[~is_test].cpu(), preds[~is_test].cpu())
        mse_test = criterion(preds[is_test], y_true[is_test]).item()
        mape_test = mean_absolute_percentage_error(y_true[is_test].cpu(), preds[is_test].cpu())

    return mse_train, mape_train, mse_test, mape_test, preds.cpu().numpy(), y_true.cpu().numpy(), inference_time


def build_event_graph(combined_df, train_df, idmap, nb_latest, time_window_days=90, k_neighbors=5):
    df = combined_df.copy()
    df['DATUM'] = pd.to_datetime(df['DATUM'])

    # === Target: log selling price ===
    y = torch.tensor(df['LOG_KOOPSOM'].to_numpy(), dtype=torch.float)

    # === Features ===
    exclude = {'TRANSID','DATE','LOG_KOOPSOM','BUURTCODE'}
    feat_cols = [c for c in df.columns if c not in exclude and np.issubdtype(df[c].dtype, np.number)]

    scaler_tx = StandardScaler().fit(train_df[feat_cols].fillna(0.0))
    df[feat_cols] = scaler_tx.transform(df[feat_cols].fillna(0.0))

    x = torch.tensor(df[feat_cols].to_numpy(), dtype=torch.float)

    # === Neighborhood & district indices ===
    neigh_idx = torch.tensor(df['BUURTCODE'].map(idmap).to_numpy(), dtype=torch.long)
    dist_map = nb_latest.set_index('neighborhood_id')['district_id']
    district_idx = torch.tensor(df['BUURTCODE'].map(dist_map).to_numpy(), dtype=torch.long)

    # === Event edges (temporal within neighborhood) ===
    df['ts'] = (df['DATUM'] - df['DATUM'].min()).dt.days
    src_list, dst_list = [], []
    for _, g in df.groupby(neigh_idx.numpy()):
        g = g.sort_values('ts')
        arr_idx = g.index.to_numpy(); ts = g['ts'].to_numpy()
        for i in range(len(arr_idx)):
            j, cnt = i-1, 0
            while j>=0 and ts[i]-ts[j]<=time_window_days and cnt<k_neighbors:
                src_list.append(arr_idx[i]); dst_list.append(arr_idx[j]); cnt+=1; j-=1
            j, cnt = i+1, 0
            while j<len(arr_idx) and ts[j]-ts[i]<=time_window_days and cnt<k_neighbors:
                src_list.append(arr_idx[i]); dst_list.append(arr_idx[j]); cnt+=1; j+=1

    if len(src_list)==0:
        edge_index = knn_graph(x, k=min(k_neighbors, max(1, x.size(0)-1)), loop=False)
    else:
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)

    event_graph = Data(x=x, edge_index=edge_index, y=y)
    event_graph.neighborhood = neigh_idx
    event_graph.district = district_idx
    event_graph.trans_id = torch.tensor(df['TRANSID'].to_numpy(), dtype=torch.long)

    return event_graph, df



def train_sliding_window_mugrep(model, optimizer, criterion,
                                transactions, comm_graph,
                                idmap_rev, dmap_rev, nb_latest,
                                window_months=61, epochs=30, batch_size=128):

    device = next(model.parameters()).device

    # Ensure datetime
    transactions["DATUM"] = pd.to_datetime(transactions["DATUM"])
    transactions.sort_values("DATUM", inplace=True)

    min_date, max_date = transactions["DATUM"].min(), transactions["DATUM"].max()
    start = min_date.to_period("M").to_timestamp()
    end = max_date.to_period("M").to_timestamp()

    all_preds, all_stats = [], []
    runtime_stats = []
    window_runtime_stats = []

    while start + relativedelta(months=window_months + 1) <= end:
        train_start = start
        train_end = train_start + relativedelta(months=window_months)
        test_month = train_end
        
        print(f"\n🪟 Window: {train_start.date()} → {train_end.date()} (Test: {test_month.strftime('%Y-%m')})")

        window_start_time = time.time()
        total_epoch_time = 0.0
        epochs_ran = 0
        total_batches_window = 0
        # === Split train/test ===
        train_df = transactions[(transactions["DATUM"] >= train_start) & (transactions["DATUM"] < train_end)].copy()
        test_df = transactions[(transactions["DATUM"].dt.to_period("M") == test_month.to_period("M"))].copy()
        if train_df.empty or test_df.empty:
            start += relativedelta(months=1)
            continue

        combined_df = pd.concat([train_df, test_df], ignore_index=True)
        is_test = torch.zeros(len(combined_df), dtype=torch.bool, device=device)
        is_test[-len(test_df):] = True

        # === Build event graph ===
        event_graph, processed_df = build_event_graph(
            combined_df, train_df, idmap_rev, nb_latest,
            time_window_days=90, k_neighbors=5
        )
        event_graph = event_graph.to(device)

        train_idx = (~is_test).nonzero(as_tuple=True)[0]
        test_idx = (is_test).nonzero(as_tuple=True)[0]

        train_dataset = TensorDataset(train_idx)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        best_mse, best_state = float("inf"), None

        # === Training loop ===
        for epoch in range(epochs):
            model.train()
            batch_losses = []

            epoch_start_time = time.time()
            epoch_subgraph_time = 0.0
            epoch_compute_time = 0.0
            epoch_total_time = 0.0
            batch_count = 0

            for (batch_idx,) in train_loader:
                batch_idx = batch_idx.to(device)

                t0 = time.time()
                # Build subgraph for this batch
                sub_edge_index, edge_attr, mapping = subgraph(
                    batch_idx,
                    event_graph.edge_index,
                    relabel_nodes=True,
                    num_nodes=event_graph.num_nodes,
                    return_edge_mask=True
                )
                sub_nodes = batch_idx
                
                 # === Build a proper HeteroData for the subgraph ===
                subgraph_data = Data(x=event_graph.x[sub_nodes], edge_index=sub_edge_index, y=event_graph.y[sub_nodes])
                subgraph_data.neighborhood = event_graph.neighborhood[sub_nodes]
                subgraph_data.district = event_graph.district[sub_nodes]
                subgraph_data.trans_id = event_graph.trans_id[sub_nodes]
                # subgraph_data['event'].x = event_graph.x[sub_nodes]
                # subgraph_data['event'].y = event_graph.y[sub_nodes]
                # subgraph_data['event'].district = event_graph.district[sub_nodes]
                # subgraph_data['event'].neighborhood = event_graph.neighborhood[sub_nodes]
                # subgraph_data['event'].edge_index = sub_edge_index

                    
                
                subgraph_data = subgraph_data.to(device)
                t1 = time.time()

                optimizer.zero_grad()
                preds = model(subgraph_data, comm_graph).squeeze(-1)  
                loss = criterion(preds, subgraph_data.y.squeeze(-1))
                loss.backward()
                optimizer.step()

                t2 = time.time()
                epoch_subgraph_time += (t1 - t0)
                epoch_compute_time += (t2 - t1)
                epoch_total_time += (t2 - t0)
                batch_count += 1
                total_batches_window += 1

                batch_losses.append(loss.item())
            
            epoch_end_time = time.time()
            epoch_time = epoch_end_time - epoch_start_time

            avg_batch_time = epoch_time / max(batch_count, 1)

            total_epoch_time += epoch_time
            epochs_ran += 1

            
            mse_train, mape_train, mse_test, mape_test, preds_eval, y_eval, inference_time = evaluate_mugrep(
                model, event_graph, comm_graph, is_test, criterion
            )

            
            runtime_stats.append({
                "window_start": train_start.strftime('%Y-%m'),
                "epoch": epoch + 1,
                "epoch_time_sec": epoch_time,
                "avg_batch_time_sec": avg_batch_time,
                "subgraph_time_sec": epoch_subgraph_time,
                "compute_time_sec": epoch_compute_time,
                "subgraph_pct": epoch_subgraph_time / epoch_time if epoch_time > 0 else 0,
                "compute_pct": epoch_compute_time / epoch_time if epoch_time > 0 else 0,
                "inference_time_full_sec": inference_time,
                "inference_time_per_node_ms": (inference_time / len(y_eval)) * 1000,
                "num_batches": batch_count
            })

            # print(f"🪟 {train_start.strftime('%Y-%m')} | "
            #       f"Epoch {epoch+1}/{epochs} | "
            #       f"Train MSE {mse_train:.4f} | Test MSE {mse_test:.4f}")
            print(f"Epoch {epoch+1}/{epochs} | Train MSE: {mse_train:.4f} | MAPE: {mape_train:.2f}% | "
                  f"Test MSE: {mse_test:.4f} | MAPE: {mape_test:.2f}%")

            if mse_test < best_mse:
                best_mse = mse_test
                best_state = deepcopy(model.state_dict())

        # === Restore best model and save predictions ===
        window_end_time = time.time()
        window_time = window_end_time - window_start_time

        avg_epoch_time = total_epoch_time / max(epochs_ran, 1)
        avg_batch_time_window = total_epoch_time / max(total_batches_window, 1)
        if best_state is not None:
            model.load_state_dict(best_state)

        _, _, mse_test, mape_test, preds_eval, y_eval, inference_time = evaluate_mugrep(
            model, event_graph, comm_graph, is_test, criterion
        )

        preds_df = pd.DataFrame({
            "TRANSID": event_graph.trans_id[test_idx].cpu().numpy(),
            "window_start": train_start.strftime('%Y-%m'),
            "BUURTCODE": [idmap_rev[n] for n in event_graph.neighborhood[test_idx].cpu().numpy()],
            "district_id": [dmap_rev[d] for d in event_graph.district[test_idx].cpu().numpy()],
            "y_true": y_eval[test_idx.cpu().numpy()],
            "y_pred": preds_eval[test_idx.cpu().numpy()]
        })

        all_preds.append(preds_df)
        all_stats.append({
            "window_start": train_start.strftime('%Y-%m'),
            "test_mse": mse_test,
            "test_mape": mape_test
        })

        window_runtime_stats.append({
            "window_start": train_start.strftime('%Y-%m'),
            "epoch": "window_total",
            "window_time_sec": window_time,
            "avg_epoch_time_sec": avg_epoch_time,
            "avg_batch_time_sec": avg_batch_time_window,
            "epochs_ran": epochs_ran,
            "total_batches": total_batches_window,
            "inference_time_full_sec": inference_time,
            "inference_time_per_node_ms": (inference_time / len(y_eval)) * 1000,
            "num_train_samples": len(train_df)
        })

       
        start += relativedelta(months=1)

    preds_all = pd.concat(all_preds, ignore_index=True)
    stats_all = pd.DataFrame(all_stats)
    runtime_df = pd.DataFrame(runtime_stats)
    runtime_df.to_csv("./outputs/mugrep_runtime_stats.csv", index=False)

    preds_all.to_csv("./outputs/mugrep_preds.csv", index=False)
    stats_all.to_csv("./outputs/mugrep_stats.csv", index=False)

    window_runtime_df = pd.DataFrame(window_runtime_stats)
    window_runtime_df.to_csv("./outputs/mugrep_window_runtime_stats.csv", index=False)

    # Log aggregate metrics to MLflow
    import mlflow
    if window_runtime_stats:
        mlflow.log_metric("total_window_time_sec", sum(r["window_time_sec"] for r in window_runtime_stats))
        mlflow.log_metric("avg_batch_time_sec", float(window_runtime_df["avg_batch_time_sec"].mean()))
        mlflow.log_metric("avg_epoch_time_sec", float(window_runtime_df["avg_epoch_time_sec"].mean()))
        mlflow.log_metric("avg_inference_time_full_sec", float(window_runtime_df["inference_time_full_sec"].mean()))
        mlflow.log_metric("avg_inference_per_node_ms", float(window_runtime_df["inference_time_per_node_ms"].mean()))
        mlflow.log_metric("avg_epoch_time_per_sample_sec", float((window_runtime_df["avg_epoch_time_sec"] / window_runtime_df["num_train_samples"]).mean()))

    return preds_all, stats_all
        
