# online_learning_hierarchical_gnn.py
"""
End‑to‑end pipeline that converts the original rolling‑window implementation
into a true **online‑learning** workflow **and** captures per‑month prediction
snapshots (`preds_df`).
"""


# … (load_graph_data, load_transaction_data, StreamingScaler, model defs, etc.)
# Keep everything from prior revision up to but **excluding** train_online.

from __future__ import annotations

import os
import random
from collections import defaultdict
from datetime import datetime
from typing import Generator, Tuple, List
from dateutil.relativedelta import relativedelta
import joblib
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import HeteroData
from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.utils import subgraph
from torch.utils.data import DataLoader, TensorDataset

from src.models.hstgnn import HierarchicalHeteroGNN
from src.utils.rolling_trainer import get_batch_subgraph

from src.graphs.create_edges import get_transaction_edges
from src.graphs.build_graph import load_graph_data, load_transaction_data, build_hetero_graph

# --------------------------------------------------------------------------------------
# 2. Streaming StandardScaler
# --------------------------------------------------------------------------------------

class StreamingScaler:
    def __init__(self, with_mean: bool = False):
        self.scaler = StandardScaler(with_mean=with_mean)
        self.initialised = False

    def partial_fit(self, x: np.ndarray):
        self.scaler.partial_fit(x)
        self.initialised = True

    def transform(self, x: np.ndarray) -> np.ndarray:
        if not self.initialised:
            raise RuntimeError("StreamingScaler used before any partial_fit call")
        return self.scaler.transform(x)

    def save(self, path: str):
        joblib.dump(self, path)

# --------------------------------------------------------------------
# Replay Buffer Class
# --------------------------------------------------------------------
class ReplayBuffer:
    def __init__(self, max_size: int = 500):
        self.buffer = []
        self.max_size = max_size

    def add(self, x: torch.Tensor, y: torch.Tensor, metadata: pd.DataFrame, node_indices: torch.Tensor, neighborhood_index: torch.Tensor):
        self.buffer.append((x.cpu(), y.cpu(), metadata, node_indices.cpu(), neighborhood_index.cpu()))
        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)


    def sample(self, batch_size: int) -> list[tuple[torch.Tensor, torch.Tensor, pd.DataFrame, torch.Tensor, torch.Tensor]]:
        if not self.buffer:
            return []
        weights = np.linspace(1, 2, len(self.buffer))
        probs = weights / weights.sum()
        sampled = np.random.choice(len(self.buffer), size=min(batch_size, len(self.buffer)), replace=False, p=probs)
        return [self.buffer[i] for i in sampled]

    def __len__(self):
        return len(self.buffer)
    



# ---------------------------------------------------------------------
# 2. Pre‑training on two 5‑year windows
# ---------------------------------------------------------------------

def _batch_train_loop(
    hetero_graph: HeteroData,
    df: pd.DataFrame,
    macro_features: dict,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    epochs: int,
    batch_size: int,
    device: torch.device,
):
    """Run mini‑batch sub‑graph training exactly like the original code."""

    trx_x = hetero_graph["transaction"].x
    full_y_true = hetero_graph["transaction"].y.squeeze(-1).cpu()
    n_nodes = trx_x.size(0)
    full_edge_index = hetero_graph["transaction", "to", "transaction"].edge_index.to(device)

    indices = torch.arange(n_nodes)
    data_loader = DataLoader(TensorDataset(indices), batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        model.train(); batch_losses = []
        batch_preds = np.zeros(n_nodes)
        for (batch_idx_tensor,) in data_loader:
            batch_idx = batch_idx_tensor.view(-1).to(device)
            
            if batch_idx.ndim == 0:
                batch_idx = batch_idx.unsqueeze(0)
            first_year = df.iloc[batch_idx[0].item()]["YEAR"]
            batch_neigh_feats = macro_features.get(first_year, macro_features[max(macro_features.keys())]).to(device)

            
            optimizer.zero_grad(set_to_none=True)

            batch_subgraph, batch_pos = get_batch_subgraph(
                    batch_idx = batch_idx,
                    hetero_graph= hetero_graph,
                    transactions_df=df,
                    batch_neighborhood_features=batch_neigh_feats,
                )

            # ---- forward / backward ----
            out = model(batch_subgraph).squeeze(-1)

            # Compute loss only on batch nodes
            target = batch_subgraph['transaction'].y
            if target.dim() == 0:
                target = target.unsqueeze(0)

            if batch_pos.dim() == 0:
                batch_pos = batch_pos.unsqueeze(0)
            
            if out.dim() == 0:
                out = out.unsqueeze(0)

            loss = criterion(out[batch_pos], target[batch_pos].squeeze(-1))

            loss.backward()
            optimizer.step()

            batch_losses.append(loss.item())
        
        model.eval()
        with torch.no_grad():
            full_preds = model(hetero_graph.to(device)).cpu()
            mape = mean_absolute_percentage_error(full_y_true, full_preds).item()
            print(f"    epoch {epoch+1:>3}/{epochs} | loss={np.mean(batch_losses):.4f} | MAPE={mape:.2f}%")
        torch.cuda.empty_cache()
        # if (epoch + 1) % 1== 0:
        #     mape = np.mean(np.abs((np.exp(y_true_full) - np.abs(batch_preds)) / (np.abs(y_true_full) + 1e-8))) * 100
        #     print(f"    epoch {epoch+1:>3}/{epochs} | loss={np.mean(batch_losses):.4f}| MAPE={mape:.2f}%")
        # torch.cuda.empty_cache()


def pretrain_two_windows(
    transactions: pd.DataFrame,
    node_features: dict[int, torch.Tensor],
    neighbor_edge_index: torch.LongTensor,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: "StreamingScaler",
    device: torch.device,
    window_years: int = 5,
    epochs_per_window: int = 200,
    batch_size: int = 128,
) -> pd.DataFrame:
    """Pre‑train on first two windows *with mini‑batch sub‑graph training*."""

    # Add DATE for slicing
    df = transactions.copy()
    df["DATE"] = pd.to_datetime(dict(year=df["YEAR"], month=df["MONTH"], day=1))
    df.sort_values("DATE", inplace=True)

    start_dates = [df["DATE"].min(), df["DATE"].min() + relativedelta(months=1)]
    criterion = nn.MSELoss(); model.to(device)

    for win_idx, win_start in enumerate(start_dates):
        win_end = win_start + relativedelta(years=window_years)
        win_df = df[(df["DATE"] >= win_start) & (df["DATE"] < win_end)].copy().reset_index(drop=True)
        print(f"  Pre‑train window {win_idx+1}/2 | {win_start.date()}–{(win_end-relativedelta(days=1)).date()} | {len(win_df)} samples")

        # --- scaler fit/update & feature tensor ---------------------
        x_raw = win_df.drop(columns=["LOG_KOOPSOM", "BUURTCODE", "TRANSID", "DATE"]).values.astype(np.float32)
        scaler.partial_fit(x_raw)
        x_tensor = torch.tensor(x_raw, dtype=torch.float32)

        # --- build hetero graph once per window --------------------
        year_macro = int(win_df.iloc[0]["YEAR"])
        macro_feats = node_features.get(year_macro, node_features[max(node_features.keys())])
        neigh_df = pd.DataFrame(macro_feats.numpy()).assign(BUURTCODE=lambda d: range(len(d)))

        window_graph = build_hetero_graph(
            transaction_df=win_df.drop(columns=["DATE"]),
            neighborhood_df=neigh_df,
            test_mask=None,
            neighbor_edge_index=neighbor_edge_index,
        ).to(device)

        # --- mini‑batch training -----------------------------------
        _batch_train_loop(window_graph, win_df.drop(columns=["DATE"]), node_features, model, optimizer, criterion, epochs_per_window, batch_size, device)

    # Return remaining rows (after second window)
    cutoff = start_dates[1] + relativedelta(years=window_years)
    remaining = df[df["DATE"] >= cutoff].drop(columns=["DATE"]).copy()
    return remaining
# ———————————————————————————————————————————————————————————————————————
# 5. Online loop with per‑month logging
# ———————————————————————————————————————————————————————————————————————

def mean_absolute_percentage_error(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs((torch.exp(y_true) - torch.exp(y_pred)) / (torch.exp(y_true) + 1e-8))) * 100


def monthly_stream(df: pd.DataFrame) -> Generator[pd.DataFrame, None, None]:
    for (_, _), g in df.groupby(["YEAR", "MONTH"], sort=True):
        yield g.copy()


def train_online(
    transactions: pd.DataFrame,
    node_features: dict[int, torch.Tensor],
    neighbor_edge_index: torch.LongTensor,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: "StreamingScaler",
    device: torch.device,
    epochs_per_month: int = 10,
    batch_size: int = 128,
    log_every: int = 1,
):
    """Streaming updates with batched sub‑graph training for each month."""

    criterion = nn.MSELoss(); step = 0; ema_mse = ema_mape = None
    all_month_preds: List[pd.DataFrame] = []
    model.to(device)
    replay_buffer = ReplayBuffer(max_size=50)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

    for month_df in monthly_stream(transactions):
        month_df = month_df.reset_index(drop=True)
        year, month = int(month_df.iloc[0]["YEAR"]), int(month_df.iloc[0]["MONTH"])
        window_id = f"{year}-{month:02d}"

        # ---------------- feature scaling & graph build -------------
        x_raw = month_df.drop(columns=["LOG_KOOPSOM", "BUURTCODE", "TRANSID"]).values.astype(np.float32)
        scaler.partial_fit(x_raw)
        x_tensor = torch.tensor(x_raw, dtype=torch.float32)

        macro = node_features.get(year, node_features[max(node_features.keys())])
        neigh_df = pd.DataFrame(macro.numpy()).assign(BUURTCODE=lambda d: range(len(d)))
        g = build_hetero_graph(month_df, neigh_df, None, neighbor_edge_index).to(device)

        # ---------------- evaluation pass (pre‑update) -------------
        model.eval()
        with torch.no_grad():
            y_pred_eval = model(g).cpu().numpy()
            preds_df = pd.DataFrame({
                "window_start": window_id,
                "BUURTCODE": month_df["BUURTCODE"].values,
                "YEAR": month_df["YEAR"].values,
                "MONTH": month_df["MONTH"].values,
                "TRANSID": month_df["TRANSID"].values,
                "y_true": month_df["LOG_KOOPSOM"].values,
                "y_pred": y_pred_eval,
            })
            all_month_preds.append(preds_df)
        
        # replay_buffer.add(x_tensor, g["transaction"].y.squeeze(-1), month_df[["BUURTCODE", "TRANSID"]])
        node_indices = torch.arange(x_tensor.size(0))  # or actual indices if you have them
        neighborhood_indices = g["transaction"].neighborhood_index[node_indices]
        replay_buffer.add(x_tensor, g["transaction"].y.squeeze(-1), month_df[["BUURTCODE", "TRANSID"]], node_indices, neighborhood_indices)
        print(replay_buffer)


        # ---------------- training epochs --------------------------
        print(f"\n📅  {window_id}: training {epochs_per_month} epochs …")
        _batch_train_loop(g, month_df, node_features, model, optimizer, criterion, epochs_per_month, batch_size, device)

        # ---------------- live EMA metrics -------------------------
        with torch.no_grad():
            updated_pred = model(g).detach()
            mse = F.mse_loss(updated_pred, g["transaction"].y.squeeze(-1))
            mape_val = mean_absolute_percentage_error(g["transaction"].y.squeeze(-1).cpu(), updated_pred.cpu())
            ema_mse = mse if ema_mse is None else 0.95 * ema_mse + 0.05 * mse
            ema_mape = mape_val if ema_mape is None else 0.95 * ema_mape + 0.05 * mape_val
        if step % log_every == 0:
            print(f"      EMA‑MSE={ema_mse:.4f} | EMA‑MAPE={ema_mape:.2f}% | month size={len(month_df)}")

        step += 1; torch.cuda.empty_cache()

    # ---------------- save predictions ----------------------------
    os.makedirs("./outputs", exist_ok=True)
    pd.concat(all_month_preds, ignore_index=True).to_csv("./outputs/online_month_predictions.csv", index=False)

# ———————————————————————————————————————————————————————————————————————
# 6. Main entry point (paths may need editing)
# ———————————————————————————————————————————————————————————————————————

def run_exp_online(data_path: str):
    # --- load data -----------------------------------------------------
    node_feats, neigh_edge = load_graph_data(
        os.path.join(data_path, "all_neighborhood_features_rotterdam.csv"),
        os.path.join(data_path, "rotterdam_adj_2023.csv"),
    )
    trans = load_transaction_data(os.path.join(data_path, "rotterdam_transaction_data.csv"))

    # --- model & optimiser --------------------------------------------
    t_in = trans.shape[1] - 3  # minus BUURTCODE/YEAR/MONTH/LOG_KOOPSOM
    m_in = next(iter(node_feats.values())).shape[1]
    model = HierarchicalHeteroGNN(t_in, m_in, 64, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scaler = StreamingScaler()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Phase 1: pre‑training ----------------------------------------
    remaining_trans = pretrain_two_windows(
        transactions=trans,
        node_features=node_feats,
        neighbor_edge_index=neigh_edge,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        device=device,
        window_years=5,
        epochs_per_window=500,
        batch_size=64,
    )

    # --- Phase 2: streaming updates -----------------------------------
    print("\n🚀  Switching to streaming updates …")
    train_online(
        transactions=remaining_trans,
        node_features=node_feats,
        neighbor_edge_index=neigh_edge,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        device=device,
        log_every=1,
        epochs_per_month=100,
        batch_size=8,
    )

    # --- Save artefacts ------------------------------------------------
    os.makedirs("./outputs", exist_ok=True)
    torch.save(model.state_dict(), "./outputs/hierarchical_gnn_online.pt")
    scaler.save("./outputs/streaming_scaler.pkl")
    print("✅  Finished.  Model and predictions stored in ./outputs/")