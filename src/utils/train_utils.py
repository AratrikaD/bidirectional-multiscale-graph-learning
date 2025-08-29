import torch
from torch.utils.data import DataLoader, TensorDataset

def scale_window_data(train_data, test_data, scaler):
    train_scaled = scaler.fit_transform(train_data)
    test_scaled = scaler.transform(test_data)

    return torch.tensor(train_scaled, dtype=torch.float32), torch.tensor(test_scaled, dtype=torch.float32)
# ### 2. Data Preparation for Sliding Windows

def split_train_test(data, test_ratio=0.2):
    split_idx = int(len(data) * (1 - test_ratio))
    return data[:split_idx], data[split_idx:]

### 3. Utility for Mini-Batch Creation
def create_batches(trans_feats, node_idx, time_feats, prices, batch_size):
    dataset = TensorDataset(trans_feats, node_idx, time_feats, prices)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
