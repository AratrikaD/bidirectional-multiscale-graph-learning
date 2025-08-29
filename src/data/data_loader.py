from torch.utils.data import Dataset
import json
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler

class HousePriceDataset(Dataset):
    def __init__(self, node_features, transaction_data, spatial_level,transaction_date, target_prices):
        self.node_features = node_features
        self.transaction_data = transaction_data
        self.spatial_level = spatial_level
        self.transaction_date = transaction_date
        self.target_prices = target_prices
    
    def __len__(self):
        return len(self.transaction_data)

    def __getitem__(self, idx):
        return {
            "node_features": self.node_features[idx],
            "transaction_vector": self.transaction_data[idx],
            "spatial_level": self.spatial_level[idx],
            "transaction_date": self.transaction_date[idx],
            "target_price": np.log(self.target_prices[idx])
        }

    def __getnumfeatures__(self):
        return len(self.transaction_data[0])
    
class HousePriceDataset(Dataset):
    def __init__(self, df, node_features, edge_index):
        """
        df: Pandas DataFrame with transaction data. Must include:
        - 'year'
        - 'neighborhood_id'
        - 'house_size' (or more features if desired)
        - 'price'
        node_features: Tensor (N, T, F) for all nodes.
        edge_index: Graph connectivity (used in the model; not needed per sample).
        """
        self.df = df.reset_index(drop=True)
        self.node_features = node_features  # available globally for the graph
        self.edge_index = edge_index

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        neighborhood_id = int(row['neighborhood_id'])
        # For simplicity, we use just one transaction feature (e.g., house size)
        transaction_features = torch.tensor([row['house_size']], dtype=torch.float32)
        target_price = torch.tensor(row['price'], dtype=torch.float32)
        # We return the neighborhood_id, transaction features, and target.
        return neighborhood_id, transaction_features, target_price

    
def dataframe_to_dataset(config_path, train_df, test_df):
    
    vars = json.load(open(config_path))
    # X_train = train_df[vars["transaction_vars"]]
    # X_test = test_df[vars["transaction_vars"]]
    X_train  = train_df.drop(columns=['TRANSID', "KOOPSOM", "LOG_KOOPSOM","BUURTCODE"])
    X_test = test_df.drop(columns=['TRANSID',"KOOPSOM","LOG_KOOPSOM", "BUURTCODE"])
    # imp_median = SimpleImputer(missing_values=np.nan, strategy='mean')
    # imp_median.fit(X_train)
    # X_train = imp_median.transform(X_train)
    # X_test = imp_median.transform(X_test)
    standard_scaler = RobustScaler()
    X_train = standard_scaler.fit_transform(X_train)
    X_test = standard_scaler.transform(X_test)

    train_data = HousePriceDataset(node_features=train_df[vars["neighborhood_vars"]].to_numpy(), transaction_data=X_train, spatial_level=train_df[vars['spatial_level']].to_numpy(), transaction_date=train_df[vars['transaction_date']].to_numpy(), target_prices=train_df[vars["target_price"]].to_numpy())
    test_data = HousePriceDataset(node_features=test_df[vars["neighborhood_vars"]].to_numpy(), transaction_data=X_test, spatial_level=test_df[vars['spatial_level']].to_numpy(), transaction_date=test_df[vars['transaction_date']].to_numpy(), target_prices=test_df[vars["target_price"]].to_numpy())
    return train_data, test_data
    




