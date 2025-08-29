from src.models.graph_embedding import EmbeddingModel
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import mlflow




def train_epoch(model, loader, optimizer, edge_index, node_features, buurt_idx_map, loss_fn,  device="cpu"):
    """ 
    Trains model for one epoch using the specified data loader and optimizer.

    Args:
        gnn: The gnn embedding model to be trained
        predictor: The neural network predictor to be trained
        loader (DataLoader): The DataLoader containing the training data
        optimizer: Theoptimizer used for training
        edge_index: The adjaceny matrix fo the neighborhoods
        node_features: The ids of the nodes (Buurtcodes mapped to indices)
        buurt_idx_map: Maps the Neighborhood codes to indices
        loss_fn: Loss function used (MSELoss mainly)
        device: The device used for training the model (default: cpu)

    Returns:
        float: The mean loss valueover all the batches in the DataLoader.
    """

    model.to(device)
   
    model.train()
    
    total_loss = 0
    total_mape = 0
    for data in loader:
        
        optimizer.zero_grad()

        neighborhood_ids= [buurt_idx_map[spatial_code] for spatial_code in data["spatial_level"]]
        neighborhood_ids = torch.tensor(neighborhood_ids,dtype=torch.int)
        predicted_price = model(node_features.to(device), data["transaction_vector"].float().to(device), edge_index.to(device), neighborhood_ids.to(device))
        
        actual_price = torch.tensor(data["target_price"],dtype=torch.float32).unsqueeze(1).to(device)
        

        loss = loss_fn(predicted_price,actual_price)
        mae = nn.L1Loss()(torch.exp(predicted_price), torch.exp(actual_price))
        mape = torch.mean(torch.abs(torch.exp(actual_price)-torch.exp(predicted_price))/torch.exp(actual_price))
    
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_mape += mape.item()
    
    return total_loss/len(loader), total_mape/len(loader)

def evaluate_epoch(model, loader, loss_fn, edge_index, node_features, buurt_idx_map, device="cpu"):
    """
    Evaluates the performance of a trained neural network model on a dataset using the specified data loader

    """

    model.to(device)
    
    model.eval()

    total_loss = 0
    total_mape = 0
    
    with torch.no_grad():
        for data in loader:
           
            neighborhood_ids= [buurt_idx_map[spatial_code] for spatial_code in data["spatial_level"]]
            neighborhood_ids = torch.tensor(neighborhood_ids, dtype=torch.int)
            predicted_price = model(node_features.to(device), data["transaction_vector"].to(device), edge_index.to(device), neighborhood_ids.to(device))

            actual_price = torch.tensor(data["target_price"],dtype=torch.float32).unsqueeze(1).to(device)
        
            loss = loss_fn(predicted_price,actual_price)
            
            mae = nn.L1Loss()(torch.exp(predicted_price), torch.exp(actual_price))
            mape = torch.mean(torch.abs(torch.exp(actual_price)-torch.exp(predicted_price))/torch.exp(actual_price))
            total_loss += loss.item()
            total_mape += mape.item()
            
    return total_loss/len(loader), total_mape/len(loader)

def train(config_path, edge_index, train_dataset, test_dataset, device, hyperparameters):
    
    print(device)
    torch.manual_seed(hyperparameters["seed"])
    batch_size=hyperparameters["batch_size"]
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size,shuffle=True)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size,shuffle=True)
    num_features = 1

    transaction_dim=train_dataset.__getnumfeatures__()

    model = EmbeddingModel(in_channels=num_features, gnn_hidden_channels=hyperparameters["gnn_hidden_channels"], out_channels=8, nn_hidden_channels=hyperparameters["nn_hidden_channels"], transaction_dims=transaction_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparameters["learning_rate"])
    loss_fn = torch.nn.MSELoss()
    

    num_epochs = 100
    buurt_ids = edge_index.columns.values

    buurt_idx_map = {k:int(v) for v,k in enumerate(buurt_ids)}
    

    edge_index = edge_index.to_numpy("float")
    edge_index = np.triu(edge_index)
    edge_index= torch.tensor(edge_index, dtype=torch.float32)
    edge_index_tensor = edge_index.to_sparse_csr()
    node_features = torch.tensor(list(buurt_idx_map.values()), dtype=torch.float).unsqueeze(1)
    
    train_losses, test_losses = [], []
    train_mapes, test_mapes = [], []

    # Training loop
    for epoch in range(num_epochs):
        train_loss, train_mape = train_epoch(model, train_data_loader, optimizer, edge_index_tensor, node_features, buurt_idx_map, loss_fn, device)

        test_loss, test_mape = evaluate_epoch(model, test_data_loader, loss_fn, edge_index_tensor, node_features, buurt_idx_map, device)

        train_losses.append(train_loss)
        train_mapes.append(train_mape)

        test_losses.append(test_loss)
        test_mapes.append(test_mape)
        print(f'Epoch {epoch+1}/{num_epochs}: Train Loss = {train_loss:.4f}, Train MAPE = {train_mape:.4f},   Test Loss = {test_loss:.4f}, Test MAPE = {test_mape:.4f}')
        metrics = {
            "train_loss": train_loss,
            "test_loss": test_loss,
            "train_mape": train_mape,
            "test_mape": test_mape,
        }
        mlflow.log_metrics(metrics)
    fig = plot_learning_curves(num_epochs, train_losses, test_losses, train_mapes, test_mapes)
    mlflow.log_figure(fig, "learning_curves.png")


def plot_learning_curves(num_epochs, train_losses, test_losses, train_mapes, test_mapes):
    # Visualize Results
    
    
    fig, axs = plt.subplots(2, 2)
    fig.set_size_inches(10, 10)
    axs[0, 0].plot(range(1, num_epochs+1), train_losses, label='Training Loss')
    axs[0, 0].plot(range(1, num_epochs+1), test_losses, label='Test Loss')
    axs[0, 0].set_title('Training and Test Losses (MSE)')
    axs[0, 1].plot(range(1, num_epochs+1), train_mapes, label='Training MAPE')
    axs[0, 1].plot(range(1, num_epochs+1), test_mapes, label='Test MAPE')
    axs[0, 1].set_title('Training and Test MAPE')
    axs[1, 0].plot(range(1, num_epochs+1), train_losses, label='Training Loss')
    axs[1, 0].plot(range(1, num_epochs+1), test_losses, label='Test Loss')
    axs[1, 0].set_yscale("log")
    axs[1, 0].set_title('Training and Test Log Losses (MSE)')
    axs[1, 1].plot(range(1, num_epochs+1), train_mapes, label='Training MAPE')
    axs[1, 1].plot(range(1, num_epochs+1), test_mapes, label='Test MAPE')
    axs[1, 1].set_yscale("log")
    axs[1, 1].set_title('Training and Test Log MAPE')

    for ax in axs.flat:
        ax.set(xlabel='x-label', ylabel='y-label')

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()
    plt.show()
    return fig
   
def create_sliding_windows(df, window_size=3):
    """
    Creates sliding window splits.
    For each window, the training set is composed of the first 'window_size' years,
    the validation set is the next year, and the test set is the following year.
    """
    years = sorted(df['year'].unique())
    splits = []
    # Ensure we have at least window_size + 2 years
    for i in range(len(years) - (window_size + 1)):
        train_years = years[i:i+window_size]
        val_year = years[i+window_size]
        test_year = years[i+window_size+1]
        train_df = df[df['year'].isin(train_years)].reset_index(drop=True)
        val_df = df[df['year'] == val_year].reset_index(drop=True)
        test_df = df[df['year'] == test_year].reset_index(drop=True)
        splits.append((train_df, val_df, test_df))
    return splits

def train_sliding_window(model, predictor, df, node_features, edge_index, epochs=5, window_size=3, batch_size=4, learning_rate=0.001):
    """
    Trains the model using a sliding window approach.
    """
    splits = create_sliding_windows(df, window_size=window_size)
    criterion = nn.MSELoss()

    for step, (train_df, val_df, test_df) in enumerate(splits):
        print("\\n--- Sliding Window Step {} ---".format(step + 1))
        print("Train Years:", sorted(train_df['year'].unique()))
        print("Validation Year:", val_df['year'].unique()[0])
        print("Test Year:", test_df['year'].unique()[0])

        # Create datasets for this window
        train_dataset = HousePriceDataset(train_df, node_features, edge_index)
        val_dataset = HousePriceDataset(val_df, node_features, edge_index)
        test_dataset = HousePriceDataset(test_df, node_features, edge_index)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # (Re)initialize the optimizer for each window if desired.
        optimizer = optim.Adam(list(model.parameters()) + list(predictor.parameters()), lr=learning_rate)

        # Train for a number of epochs within the current window.
        for epoch in range(epochs):
            model.train()
            predictor.train()
            train_loss = 0.0
            for batch in train_loader:
                optimizer.zero_grad()
                neighborhood_ids, transaction_features, target_prices = batch
                # Generate embeddings for the entire graph.
                node_embeddings = model(node_features, edge_index)  # shape: (N, output_dim)
                # Extract embeddings for the batch (using neighborhood_ids)
                batch_embeddings = node_embeddings[neighborhood_ids]  # shape: (batch_size, output_dim)
                predictions = predictor(batch_embeddings, transaction_features)
                loss = criterion(predictions, target_prices.unsqueeze(1))
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            avg_train_loss = train_loss / len(train_loader)
            print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}")

            # Validation phase
            model.eval()
            predictor.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    neighborhood_ids, transaction_features, target_prices = batch
                    node_embeddings = model(node_features, edge_index)
                    batch_embeddings = node_embeddings[neighborhood_ids]
                    predictions = predictor(batch_embeddings, transaction_features)
                    loss = criterion(predictions, target_prices.unsqueeze(1))
                    val_loss += loss.item()
            avg_val_loss = val_loss / len(val_loader)
            print(f"Epoch {epoch+1}, Val Loss: {avg_val_loss:.4f}")

        # Final test evaluation for this window
        model.eval()
        predictor.eval()
        test_loss = 0.0
        with torch.no_grad():
            for batch in test_loader:
                neighborhood_ids, transaction_features, target_prices = batch
                node_embeddings = model(node_features, edge_index)
                batch_embeddings = node_embeddings[neighborhood_ids]
                predictions = predictor(batch_embeddings, transaction_features)
                loss = criterion(predictions, target_prices.unsqueeze(1))
                test_loss += loss.item()
        avg_test_loss = test_loss / len(test_loader)
        print(f"Sliding Window Step {step+1}, Test Loss: {avg_test_loss:.4f}")

# def visualize_embedding():
#     embeddings = gnn_embeddings.detach().cpu().numpy()
#     labels = df_train['log_prices'].
