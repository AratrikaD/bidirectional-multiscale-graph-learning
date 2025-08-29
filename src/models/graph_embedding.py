import torch
from torch_geometric.nn import GCNConv, Linear
import torch.nn as nn

class NeighborhoodEmbedder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(NeighborhoodEmbedder, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x


class PricePredictor(torch.nn.Module):
    def __init__(self, gnn_out_channels, hidden_dims, transaction_dims):
        super(PricePredictor, self).__init__()
        self.fc1 = Linear(gnn_out_channels+transaction_dims, hidden_dims)
        self.fc2 = Linear(hidden_dims, 1)
    
    def forward(self, neighbor_embedding, transaction):
        x = torch.cat([neighbor_embedding, transaction], dim=1).float()
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class SpatioTemporalGNN(nn.Module):
	def __init__(self, input_dim, hidden_dim, output_dim, num_timesteps):
		"""
		input_dim: Dimension of node features per timestep.
		hidden_dim: Hidden size for GRU.
		output_dim: Final embedding dimension (output of GCN).
		num_timesteps: Number of timesteps in the historical sequence.
		"""
		super(SpatioTemporalGNN, self).__init__()
		self.num_timesteps = num_timesteps
		# Temporal component: GRU processes features over time for each node.
		self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
		# Spatial component: GCN propagates the temporal representation over the graph.
		self.gcn = GCNConv(hidden_dim, output_dim)

	def forward(self, node_features, edge_index):
		"""
		node_features: Tensor of shape (N, T, F) for all N nodes.
	    edge_index: Graph connectivity tensor (shape: [2, num_edges]).
		"""
		_, hidden_state = self.gru(node_features)  # shape: (1, N, hidden_dim)
		hidden_state = hidden_state.squeeze(0) # shape: (N, hidden_dim)
		# Apply the GCN to propagate information among nodes
		node_embeddings = self.gcn(hidden_state, edge_index)
		return node_embeddings

class EmbeddingModel(torch.nn.Module):
    def __init__(self, in_channels, gnn_hidden_channels, out_channels, nn_hidden_channels, transaction_dims):
        super(EmbeddingModel, self).__init__()
        self.gnn = NeighborhoodEmbedder(in_channels, gnn_hidden_channels, out_channels)
        self.predictor = PricePredictor(out_channels, nn_hidden_channels, transaction_dims)
    
    def forward(self, neighbor_features, transaction_features, edge_index, neighbor_id):
        graph_embedding = self.gnn(neighbor_features, edge_index)
        y = self.predictor(graph_embedding[neighbor_id], transaction_features)
        return y

class EmbeddingModelDynamic(torch.nn.Module):
    def __init__(self, in_channels, gnn_hidden_channels, out_channels, nn_hidden_channels, num_timesteps, transaction_dims):
        super(EmbeddingModelDynamic, self).__init__()
        self.gnn = SpatioTemporalGNN(in_channels, gnn_hidden_channels, out_channels, num_timesteps)
        self.predictor = PricePredictor(out_channels, nn_hidden_channels, transaction_dims)
    
    def forward(self, neighbor_features, transaction_features, edge_index, neighbor_id):
        graph_embedding = self.gnn(neighbor_features, edge_index)
        y = self.predictor(graph_embedding[neighbor_id], transaction_features)
        return y

class NeuralNet(torch.nn.Module):
    def __init__(self, hidden_dims, transaction_dims):
        super(NeuralNet, self).__init__()
        self.fc1 = Linear(transaction_dims, hidden_dims)
        self.fc2 = Linear(hidden_dims, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

