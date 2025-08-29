import torch
from collections import defaultdict

def get_transaction_edges(transaction_df):
    """
    Generate edge_index for transactions: all transactions in the same neighborhood
    are fully connected with weights based on spatial proximity.
    Returns edge_index (2, num_edges) as torch.Tensor.
    """
    
    idx = transaction_df.index.to_numpy()
    buurtcodes = transaction_df['BUURTCODE'].to_numpy()
    edges = []

    # Group by neighborhood
    buurt_to_indices = defaultdict(list)
    for i, buurt in zip(idx, buurtcodes):
        buurt_to_indices[buurt].append(i)
        
    # For each neighborhood, connect all pairs with weights
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

