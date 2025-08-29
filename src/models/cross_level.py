import torch
import torch.nn as nn
import torch.nn.functional as F

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


# class CrossLevelInteraction(nn.Module):
#     def __init__(self, trans_dim, macro_dim, hidden_dim):
#         super().__init__()
#         # Projection layers
#         self.trans_proj = nn.Linear(trans_dim, hidden_dim)
#         self.macro_proj = nn.Linear(macro_dim, hidden_dim)


#         # Dynamic gating
#         self.W_gamma = nn.Linear(hidden_dim, hidden_dim)
#         self.U_gamma = nn.Linear(hidden_dim, hidden_dim)
#         self.V_gamma = nn.Linear(hidden_dim, hidden_dim)

#     def forward(self, trans_embed, macro_embed, edge_index):
#         # Project embeddings
#         h_t = self.trans_proj(trans_embed)
#         h_m = self.macro_proj(macro_embed)
#         global_context = macro_embed.mean(dim=0)  
#         src, dst = edge_index  # transaction -> neighborhood

        

#         # Neighborhood aggregation

#         macro_agg = torch.zeros_like(h_m)
#         macro_agg.index_add_(0, dst, h_t[src])

#         # Fuse with existing macro embeddings
#         fused_macro = macro_agg  # (could also include a residual)

#         # === Dynamic top-down gating ===
#         aligned_macro = fused_macro[dst]
#         gamma = torch.sigmoid(
#             self.W_gamma(h_t[src]) +
#             self.U_gamma(aligned_macro) +
#             self.V_gamma(global_context).expand_as(h_t[src])
#         )

#         trans_out = gamma * h_t[src] + (1 - gamma) * aligned_macro

#         return trans_out, fused_macro
