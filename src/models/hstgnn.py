import torch
import torch.nn as nn
from src.models.cross_level import CrossLevelInteraction
from src.models.macro_encoder import MacroEncoder
from src.models.micro_encoder import MicroEncoder

class HierarchicalHeteroGNN(nn.Module):
    def __init__(self, trans_in, macro_in, hidden_dim, out_dim):
        super().__init__()
        self.micro_encoder = MicroEncoder(trans_in, hidden_dim)
        self.macro_encoder = MacroEncoder(macro_in, hidden_dim)
        self.cross_layer = CrossLevelInteraction(hidden_dim, hidden_dim, hidden_dim)
        self.predictor = nn.Linear(hidden_dim, out_dim)
        self.prev_macro_state = None

    def forward(self, data):
        trans_x = data['transaction'].x
        macro_x = data['neighborhood'].x
        trans_edge_index = data['transaction', 'to', 'transaction'].edge_index
        macro_edge_index = data['neighborhood', 'to', 'neighborhood'].edge_index


        trans_to_neigh =  data['transaction'].neighborhood_index

        # data['transaction', 'belongs_to', 'neighborhood'].edge_index
        # data['transaction'].neighborhood_index

        h_micro = self.micro_encoder(trans_x, trans_edge_index)
        h_macro_gated, h_macro_raw = self.macro_encoder(macro_x, macro_edge_index, self.prev_macro_state)

        h_trans_final, h_macro_final = self.cross_layer(h_micro, h_macro_gated, trans_to_neigh)

        self.prev_macro_state = h_macro_final.detach()

        out = self.predictor(h_trans_final).squeeze(-1)
        return out

    def forward_with_embeddings(self, data):
        trans_x = data['transaction'].x
        macro_x = data['neighborhood'].x
        trans_edge_index = data['transaction', 'to', 'transaction'].edge_index
        macro_edge_index = data['neighborhood', 'to', 'neighborhood'].edge_index
        trans_to_neigh = data['transaction'].neighborhood_index

        h_micro = self.micro_encoder(trans_x, trans_edge_index)
        h_macro_gated, h_macro_raw = self.macro_encoder(macro_x, macro_edge_index, self.prev_macro_state)
        h_trans_final, h_macro_final = self.cross_layer(h_micro, h_macro_gated, trans_to_neigh)

        self.prev_macro_state = h_macro_final.detach()

        preds = self.predictor(h_trans_final).squeeze(-1)
        return preds, h_trans_final
