import torch
from torch import nn

from .data import N_CLASSES, PAD


class CareTransformer(nn.Module):
    def __init__(self, max_context=24, d_model=32, heads=4, layers=2, feedforward=64, dropout=0.1):
        super().__init__()
        self.max_context = max_context
        self.token_embedding = nn.Embedding(N_CLASSES + 1, d_model, padding_idx=PAD)
        self.position_embedding = nn.Embedding(max_context, d_model)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=heads, dim_feedforward=feedforward,
                                            dropout=dropout, activation="gelu", batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers, enable_nested_tensor=False)
        self.head = nn.Linear(d_model, N_CLASSES)
        # Initialisation distincte des couches copiées par TransformerEncoder.
        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)
        with torch.no_grad():
            self.token_embedding.weight[PAD].zero_()

    def forward(self, tokens, lengths):
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        hidden = self.token_embedding(tokens) + self.position_embedding(positions)
        padding_mask = tokens.eq(PAD)
        # Tout le préfixe est passé. On ne calcule une cible qu'après ce préfixe.
        # Un masque causal triangulaire n'est donc pas nécessaire ici.
        encoded = self.encoder(hidden, src_key_padding_mask=padding_mask)
        last = encoded[torch.arange(len(tokens), device=tokens.device), lengths - 1]
        return self.head(last)


def architecture(config):
    return {key: config[key] for key in ["max_context", "d_model", "heads", "layers", "feedforward", "dropout"]}
