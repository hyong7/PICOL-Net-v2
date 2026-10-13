import math
import torch
import torch.nn as nn


class PositionEncoding(nn.Module):

    def __init__(self, max_len, d_model):
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class SSPHeader(nn.Module):

    def __init__(
        self,
        feature_dim,
        window=50,
        d_model=128,
        num_heads=4,
        num_layers=2,
        dropout=0.1,
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.feature_dim = feature_dim
        self.window = window
        self.input_proj = nn.Linear(feature_dim, d_model)
        self.pos_encoder = PositionEncoding(max_len=window, d_model=d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, window * feature_dim),
        )

    def pool(self, x):
        return x.mean(dim=1)

    def forward(self, x):
        hidden = self.input_proj(x)
        hidden = self.pos_encoder(hidden)
        hidden = self.transformer_encoder(hidden)
        latent = self.pool(hidden)
        recon = self.decoder(latent).view(-1, self.window, self.feature_dim)
        return {"recon": recon, "latent": latent, "hidden": hidden}
