import torch
import torch.nn as nn
from mamba_ssm import Mamba


class MambaBlock(nn.Module):

    def __init__(self, hidden_dim, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.mamba = Mamba(
            d_model=hidden_dim, d_state=d_state, d_conv=d_conv, expand=expand
        )
        self.dropout = nn.Dropout(dropout)

    def cpu_forward(self, x):
        xz = self.mamba.in_proj(x)
        x_part, z = xz.chunk(2, dim=-1)
        conv = self.mamba.conv1d(x_part.transpose(1, 2))[..., : x.size(1)]
        conv = self.mamba.act(conv).transpose(1, 2)
        gated = conv * torch.sigmoid(z)
        return self.mamba.out_proj(gated)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        if x.device.type == "cpu":
            x = self.cpu_forward(x)
        else:
            x = self.mamba(x)
        x = self.dropout(x)
        return x + residual


class MambaPredictor(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_dim=128,
        num_layers=2,
        d_state=16,
        d_conv=4,
        expand=2,
        dropout=0.1,
    ):
        super().__init__()
        self.readout = "last"
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.mamba_blocks = nn.ModuleList(
            [
                MambaBlock(hidden_dim, d_state, d_conv, expand, dropout)
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def pool(self, x):
        return x[:, -1]

    def forward(self, x, memory=None):
        hidden = self.input_proj(x)
        for mamba_block in self.mamba_blocks:
            hidden = mamba_block(hidden)
        hidden = self.norm(hidden)
        latent = self.pool(hidden)
        memory_info = None
        head_latent = latent
        if memory is not None:
            head_latent, memory_info = memory.correct_latent(latent)
        output = self.head(head_latent).squeeze(-1)
        if memory is not None:
            output = memory.correct_output(output, memory_info)
        return {
            "output": output,
            "latent": latent,
            "memory_latent": head_latent,
            "hidden": hidden,
            "memory": memory_info,
        }
