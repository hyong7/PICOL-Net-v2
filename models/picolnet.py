import torch.nn as nn
from .ssp import SSPHeader
from .predictor import MambaPredictor
from src.config import ModelConfig
from .ecm import ECMNet


class PicolNet(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.ssp = None
        predictor_input_dim = config.feature_dim
        self.ssp = SSPHeader(
            feature_dim=config.feature_dim,
            window=config.window,
            d_model=config.ssp.d_model,
            num_heads=config.ssp.num_heads,
            num_layers=config.ssp.num_layers,
            dropout=config.ssp.dropout,
        )
        predictor_input_dim = config.ssp.d_model
        self.predictor = MambaPredictor(
            input_dim=predictor_input_dim,
            hidden_dim=config.predictor.hidden_dim,
            num_layers=config.predictor.num_layers,
            d_state=config.predictor.d_state,
            d_conv=config.predictor.d_conv,
            expand=config.predictor.expand,
            dropout=config.predictor.dropout,
        )
        self.ecm = None
        self.ecm = ECMNet(
            hidden_dim=config.predictor.hidden_dim,
            eps=config.ecm.eps,
            current_sign=config.ecm.current_sign,
            r_min=config.ecm.r_min,
            c_min=config.ecm.c_min,
            r_base=config.ecm.r_base,
            c_base=config.ecm.c_base,
        )
        self.detach_ecm_hidden = False

    def forward(self, x, pinn=None, memory=None):
        ssp_out = self.ssp(x)
        token_latents = ssp_out["hidden"]
        ssp_latent = ssp_out["latent"]
        ssp_recon = ssp_out["recon"]
        pred_out = self.predictor(token_latents, memory=memory)
        out = {
            "ssp_hidden": token_latents,
            "ssp_latent": ssp_latent,
            "ssp_recon": ssp_recon,
            "pred": pred_out["output"],
            "predictor_hidden": pred_out["hidden"],
            "predictor_latent": pred_out["latent"],
            "memory_latent": pred_out["memory_latent"],
        }
        if pred_out["memory"] is not None:
            out["memory"] = pred_out["memory"]
        if pinn is not None:
            ecm_hidden = pred_out["hidden"]
            ecm_out = self.ecm(ecm_hidden, pinn)
            out.update(ecm_out)
        return out
