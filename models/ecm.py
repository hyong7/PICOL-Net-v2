import torch
import torch.nn as nn
import torch.nn.functional as F


class OCVNet(nn.Module):

    def __init__(self, hidden_dim=32, v_min=3.0, v_max=4.3):
        super().__init__()
        self.v_min = v_min
        self.v_max = v_max
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        x = self.net(x.unsqueeze(-1))
        x = torch.sigmoid(x).squeeze(-1)
        return self.v_min + x * (self.v_max - self.v_min)


class ECMNet(nn.Module):

    def __init__(
        self,
        hidden_dim=32,
        eps=1e-06,
        current_sign=1.0,
        r_min=1e-05,
        c_min=0.001,
        r_base=0.001,
        c_base=10000.0,
    ):
        super().__init__()
        self.eps = eps
        self.current_sign = current_sign
        self.r_min = r_min
        self.c_min = c_min
        self.use_physical_scaling = True
        self.r_base = r_base
        self.c_base = c_base
        self.param_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),
        )
        self.ocv_net = OCVNet(hidden_dim=hidden_dim)

    def estimate_params(self, hidden):
        raw = self.param_head(hidden)
        positive = F.softplus(raw) + self.eps
        r0 = positive[..., 0] * self.r_base + self.r_min
        rp = positive[..., 1] * self.r_base + self.r_min
        cp = positive[..., 2] * self.c_base + self.c_min
        return (r0, rp, cp)

    def recon_voltage(self, current, soc, dt, r0, rp, cp):
        b, w, s = current.shape
        tau = rp * cp + self.eps
        tau = tau.unsqueeze(-1)
        alpha = torch.exp(-dt / tau).clamp(0.0, 1.0)
        ocv = self.ocv_net(soc)
        up = torch.zeros(b, w, device=current.device, dtype=current.dtype)
        voltage_step = []
        for k in range(s):
            vk = ocv[..., k] - current[..., k] * r0 - up
            voltage_step.append(vk)
            alpha_k = alpha[..., k]
            up = alpha_k * up + rp * (1.0 - alpha_k) * current[..., k]
        voltage_pred = torch.stack(voltage_step, dim=-1)
        return voltage_pred

    def forward(self, predictor_hidden, pinn):
        time = pinn[..., 0]
        soc = pinn[..., 1]
        current = pinn[..., 2]
        mean_v = pinn[..., 3]
        dt = pinn[..., 4]
        del time
        soc = (soc / 100.0).clamp(0.0, 1.0)
        current = current * self.current_sign
        dt = torch.clamp(dt, min=0.0)
        r0, rp, cp = self.estimate_params(predictor_hidden)
        voltage_pred = self.recon_voltage(current, soc, dt, r0, rp, cp)
        return {
            "voltage_pred": voltage_pred,
            "voltage_true": mean_v,
            "r0": r0,
            "rp": rp,
            "cp": cp,
        }


def ecm_voltage_loss(voltage_pred, voltage_true):
    mask = torch.isfinite(voltage_pred) & torch.isfinite(voltage_true)
    if not mask.any():
        return voltage_pred.new_tensor(0.0)
    return F.mse_loss(voltage_pred[mask], voltage_true[mask])
