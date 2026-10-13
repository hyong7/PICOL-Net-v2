import math
from dataclasses import dataclass
from typing import Dict, Optional
import torch
import torch.nn.functional as F


@dataclass
class AnchorPool:
    keys: torch.Tensor
    latent_values: torch.Tensor
    residual_values: torch.Tensor
    gradients: torch.Tensor
    raw_preds: torch.Tensor
    soh_raw: torch.Tensor
    vehicle_ids: torch.Tensor

    @classmethod
    def empty(cls, dim: int, g_dim: int, device: torch.device):
        return cls(
            keys=torch.empty(0, dim, device=device),
            latent_values=torch.empty(0, dim, device=device),
            residual_values=torch.empty(0, device=device),
            gradients=torch.empty(0, g_dim, device=device),
            raw_preds=torch.empty(0, device=device),
            soh_raw=torch.empty(0, device=device),
            vehicle_ids=torch.empty(0, dtype=torch.long, device=device),
        )

    def __len__(self) -> int:
        return int(self.keys.size(0))


class MemoryV2:

    def __init__(
        self, dim: int, gradient_dim: int = 0, device: Optional[torch.device] = None
    ):
        self.dim = dim
        self.gradient_dim = gradient_dim
        self.device = torch.device("cpu") if device is None else torch.device(device)
        self.anchor = AnchorPool.empty(dim, self.gradient_dim, self.device)
        self.current_vehicle: Optional[int] = None
        self.bias_state: Dict[int, dict] = {}
        self.use_experience = bool(True)
        self.experience = AnchorPool.empty(dim, self.gradient_dim, self.device)
        self.diag = {
            "retrievals": 0,
            "neighbors_total": 0,
            "neighbors_from_experience": 0,
            "neighbors_cross_vehicle": 0,
            "retrievals_with_cross_vehicle": 0,
            "exp_retrievals_nonempty": 0,
        }

    @torch.no_grad()
    def seed_anchor(
        self,
        keys,
        residual_values,
        raw_preds,
        soh_raw,
        vehicle_ids,
        gradients: Optional[torch.Tensor] = None,
    ):
        keys = keys.detach().to(self.device, dtype=torch.float32)
        residual_values = (
            residual_values.detach().to(self.device, dtype=torch.float32).view(-1)
        )
        raw_preds = raw_preds.detach().to(self.device, dtype=torch.float32).view(-1)
        soh_raw = soh_raw.detach().to(self.device, dtype=torch.float32).view(-1)
        vehicle_ids = vehicle_ids.detach().to(self.device, dtype=torch.long).view(-1)
        if self.gradient_dim > 0:
            assert gradients is not None and gradients.shape[1] == self.gradient_dim
            gradients = gradients.detach().to(self.device, dtype=torch.float32)
        else:
            gradients = torch.empty(keys.size(0), 0, device=self.device)
        if keys.size(0) > 512:
            idx = self._soh_balanced_reservoir(soh_raw, 512, 8)
        else:
            idx = torch.arange(keys.size(0), device=self.device)
        self.anchor.keys = keys[idx]
        self.anchor.latent_values = keys[idx]
        self.anchor.residual_values = residual_values[idx]
        self.anchor.gradients = gradients[idx]
        self.anchor.raw_preds = raw_preds[idx]
        self.anchor.soh_raw = soh_raw[idx]
        self.anchor.vehicle_ids = vehicle_ids[idx]

    def _soh_balanced_reservoir(self, soh_raw, capacity, n_bins):
        if soh_raw.numel() <= capacity:
            return torch.arange(soh_raw.numel(), device=soh_raw.device)
        lo, hi = (soh_raw.min().item(), soh_raw.max().item())
        if hi - lo < 1e-09:
            return (
                torch.randperm(soh_raw.numel(), device=soh_raw.device)[:capacity]
                .sort()
                .values
            )
        edges = torch.linspace(lo, hi + 1e-06, n_bins + 1, device=soh_raw.device)
        bin_ids = torch.bucketize(soh_raw, edges) - 1
        bin_ids = bin_ids.clamp(0, n_bins - 1)
        per_bin = capacity // n_bins
        leftover = capacity - per_bin * n_bins
        keep = []
        for b in range(n_bins):
            mask = (bin_ids == b).nonzero(as_tuple=False).view(-1)
            if mask.numel() == 0:
                continue
            take = min(per_bin + (1 if b < leftover else 0), mask.numel())
            perm = torch.randperm(mask.numel(), device=soh_raw.device)[:take]
            keep.append(mask[perm])
        if not keep:
            return torch.arange(soh_raw.numel(), device=soh_raw.device)[:capacity]
        return torch.cat(keep, dim=0).sort().values

    def begin_vehicle(self, vehicle_id: int, stream_len: int = None):
        self.current_vehicle = int(vehicle_id)
        self.bias_state.setdefault(
            int(vehicle_id), {"bias_ema": 0.0, "n_obs": 0, "residuals": []}
        )

    @torch.no_grad()
    def update_bias(self, residual_raw_normalized: float, vehicle_id: int):
        v = int(vehicle_id)
        st = self.bias_state.setdefault(
            v, {"bias_ema": 0.0, "n_obs": 0, "residuals": []}
        )
        st["n_obs"] += 1
        r = float(residual_raw_normalized)
        st["residuals"].append(r)
        if len(st["residuals"]) > 64:
            st["residuals"].pop(0)
        if st["n_obs"] <= 3:
            st["bias_ema"] = sum(st["residuals"]) / len(st["residuals"])
        else:
            st["bias_ema"] = (1 - 0.25) * st["bias_ema"] + 0.25 * r

    def get_bias(self, vehicle_id) -> float:
        if vehicle_id is None:
            return 0.0
        st = self.bias_state.get(int(vehicle_id))
        return float(st["bias_ema"]) if st else 0.0

    @torch.no_grad()
    def add_experience(
        self,
        key,
        residual_value,
        raw_pred,
        soh_raw,
        vehicle_id,
        gradient: Optional[torch.Tensor] = None,
    ):
        key = key.detach().to(self.device, dtype=torch.float32).view(1, self.dim)
        res = torch.as_tensor(
            residual_value, device=self.device, dtype=torch.float32
        ).view(1)
        rp = torch.as_tensor(raw_pred, device=self.device, dtype=torch.float32).view(1)
        soh = torch.as_tensor(soh_raw, device=self.device, dtype=torch.float32).view(1)
        vid = torch.as_tensor(
            int(vehicle_id), device=self.device, dtype=torch.long
        ).view(1)
        if self.gradient_dim > 0:
            if gradient is None:
                grad = torch.zeros(1, self.gradient_dim, device=self.device)
            else:
                grad = (
                    gradient.detach()
                    .to(self.device, dtype=torch.float32)
                    .view(1, self.gradient_dim)
                )
        else:
            grad = torch.empty(1, 0, device=self.device)
        e = self.experience
        e.keys = torch.cat([e.keys, key], dim=0)
        e.latent_values = torch.cat([e.latent_values, key], dim=0)
        e.residual_values = torch.cat([e.residual_values, res], dim=0)
        e.gradients = torch.cat([e.gradients, grad], dim=0)
        e.raw_preds = torch.cat([e.raw_preds, rp], dim=0)
        e.soh_raw = torch.cat([e.soh_raw, soh], dim=0)
        e.vehicle_ids = torch.cat([e.vehicle_ids, vid], dim=0)
        self._evict_experience()

    def _evict_experience(self):
        e = self.experience
        keep_parts = []
        for v in e.vehicle_ids.unique().tolist():
            pos = (e.vehicle_ids == v).nonzero(as_tuple=False).view(-1)
            if pos.numel() > 64:
                pos = self._select_keep(pos, e.soh_raw[pos], 64)
            keep_parts.append(pos)
        keep = torch.cat(keep_parts).sort().values
        if keep.numel() != len(e):
            self._index_experience(keep)
        if len(e) > 1024:
            pos = torch.arange(len(e), device=self.device)
            keep = self._select_keep(pos, e.soh_raw, 1024).sort().values
            self._index_experience(keep)

    def _select_keep(self, pos, soh, budget):
        if pos.numel() <= budget:
            return pos
        return pos[-budget:]

    def _index_experience(self, idx):
        e = self.experience
        e.keys = e.keys[idx]
        e.latent_values = e.latent_values[idx]
        e.residual_values = e.residual_values[idx]
        e.gradients = e.gradients[idx]
        e.raw_preds = e.raw_preds[idx]
        e.soh_raw = e.soh_raw[idx]
        e.vehicle_ids = e.vehicle_ids[idx]

    def _knn_block(
        self, flat, pool, k, device, dtype, exclude_vehicle=None, self_vehicle=None
    ):
        B = flat.size(0)
        keys = pool.keys.to(device=device, dtype=dtype)
        cand = torch.arange(keys.size(0), device=device)
        if exclude_vehicle is not None and keys.size(0) > 0:
            cand = cand[pool.vehicle_ids.to(device) != int(exclude_vehicle)]
        out = {
            "residual_delta": flat.new_zeros(B),
            "latent_mean": flat.clone(),
            "grad_mean": (
                flat.new_zeros(B, self.gradient_dim) if self.gradient_dim > 0 else None
            ),
            "q05": flat.new_zeros(B),
            "q10": flat.new_zeros(B),
            "q90": flat.new_zeros(B),
            "q95": flat.new_zeros(B),
            "weights": flat.new_zeros(B, 0),
            "idx": torch.zeros(B, 0, dtype=torch.long, device=device),
            "vids": torch.zeros(B, 0, dtype=torch.long, device=device),
            "rp": flat.new_zeros(B, 0),
            "valid": torch.zeros(B, dtype=torch.bool, device=device),
            "k_eff": 0,
        }
        if cand.numel() == 0:
            return out
        k_eff = min(k, cand.numel())
        q_n = F.normalize(flat, dim=-1)
        k_n = F.normalize(keys[cand], dim=-1)
        scores = torch.matmul(q_n, k_n.t()) / 0.5
        sw = getattr(self, "self_weight", 1.0)
        if self_vehicle is not None and sw < 1.0:
            cand_vids = pool.vehicle_ids.to(device)[cand]
            self_mask = (cand_vids == int(self_vehicle)).unsqueeze(0)
            penalty = float(math.log(max(sw, 1e-06)))
            scores = scores + self_mask.to(scores.dtype) * penalty
        top_scores, top_local = torch.topk(scores, k=k_eff, dim=-1)
        top_idx = cand[top_local]
        weights = torch.softmax(top_scores, dim=-1)
        latent_neigh = pool.latent_values.to(device=device, dtype=dtype)[top_idx]
        residual_neigh = pool.residual_values.to(device=device, dtype=dtype)[top_idx]
        out["latent_mean"] = (weights.unsqueeze(-1) * latent_neigh).sum(dim=1)
        out["residual_delta"] = (weights * residual_neigh).sum(dim=1) * 1.0
        if self.gradient_dim > 0:
            grad_neigh = pool.gradients.to(device=device, dtype=dtype)[top_idx]
            out["grad_mean"] = (weights.unsqueeze(-1) * grad_neigh).sum(dim=1)
        sorted_v, order = residual_neigh.sort(dim=-1)
        sorted_w = torch.gather(weights, -1, order)
        cdf = sorted_w.cumsum(-1) / sorted_w.sum(-1, keepdim=True).clamp_min(1e-09)
        for name, q in zip(("q05", "q10", "q90", "q95"), (0.05, 0.1, 0.9, 0.95)):
            idx_q = (cdf >= q).float().argmax(-1)
            out[name] = torch.gather(sorted_v, -1, idx_q.unsqueeze(-1)).squeeze(-1)
        out.update(
            weights=weights,
            idx=top_idx,
            vids=pool.vehicle_ids.to(device)[top_idx],
            rp=pool.raw_preds.to(device=device, dtype=dtype)[top_idx],
            valid=torch.ones(B, dtype=torch.bool, device=device),
            k_eff=k_eff,
        )
        return out

    def retrieve(self, query: torch.Tensor) -> dict:
        return self._retrieve_split(query)

    def _retrieve_split(self, query: torch.Tensor) -> dict:
        orig_shape = query.shape[:-1]
        flat = query.reshape(-1, self.dim)
        device, dtype = (query.device, query.dtype)
        B = flat.size(0)
        bias = self.get_bias(self.current_vehicle)
        a = self._knn_block(flat, self.anchor, 4, device, dtype)
        excl = None
        e = self._knn_block(
            flat,
            self.experience,
            4,
            device,
            dtype,
            exclude_vehicle=excl,
            self_vehicle=self.current_vehicle,
        )
        residual_delta = 1.0 * a["residual_delta"] + 1.0 * e["residual_delta"] + bias
        latent_delta = (a["latent_mean"] - flat) * 0.0
        grad_mean = a["grad_mean"]
        if grad_mean is not None and grad_mean.size(0) == 1:
            grad_mean = grad_mean.squeeze(0)
        q05, q10, q90, q95 = (a["q05"], a["q10"], a["q90"], a["q95"])
        width95 = q95 - q05
        confidence = 1.0 / (1.0 + width95)
        self.diag["retrievals"] += 1
        n_a = a["k_eff"] * B
        n_e = e["k_eff"] * B
        self.diag["neighbors_total"] += n_a + n_e
        self.diag["neighbors_from_experience"] += n_e
        if e["k_eff"] > 0:
            self.diag["exp_retrievals_nonempty"] += 1
            if self.current_vehicle is not None:
                cross = (e["vids"] != int(self.current_vehicle)).sum().item()
                self.diag["neighbors_cross_vehicle"] += int(cross)
                if cross > 0:
                    self.diag["retrievals_with_cross_vehicle"] += 1
        ka, ke = (a["k_eff"], e["k_eff"])
        nb_v = (
            torch.cat([a["vids"], e["vids"]], dim=1)
            if ka or ke
            else torch.zeros(B, 0, dtype=torch.long, device=device)
        )
        nb_w = (
            torch.cat([a["weights"], e["weights"]], dim=1)
            if ka or ke
            else flat.new_zeros(B, 0)
        )
        nb_i = (
            torch.cat([a["idx"], e["idx"]], dim=1)
            if ka or ke
            else torch.zeros(B, 0, dtype=torch.long, device=device)
        )
        nb_rp = (
            torch.cat([a["rp"], e["rp"]], dim=1) if ka or ke else flat.new_zeros(B, 0)
        )
        nb_src = torch.cat(
            [
                torch.zeros(B, ka, dtype=torch.long, device=device),
                torch.ones(B, ke, dtype=torch.long, device=device),
            ],
            dim=1,
        )
        ktot = ka + ke
        anchor_contrib = 1.0 * a["residual_delta"]
        exp_contrib = 1.0 * e["residual_delta"]
        denom = (anchor_contrib.abs() + exp_contrib.abs()).clamp_min(1e-09)
        exp_weight_share = exp_contrib.abs() / denom
        return {
            "latent_delta": latent_delta.reshape(*orig_shape, self.dim),
            "residual_delta": residual_delta.reshape(*orig_shape),
            "residual_q05": q05.reshape(*orig_shape),
            "residual_q10": q10.reshape(*orig_shape),
            "residual_q90": q90.reshape(*orig_shape),
            "residual_q95": q95.reshape(*orig_shape),
            "residual_width95": width95.reshape(*orig_shape),
            "memory_confidence": confidence.reshape(*orig_shape),
            "interval_available": a["valid"].reshape(*orig_shape),
            "weights": nb_w.reshape(*orig_shape, ktot),
            "indices": nb_i.reshape(*orig_shape, ktot),
            "memory_size": len(self.anchor) + len(self.experience),
            "_anchor_size": len(self.anchor),
            "_warm_size": 0,
            "_exchange_size": len(self.experience),
            "_bias": bias,
            "_anchor_contrib": anchor_contrib.reshape(*orig_shape),
            "_exp_contrib": exp_contrib.reshape(*orig_shape),
            "_exp_weight_share": exp_weight_share.reshape(*orig_shape),
            "_neighbor_vehicles": nb_v.reshape(*orig_shape, ktot),
            "_neighbor_raw_preds": nb_rp.reshape(*orig_shape, ktot),
            "_neighbor_sources": nb_src.reshape(*orig_shape, ktot),
            "_grad_mean": grad_mean,
        }

    def correct_latent(self, latent):
        info = self.retrieve(latent)
        return (latent + info["latent_delta"], info)

    def correct_output(self, output, info):
        return output + info["residual_delta"]

    @property
    def size(self) -> int:
        return len(self.anchor)

    def state_dict(self) -> dict:

        def _cpu(v):
            return v.cpu() if torch.is_tensor(v) else v

        return {
            "anchor": {
                k: _cpu(getattr(self.anchor, k))
                for k in self.anchor.__dataclass_fields__
            },
            "current_vehicle": self.current_vehicle,
            "bias_state": self.bias_state,
            "diag": dict(self.diag),
            "config": {
                "dim": self.dim,
                "gradient_dim": self.gradient_dim,
                "anchor_capacity": 512,
                "top_k": 4,
                "temperature": 0.5,
                "latent_scale": 0.0,
                "residual_scale": 1.0,
                "soh_bins": 8,
                "bias_alpha": 0.25,
                "value_mode": "both",
                "use_experience": True,
                "experience_capacity": 1024,
                "retrieve_mode": "split",
                "exp_top_k": 4,
                "exp_per_vehicle_quota": 64,
                "exp_exclude_self": False,
                "exp_residual_weight": 1.0,
                "anchor_residual_weight": 1.0,
                "exp_evict_policy": "recency",
                "exp_quota_frac": None,
                "self_weight": 1.0,
            },
        }
