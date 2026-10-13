from src.dataloader import list_vehicle_ids
import math
import random
import sys
from collections import deque
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import torch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from exp.finetune_target_dora import config_from_checkpoint
from models.dora import (
    dora_parameter_report,
    freeze_non_dora_parameters,
    inject_dora_linear_layers,
)
from models.memory_v2 import MemoryV2
from models.picolnet import PicolNet
from src.dataloader import BatteryWindowDataset
from trainer_offline import (
    compute_losses,
    denormalize_labels,
    load_scaler_from_checkpoint,
    normalize_features,
    normalize_labels,
    raw_segment_available_rate,
    regression_metrics,
    resolve_device,
    resolve_online_vehicle_scalers,
    set_seed,
    write_csv,
    write_json,
)

LABEL_DELAY = 1


def resolve_test_vehicles():
    return [v for v in list_vehicle_ids(1) if v not in [1, 15, 19]]


def resolve_data_args():
    return {"window": 50, "horizon": 10, "stride": 1, "raw_length": 128}


def resolve_memory_update_count(num_samples, ratio):
    if not 0.0 <= ratio <= 1.0:
        raise ValueError("memory_update_ratio must be in [0, 1].")
    if num_samples <= 0:
        raise ValueError("Online stream must contain at least one sample.")
    return min(num_samples, int(math.ceil(num_samples * ratio)))


def resolve_memory_update_seed(base_seed, vehicle_id):
    return int(base_seed) + int(vehicle_id) * 1000003


def resolve_memory_update_indices(num_samples, ratio, seed=None):
    update_count = resolve_memory_update_count(num_samples, ratio)
    if update_count == 0:
        return []
    indices = random.Random(seed).sample(range(num_samples), k=update_count)
    indices.sort()
    return indices


def inject_dora_if_needed(model, state_dict):
    return inject_dora_linear_layers(model)


def resolve_loss_weights():
    return {"pred": 1.0, "recon": 0.025, "phy": 0.01}


def build_loss_args(loss_weights):
    return SimpleNamespace(
        lambda_pred=loss_weights["pred"],
        lambda_recon=loss_weights["recon"],
        lambda_phy=loss_weights["phy"],
        lambda_mmd=0.0,
        grad_clip=1.0,
    )


def prepare_item_tensors(item, x_scaler, y_scaler, device):
    vehicle_ids = item["vehicle_id"].view(1)
    features = normalize_features(
        item["features"].unsqueeze(0), vehicle_ids, x_scaler, device
    )
    pinn = item["pinn"].unsqueeze(0).to(device=device, dtype=torch.float32)
    label = normalize_labels(item["label"].view(1), vehicle_ids, y_scaler, device)
    return (vehicle_ids, features, pinn, label)


def forward_model(model, features, pinn, memory):
    return model(features, pinn=pinn, memory=memory)


def build_online_model(config, checkpoint, settings, device):
    model = PicolNet(config).to(device)
    dora_replaced = inject_dora_if_needed(model, checkpoint["model_state_dict"])
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    optimizer = None
    online_dora_report = None
    if not dora_replaced:
        raise ValueError(
            "Online adaptation requires a DoRA checkpoint with injected DoRA layers."
        )
    freeze_non_dora_parameters(model)
    online_dora_report = dora_parameter_report(model, replaced_modules=dora_replaced)
    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not trainable:
        raise RuntimeError("No trainable DoRA parameters found for online adaptation.")
    optimizer = torch.optim.AdamW(trainable, lr=0.0005, weight_decay=0.0)
    model.eval()
    return (model, optimizer, dora_replaced, online_dora_report)


def _head_param_list(model):
    head = model.predictor.head
    return [p for p in head[-1].parameters() if p.requires_grad]


def _flatten_grads(params):
    return torch.cat(
        [p.grad.detach().reshape(-1) for p in params if p.grad is not None]
    )


def _num_head_params(model) -> int:
    return sum((p.numel() for p in _head_param_list(model)))


def _collect_dora_state(model):
    return {
        n: p.detach().clone()
        for (n, p) in model.named_parameters()
        if p.requires_grad
        and (
            n.endswith("lora_a") or n.endswith("lora_b") or n.endswith("dora_magnitude")
        )
    }


def _apply_dora_state(model, state):
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in state:
                p.data.copy_(state[n].to(p.device, p.dtype))


def _ema_dora(ema_state, current_state, alpha: float):
    return {
        n: (1.0 - alpha) * ema_state[n]
        + alpha * current_state[n].to(ema_state[n].device, ema_state[n].dtype)
        for n in ema_state
    }


def _seed_anchor_with_grad(
    model, known_vehicles, dataset_id, data_args, x_scaler, y_scaler, device, args
):
    head_params = _head_param_list(model)
    keys, residuals, raw_preds, sohs, vids, grads = ([], [], [], [], [], [])
    for p in head_params:
        p.requires_grad_(True)
    model.eval()
    for vid in known_vehicles:
        ds = BatteryWindowDataset(
            dataset_id=dataset_id,
            vehicle_id=int(vid),
            window=data_args["window"],
            horizon=data_args["horizon"],
            stride=data_args["stride"],
            raw_length=data_args["raw_length"],
        )
        for i in range(0, len(ds), max(1, int(1))):
            item = ds[i]
            with torch.no_grad():
                vehicle_ids, features, pinn, label = prepare_item_tensors(
                    item, x_scaler, y_scaler, device
                )
            out = forward_model(model, features, pinn, memory=None)
            pred = out["pred"].view(())
            loss = (pred - label.view(())) ** 2
            for p in head_params:
                if p.grad is not None:
                    p.grad = None
            loss.backward()
            g_flat = _flatten_grads(head_params).detach()
            grads.append(g_flat.cpu())
            with torch.no_grad():
                pred_det = pred.detach()
                keys.append(out["predictor_latent"].detach().view(-1))
                residuals.append((label.view(()) - pred_det).detach())
                raw_preds.append(pred_det)
                sohs.append(label.detach().view(()))
                vids.append(int(vid))
    for p in head_params:
        if p.grad is not None:
            p.grad = None
    return (keys, residuals, raw_preds, sohs, vids, grads)


def _seed_anchor_pool(
    model,
    memory: MemoryV2,
    known_vehicles,
    dataset_id,
    data_args,
    x_scaler,
    y_scaler,
    device,
    args,
):
    print(
        f"[v2:seed_anchor] known={known_vehicles} dataset={dataset_id} value_mode={'both'}"
    )
    keys, residuals, raw_preds, sohs, vids, grads = _seed_anchor_with_grad(
        model, known_vehicles, dataset_id, data_args, x_scaler, y_scaler, device, args
    )
    if not keys:
        raise RuntimeError("anchor seed produced 0 entries.")
    keys_t = torch.stack(keys, dim=0)
    res_t = torch.stack(residuals, dim=0)
    rp_t = torch.stack(raw_preds, dim=0)
    soh_t = torch.stack(sohs, dim=0)
    vids_t = torch.tensor(vids, dtype=torch.long, device=device)
    grads_t = torch.stack(grads, dim=0).to(device) if grads else None
    memory.seed_anchor(
        keys=keys_t,
        residual_values=res_t,
        raw_preds=rp_t,
        soh_raw=soh_t,
        vehicle_ids=vids_t,
        gradients=grads_t,
    )
    print(
        f"[v2:seed_anchor] anchor_size={len(memory.anchor)} grad_dim={memory.gradient_dim}"
    )


def _stream_vehicle_v2(
    model,
    dataset,
    vehicle_id,
    memory: MemoryV2,
    x_scaler,
    y_scaler,
    device,
    args,
    dora_optimizer,
    loss_args,
    vehicle_pos: int,
):
    rows = []
    full_preds, full_labels = ([], [])
    trace = None
    max_samples = len(dataset)
    memory.begin_vehicle(int(vehicle_id), stream_len=max_samples)
    seed = resolve_memory_update_seed(2026, int(vehicle_id))
    upd_idx = set(
        resolve_memory_update_indices(num_samples=max_samples, ratio=0.15, seed=seed)
    )
    burn_in = int(20)
    n_supervised = 0
    pending_updates = deque()
    applied_updates = 0

    def _apply_update(payload):
        nonlocal n_supervised
        feats = payload["features"]
        pn = payload["pinn"]
        lbl = payload["label"]
        pred_model = payload["pred_raw_model"]
        sup_loss_local = None
        if dora_optimizer is not None and (not payload["in_burn"]):
            model.train()
            for _ in range(3):
                out_t = forward_model(model, feats, pn, memory)
                losses = compute_losses(out_t, feats, lbl, loss_args)
                dora_optimizer.zero_grad(set_to_none=True)
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                dora_optimizer.step()
                sup_loss_local = float(losses["total"].item())
            model.eval()
            n_supervised += 1
        residual_bias = float((lbl - pred_model).item())
        memory.update_bias(residual_bias, int(vehicle_id))
        memory.add_experience(
            key=payload["exp_key"],
            residual_value=float((lbl.view(()) - pred_model.view(())).item()),
            raw_pred=float(pred_model.view(()).item()),
            soh_raw=float(lbl.view(()).item()),
            vehicle_id=int(vehicle_id),
        )
        return sup_loss_local

    model.eval()
    for stream_idx in range(max_samples):
        item = dataset[stream_idx]
        with torch.no_grad():
            vehicle_ids, features, pinn, label = prepare_item_tensors(
                item, x_scaler, y_scaler, device
            )
            out = forward_model(model, features, pinn, memory)
            pred_raw_model = out["pred"].detach()
            pred = pred_raw_model
            exp_key = out["predictor_latent"].detach().view(-1)
            mem_info = out.get("memory", {})
            pred_raw = denormalize_labels(pred, vehicle_ids, y_scaler)
            label_raw = denormalize_labels(label, vehicle_ids, y_scaler)
        allow_update = stream_idx in upd_idx
        in_burn = stream_idx < burn_in
        full_preds.append(pred_raw)
        full_labels.append(label_raw)
        sup_loss = None
        if allow_update:
            pending_updates.append(
                {
                    "source_idx": stream_idx,
                    "features": features,
                    "pinn": pinn,
                    "label": label,
                    "exp_key": exp_key,
                    "pred_raw_model": pred_raw_model,
                    "in_burn": in_burn,
                }
            )
        update_source_idx = None
        supervised_updated = False
        while (
            pending_updates
            and pending_updates[0]["source_idx"] + LABEL_DELAY <= stream_idx
        ):
            payload = pending_updates.popleft()
            sup_loss = _apply_update(payload)
            update_source_idx = payload["source_idx"]
            supervised_updated = sup_loss is not None
            applied_updates += 1
        rows.append(
            {
                "stage": "memory_v2",
                "task": "3to1",
                "vehicle_id": int(vehicle_id),
                "vehicle_pos": int(vehicle_pos),
                "stream_idx": int(stream_idx),
                "sample_idx": int(item["sample_idx"].item()),
                "cycle_number": int(item["cycle_number"].item()),
                "anchor_size": int(len(memory.anchor)),
                "experience_size": int(len(memory.experience)),
                "neighbors_from_experience": int(
                    mem_info.get("_neighbor_sources").sum().item()
                    if mem_info.get("_neighbor_sources") is not None
                    else 0
                ),
                "in_burn_in": int(in_burn),
                "memory_update_enabled": int(allow_update),
                "update_source_idx": update_source_idx,
                "supervised_updated": int(supervised_updated),
                "y_pred": float(pred_raw.item()),
                "y_true_full": float(label_raw.item()),
                "y_true": float(label_raw.item()) if allow_update else None,
                "abs_error": float((label_raw - pred_raw).abs().item()),
                "online_supervised_train_total": sup_loss,
            }
        )
    metrics = regression_metrics(torch.cat(full_preds), torch.cat(full_labels))
    metrics["vehicle_id"] = int(vehicle_id)
    metrics["vehicle_pos"] = int(vehicle_pos)
    metrics["num_samples"] = int(max_samples)
    metrics["memory_update_samples"] = int(len(upd_idx))
    metrics["memory_updates_applied"] = applied_updates
    metrics["supervised_updates"] = int(n_supervised)
    metrics["anchor_size"] = int(len(memory.anchor))
    metrics["experience_size"] = int(len(memory.experience))
    return (rows, metrics, trace)


def _split_metrics_from_rows(rows):
    if not rows:
        empty = {"mae": None, "rmse": None, "mape": None, "r2": None, "n": 0}
        return {"all": empty, "prefix_labeled": empty, "suffix_unlabeled": empty}

    def _bucket(filtered_rows):
        if not filtered_rows:
            return {"mae": None, "rmse": None, "mape": None, "r2": None, "n": 0}
        pred = torch.tensor(
            [row["y_pred"] for row in filtered_rows], dtype=torch.float32
        )
        true = torch.tensor(
            [row["y_true_full"] for row in filtered_rows], dtype=torch.float32
        )
        bucket = regression_metrics(pred, true)
        bucket["n"] = int(len(filtered_rows))
        return bucket

    prefix_rows = [row for row in rows if int(row.get("memory_update_enabled", 0)) == 1]
    suffix_rows = [row for row in rows if int(row.get("memory_update_enabled", 0)) == 0]
    return {
        "all": _bucket(rows),
        "prefix_labeled": _bucket(prefix_rows),
        "suffix_unlabeled": _bucket(suffix_rows),
    }


def _fmt(value):
    if value is None:
        return "nan"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def main_v2(output_dir, checkpoint_path):
    args = SimpleNamespace()
    set_seed(2026)
    device = resolve_device("cuda")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    settings = {
        "split": {"known_vehicles": [1, 15, 19]},
        "dora": {"rank": 64, "alpha": 16.0, "eps": 5e-06},
    }
    config = config_from_checkpoint(checkpoint)
    data_args = resolve_data_args()
    test_vehicles = resolve_test_vehicles()
    known_vehicles = settings.get("split", {}).get("known_vehicles", [])
    if not known_vehicles:
        raise RuntimeError("settings missing known_vehicles")
    loss_weights = resolve_loss_weights()
    loss_args = build_loss_args(loss_weights)
    task_name = "3to1"
    source_x_scaler = load_scaler_from_checkpoint(checkpoint, "x")
    source_y_scaler = load_scaler_from_checkpoint(checkpoint, "y")
    model, optimizer, dora_replaced, _ = build_online_model(
        config=config, checkpoint=checkpoint, settings=settings, device=device
    )
    predictor_params = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if not (
            n.endswith("lora_a") or n.endswith("lora_b") or n.endswith("dora_magnitude")
        ):
            continue
        if not n.startswith("ssp."):
            predictor_params.append(p)
    print(f"[v2:predictor_opt] predictor_dora_params={len(predictor_params)}")
    if predictor_params:
        optimizer = torch.optim.AdamW(predictor_params, lr=0.0005, weight_decay=0.0)
    ckpt_dora_state = _collect_dora_state(model)
    dora_ema_state = {n: t.clone() for (n, t) in ckpt_dora_state.items()}
    ema_config = {"dora_alpha": 0.1, "bias_alpha": 0.25, "init_from_ema": True}
    g_dim = _num_head_params(model)
    memory = MemoryV2(
        dim=config.predictor.hidden_dim, gradient_dim=g_dim, device=device
    )
    _seed_anchor_pool(
        model=model,
        memory=memory,
        known_vehicles=known_vehicles,
        dataset_id=1,
        data_args=data_args,
        x_scaler=source_x_scaler,
        y_scaler=source_y_scaler,
        device=device,
        args=args,
    )
    print(
        f"[v2:start] task={task_name} test={test_vehicles} seed=2026 visible_label_ratio=0.15 label_delay={LABEL_DELAY} value_mode={'both'} dora_ema_alpha=0.1 bias_alpha=0.25 init_from_ema=True"
    )
    prediction_rows = []
    vehicle_rows = []
    raw_rates = {}
    cumulative_r2 = []
    for vehicle_pos, vid in enumerate(test_vehicles):
        _apply_dora_state(model, dora_ema_state)
        ds = BatteryWindowDataset(
            dataset_id=1,
            vehicle_id=int(vid),
            window=data_args["window"],
            horizon=data_args["horizon"],
            stride=data_args["stride"],
            raw_length=data_args["raw_length"],
        )
        x_scaler, y_scaler = resolve_online_vehicle_scalers(
            dataset=ds, source_x_scaler=source_x_scaler, source_y_scaler=source_y_scaler
        )
        raw_rates[str(vid)] = raw_segment_available_rate(ds)
        print(f"[raw] vehicle={vid} available_rate={raw_rates[str(vid)]}")
        rows, metrics, trace = _stream_vehicle_v2(
            model=model,
            dataset=ds,
            vehicle_id=int(vid),
            memory=memory,
            x_scaler=x_scaler,
            y_scaler=y_scaler,
            device=device,
            args=args,
            dora_optimizer=optimizer,
            loss_args=loss_args,
            vehicle_pos=vehicle_pos,
        )
        split_metrics = _split_metrics_from_rows(rows)
        metrics["split_metrics"] = split_metrics
        metrics["prefix_k"] = float(0.15)
        prediction_rows.extend(rows)
        vehicle_rows.append(metrics)
        cur_state = _collect_dora_state(model)
        dora_ema_state = _ema_dora(dora_ema_state, cur_state, ema_config["dora_alpha"])
        r2_so_far = [v["r2"] for v in vehicle_rows]
        cumulative_r2.append(float(np.mean(r2_so_far)))
        all_m = split_metrics["all"]
        pre_m = split_metrics["prefix_labeled"]
        suf_m = split_metrics["suffix_unlabeled"]
        print(
            f"[v2:done] v{vid} pos={vehicle_pos} sup={metrics['supervised_updates']} anchor={metrics['anchor_size']} cum_mean_r2={cumulative_r2[-1]:+.4f}"
        )
        print(
            f"          all     n={all_m['n']:>5d}  mae={_fmt(all_m['mae'])}  rmse={_fmt(all_m['rmse'])}  mape={_fmt(all_m['mape'])}  r2={_fmt(all_m['r2'])}"
        )
        print(
            f"          prefix  n={pre_m['n']:>5d}  mae={_fmt(pre_m['mae'])}  rmse={_fmt(pre_m['rmse'])}  mape={_fmt(pre_m['mape'])}  r2={_fmt(pre_m['r2'])}"
        )
        print(
            f"          suffix  n={suf_m['n']:>5d}  mae={_fmt(suf_m['mae'])}  rmse={_fmt(suf_m['rmse'])}  mape={_fmt(suf_m['mape'])}  r2={_fmt(suf_m['r2'])}"
        )
    full_pred = torch.tensor(
        [r["y_pred"] for r in prediction_rows], dtype=torch.float32
    )
    full_label = torch.tensor(
        [r["y_true_full"] for r in prediction_rows], dtype=torch.float32
    )
    overall = regression_metrics(full_pred, full_label)
    r2s = [v["r2"] for v in vehicle_rows]
    mean_r2 = float(np.mean(r2s))
    print(
        f"\n[v2:overall] r2={overall['r2']:.4f} mae={overall['mae']:.4f} mean_per_vehicle_r2={mean_r2:.4f}"
    )
    diag = dict(memory.diag)
    diag["use_experience"] = bool(memory.use_experience)
    diag["final_experience_size"] = int(len(memory.experience))
    frac_exp = (
        diag["neighbors_from_experience"] / diag["neighbors_total"]
        if diag["neighbors_total"]
        else 0.0
    )
    frac_cross = (
        diag["neighbors_cross_vehicle"] / diag["neighbors_total"]
        if diag["neighbors_total"]
        else 0.0
    )
    diag["frac_neighbors_from_experience"] = frac_exp
    diag["frac_neighbors_cross_vehicle"] = frac_cross
    print(
        f"[v2:flow] use_experience={diag['use_experience']} exp_size={diag['final_experience_size']} neigh_from_exp={frac_exp:.3f} neigh_cross_vehicle={frac_cross:.3f} retrievals_with_cross={diag['retrievals_with_cross_vehicle']}/{diag['retrievals']}"
    )
    final_metrics = {
        "label_delay": LABEL_DELAY,
        "visible_label_ratio": 0.15,
        "overall": overall,
        "mean_per_vehicle_r2": mean_r2,
        "min_per_vehicle_r2": min(r2s),
        "vehicles": vehicle_rows,
        "sample_count": len(prediction_rows),
        "cumulative_mean_r2": cumulative_r2,
        "experience_flow": diag,
    }
    write_json(output_dir / "final_metrics.json", final_metrics)
    write_csv(output_dir / "online_predictions.csv", prediction_rows)
    write_csv(output_dir / "vehicle_metrics.csv", vehicle_rows)
    exp_snapshot = {
        k: getattr(memory.experience, k).cpu()
        for k in memory.experience.__dataclass_fields__
    }
    torch.save(
        {
            "memory": memory.state_dict(),
            "dora_ema": dora_ema_state,
            "experience": exp_snapshot,
            "experience_flow": diag,
        },
        output_dir / "state.pt",
    )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": asdict(config),
            "x_scaler": source_x_scaler,
            "y_scaler": source_y_scaler,
            "memory": memory.state_dict(),
            "experience": exp_snapshot,
            "dora_ema": dora_ema_state,
            "dora": settings["dora"],
        },
        output_dir / "after_online.pt",
    )
