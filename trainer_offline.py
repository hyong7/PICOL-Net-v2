from types import SimpleNamespace
import csv
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from models.ecm import ecm_voltage_loss
from models.picolnet import PicolNet
from src.config import ModelConfig
from src.dataloader import MultiVehicleBatteryDataset


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and (not torch.cuda.is_available()):
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(name)


def _init_running_stats(dim):
    return {
        "count": 0,
        "sum": torch.zeros(dim, dtype=torch.float64),
        "sum_sq": torch.zeros(dim, dtype=torch.float64),
    }


def _update_running_stats(stats, values):
    if values.numel() == 0:
        return
    values = values.detach().to(dtype=torch.float64)
    stats["count"] += int(values.shape[0])
    stats["sum"] += values.sum(dim=0).cpu()
    stats["sum_sq"] += (values * values).sum(dim=0).cpu()


def _finalize_running_stats(stats, eps=1e-06):
    if stats["count"] <= 0:
        raise ValueError("Cannot finalize scaler stats with zero samples.")
    count = float(stats["count"])
    mean = stats["sum"] / count
    var = torch.clamp(stats["sum_sq"] / count - mean * mean, min=0.0)
    std = torch.sqrt(var)
    std = torch.where(std < eps, torch.ones_like(std), std)
    return (mean.to(dtype=torch.float32), std.to(dtype=torch.float32))


def _collect_scaler_stats(dataset, x_global_stats=None, y_global_stats=None):
    for batch in DataLoader(dataset, batch_size=512, shuffle=False):
        features = batch["features"].to(dtype=torch.float32)
        labels = batch["label"].view(-1, 1).to(dtype=torch.float32)
        if x_global_stats is not None:
            _update_running_stats(
                x_global_stats, features.reshape(-1, features.shape[-1])
            )
        if y_global_stats is not None:
            _update_running_stats(y_global_stats, labels)


def _global_scaler(kind, mean, std):
    return {
        "kind": kind,
        "scope": "train",
        "mean": mean.detach().cpu(),
        "std": std.detach().cpu(),
    }


def _clone_scaler_to_cpu(scaler):
    return {
        "kind": scaler["kind"],
        "scope": "train",
        "mean": scaler["mean"].detach().cpu(),
        "std": scaler["std"].detach().cpu(),
    }


def prepare_dataset_scaler(dataset, kind):
    dim = dataset[0]["features"].shape[-1] if kind == "x" else 1
    stats = _init_running_stats(dim)
    _collect_scaler_stats(
        dataset, **{"x_global_stats" if kind == "x" else "y_global_stats": stats}
    )
    return _global_scaler(kind, *_finalize_running_stats(stats))


def prepare_split_scalers(train_dataset, val_dataset, test_dataset):
    x = prepare_dataset_scaler(train_dataset, "x")
    y = prepare_dataset_scaler(train_dataset, "y")
    return {split: {"x": x, "y": y} for split in ("train", "val", "test")}


def _apply_vehicle_scaler(values, vehicle_ids, scaler, device, value_kind):
    values = values.to(device=device, dtype=torch.float32)
    mean = scaler["mean"].to(device=device, dtype=torch.float32)
    std = scaler["std"].to(device=device, dtype=torch.float32)
    if value_kind == "x":
        return (values - mean.view(1, 1, -1)) / std.view(1, 1, -1)
    return (values - mean.view(1)) / std.view(1)


def normalize_features(features, vehicle_ids, scaler, device):
    return _apply_vehicle_scaler(features, vehicle_ids, scaler, device, value_kind="x")


def normalize_labels(labels, vehicle_ids, scaler, device):
    labels = labels.view(-1)
    return _apply_vehicle_scaler(labels, vehicle_ids, scaler, device, value_kind="y")


def denormalize_labels(labels, vehicle_ids, scaler):
    labels = labels.detach().cpu().to(dtype=torch.float32).view(-1)
    mean = scaler["mean"].detach().cpu().to(dtype=torch.float32).view(1)
    std = scaler["std"].detach().cpu().to(dtype=torch.float32).view(1)
    return labels * std + mean


def load_scaler_from_checkpoint(checkpoint, kind):
    return _clone_scaler_to_cpu(checkpoint[f"{kind}_scaler"])


def resolve_transfer_split_scalers(
    train_dataset, val_dataset, test_dataset, source_x_scaler, source_y_scaler
):
    scalers = prepare_split_scalers(train_dataset, val_dataset, test_dataset)
    for split in scalers.values():
        split["x"] = _clone_scaler_to_cpu(source_x_scaler)
        split["y"] = _clone_scaler_to_cpu(source_y_scaler)
    return scalers


def resolve_online_vehicle_scalers(dataset, source_x_scaler, source_y_scaler):
    return (
        _clone_scaler_to_cpu(source_x_scaler),
        _clone_scaler_to_cpu(source_y_scaler),
    )


def raw_segment_available_rate(dataset):
    loader = DataLoader(dataset, batch_size=512, shuffle=False)
    valid_segments = 0
    total_segments = 0
    for batch in loader:
        pinn = batch["pinn"].numpy()
        valid = np.isfinite(pinn).all(axis=(2, 3))
        valid_segments += int(valid.sum())
        total_segments += int(valid.size)
    if total_segments == 0:
        return None
    return valid_segments / total_segments


def build_datasets():
    splits = {
        "train": [346, 481, 341, 406, 431, 369, 392],
        "val": [368],
        "test": [428, 379],
    }
    datasets = [
        MultiVehicleBatteryDataset(
            dataset_id=3,
            vehicle_ids=vehicles,
            window=50,
            horizon=10,
            stride=1,
            raw_length=128,
        )
        for vehicles in splits.values()
    ]
    info = {f"{name}_vehicles": vehicles for (name, vehicles) in splits.items()}
    info.update(
        {f"{name}_windows": len(dataset) for (name, dataset) in zip(splits, datasets)}
    )
    return (*datasets, info)


def build_config(feature_dim):
    return ModelConfig(feature_dim=feature_dim)


def flatten_latent_for_mmd(latent):
    if latent is None:
        return None
    if latent.dim() <= 2:
        return latent
    return latent.reshape(-1, latent.shape[-1])


def maximum_mean_discrepancy(source, target):
    if source is None or target is None:
        raise ValueError("MMD requires non-null source and target latents.")
    source = flatten_latent_for_mmd(source)
    target = flatten_latent_for_mmd(target)
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0)
    total1 = total.unsqueeze(1)
    l2_distance = ((total0 - total1) ** 2).sum(dim=2)
    denom = max(total.shape[0] * total.shape[0] - total.shape[0], 1)
    bandwidth = (l2_distance.detach().sum() / denom).item()
    bandwidth = max(bandwidth, 1e-12)
    bandwidth /= 2.0 ** (5 // 2)
    kernel_sum = 0.0
    for kernel_idx in range(5):
        current_bandwidth = bandwidth * 2.0**kernel_idx
        kernel_sum = kernel_sum + torch.exp(
            -l2_distance / max(current_bandwidth, 1e-12)
        )
    source_size = source.shape[0]
    xx = kernel_sum[:source_size, :source_size].mean()
    yy = kernel_sum[source_size:, source_size:].mean()
    xy = kernel_sum[:source_size, source_size:].mean()
    yx = kernel_sum[source_size:, :source_size].mean()
    return xx + yy - xy - yx


def compute_losses(out, features, labels, args, loss_mmd=None):
    loss_pred = F.mse_loss(out["pred"], labels)
    if out["ssp_recon"] is not None:
        loss_recon = F.mse_loss(out["ssp_recon"], features)
    else:
        loss_recon = features.new_tensor(0.0)
    if "voltage_pred" in out:
        loss_phy = ecm_voltage_loss(out["voltage_pred"], out["voltage_true"])
    else:
        loss_phy = features.new_tensor(0.0)
    if loss_mmd is None:
        loss_mmd = features.new_tensor(0.0)
    total = (
        args.lambda_pred * loss_pred
        + args.lambda_recon * loss_recon
        + args.lambda_phy * loss_phy
        + args.lambda_mmd * loss_mmd
    )
    return {
        "total": total,
        "pred": loss_pred,
        "recon": loss_recon,
        "phy": loss_phy,
        "mmd": loss_mmd,
    }


def weighted_loss_values(losses, args):
    return {
        "pred": float(args.lambda_pred) * float(losses["pred"]),
        "recon": float(args.lambda_recon) * float(losses["recon"]),
        "phy": float(args.lambda_phy) * float(losses["phy"]),
        "mmd": float(args.lambda_mmd) * float(losses["mmd"]),
    }


def add_weighted_loss_columns(row, prefix, losses, args):
    weighted = weighted_loss_values(losses, args)
    for name, value in weighted.items():
        row[f"{prefix}_weighted_{name}"] = value


def format_loss_weights(args):
    return f"weights(pred={args.lambda_pred:g},rec={args.lambda_recon:g},phy={args.lambda_phy:g},mmd={args.lambda_mmd:g})"


def format_weighted_losses(losses, args):
    weighted = weighted_loss_values(losses, args)
    return f"weighted(pred={weighted['pred']:.4f},rec={weighted['recon']:.4f},phy={weighted['phy']:.4f},mmd={weighted['mmd']:.4f})"


def infer_pretrain_task_name(args):
    return "source3"


def infer_pretrain_variant_name(args):
    return "wo_mmd"


def regression_metrics(pred, target, eps=1e-08):
    pred = pred.detach().cpu()
    target = target.detach().cpu()
    err = pred - target
    mae = err.abs().mean().item()
    rmse = torch.sqrt((err**2).mean()).item()
    mape = (err.abs() / target.abs().clamp_min(eps)).mean().item() * 100.0
    ss_res = ((target - pred) ** 2).sum()
    ss_tot = ((target - target.mean()) ** 2).sum().clamp_min(eps)
    r2 = (1.0 - ss_res / ss_tot).item()
    return {"mae": mae, "rmse": rmse, "mape": mape, "r2": r2}


def init_physics_summary():
    return {"r0_sum": 0.0, "rp_sum": 0.0, "cp_sum": 0.0, "tau_sum": 0.0, "count": 0}


def update_physics_summary(summary, out, batch_size):
    if "r0" not in out or "rp" not in out or "cp" not in out:
        return
    r0 = out["r0"].detach()
    rp = out["rp"].detach()
    cp = out["cp"].detach()
    tau = rp * cp
    summary["r0_sum"] += float(r0.mean().item()) * batch_size
    summary["rp_sum"] += float(rp.mean().item()) * batch_size
    summary["cp_sum"] += float(cp.mean().item()) * batch_size
    summary["tau_sum"] += float(tau.mean().item()) * batch_size
    summary["count"] += batch_size


def finalize_physics_summary(summary):
    if summary["count"] <= 0:
        return None
    denom = float(summary["count"])
    return {
        "r0_mean": summary["r0_sum"] / denom,
        "rp_mean": summary["rp_sum"] / denom,
        "cp_mean": summary["cp_sum"] / denom,
        "tau_mean": summary["tau_sum"] / denom,
    }


def progress_label(args):
    return getattr(args, "stage", "training")


def train_one_epoch(
    model,
    loader,
    optimizer,
    x_scaler,
    y_scaler,
    device,
    args,
    target_align_loader=None,
    target_align_x_scaler=None,
):
    model.train()
    totals = {"total": 0.0, "pred": 0.0, "recon": 0.0, "phy": 0.0, "mmd": 0.0}
    count = 0
    all_pred = []
    all_label = []
    align_iterator = (
        iter(target_align_loader) if target_align_loader is not None else None
    )
    physics_summary = init_physics_summary()
    max_batches = len(loader)
    progress = tqdm(
        enumerate(loader),
        total=min(len(loader), max_batches),
        desc=f"train {progress_label(args)}",
        leave=False,
        dynamic_ncols=True,
    )
    for batch_idx, batch in progress:
        vehicle_ids = batch["vehicle_id"]
        raw_labels = batch["label"].detach().cpu().to(dtype=torch.float32).view(-1)
        features = normalize_features(batch["features"], vehicle_ids, x_scaler, device)
        pinn = batch["pinn"].to(device=device, dtype=torch.float32)
        labels = normalize_labels(batch["label"], vehicle_ids, y_scaler, device)
        out = model(features, pinn=pinn)
        loss_mmd = features.new_tensor(0.0)
        if target_align_loader is not None:
            try:
                align_batch = next(align_iterator)
            except StopIteration:
                align_iterator = iter(target_align_loader)
                align_batch = next(align_iterator)
            align_vehicle_ids = align_batch["vehicle_id"]
            align_features = normalize_features(
                align_batch["features"],
                align_vehicle_ids,
                target_align_x_scaler,
                device,
            )
            align_out = model(align_features)
            src_latent = out["ssp_latent"]
            tgt_latent = align_out["ssp_latent"]
            loss_mmd = maximum_mean_discrepancy(src_latent, tgt_latent)
        losses = compute_losses(out, features, labels, args, loss_mmd=loss_mmd)
        optimizer.zero_grad(set_to_none=True)
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        bsz = features.size(0)
        for key in totals:
            totals[key] += losses[key].item() * bsz
        count += bsz
        pred_raw = denormalize_labels(out["pred"].detach(), vehicle_ids, y_scaler)
        all_pred.append(pred_raw)
        all_label.append(raw_labels)
        update_physics_summary(physics_summary, out, bsz)
        progress.set_postfix(
            total=f"{losses['total'].item():.4f}",
            pred=f"{losses['pred'].item():.4f}",
            recon=f"{losses['recon'].item():.4f}",
            phy=f"{losses['phy'].item():.4f}",
            mmd=f"{losses['mmd'].item():.4f}",
        )
    loss_stats = {key: value / max(count, 1) for (key, value) in totals.items()}
    metrics = regression_metrics(torch.cat(all_pred), torch.cat(all_label))
    return (loss_stats, metrics, finalize_physics_summary(physics_summary))


@torch.no_grad()
def evaluate(
    model,
    loader,
    x_scaler,
    y_scaler,
    device,
    args,
    split_name=None,
    return_predictions=False,
):
    model.eval()
    totals = {"total": 0.0, "pred": 0.0, "recon": 0.0, "phy": 0.0, "mmd": 0.0}
    count = 0
    all_pred = []
    all_label = []
    prediction_rows = []
    physics_summary = init_physics_summary()
    max_batches = len(loader)
    desc = f"eval {split_name or 'split'}"
    progress = tqdm(
        enumerate(loader),
        total=min(len(loader), max_batches),
        desc=desc,
        leave=False,
        dynamic_ncols=True,
    )
    for batch_idx, batch in progress:
        vehicle_ids = batch["vehicle_id"]
        raw_labels = batch["label"].detach().cpu().to(dtype=torch.float32).view(-1)
        features = normalize_features(batch["features"], vehicle_ids, x_scaler, device)
        pinn = batch["pinn"].to(device=device, dtype=torch.float32)
        labels = normalize_labels(batch["label"], vehicle_ids, y_scaler, device)
        out = model(features, pinn=pinn)
        losses = compute_losses(out, features, labels, args)
        bsz = features.size(0)
        for key in totals:
            totals[key] += losses[key].item() * bsz
        count += bsz
        pred_raw = denormalize_labels(out["pred"].detach(), vehicle_ids, y_scaler)
        all_pred.append(pred_raw)
        all_label.append(raw_labels)
        update_physics_summary(physics_summary, out, bsz)
        progress.set_postfix(
            total=f"{losses['total'].item():.4f}",
            pred=f"{losses['pred'].item():.4f}",
            recon=f"{losses['recon'].item():.4f}",
            phy=f"{losses['phy'].item():.4f}",
        )
        if return_predictions:
            pred_cpu = pred_raw.numpy()
            label_cpu = raw_labels.numpy()
            vehicle_cpu = batch.get("vehicle_id")
            sample_cpu = batch.get("sample_idx")
            cycle_cpu = batch.get("cycle_number")
            for i in range(bsz):
                row = {
                    "split": split_name,
                    "vehicle_id": (
                        int(vehicle_cpu[i].item()) if vehicle_cpu is not None else ""
                    ),
                    "sample_idx": (
                        int(sample_cpu[i].item()) if sample_cpu is not None else ""
                    ),
                    "cycle_number": (
                        int(cycle_cpu[i].item()) if cycle_cpu is not None else ""
                    ),
                    "y_true": float(label_cpu[i]),
                    "y_pred": float(pred_cpu[i]),
                    "abs_error": float(abs(pred_cpu[i] - label_cpu[i])),
                }
                prediction_rows.append(row)
    loss_stats = {key: value / max(count, 1) for (key, value) in totals.items()}
    metrics = regression_metrics(torch.cat(all_pred), torch.cat(all_label))
    physics_summary = finalize_physics_summary(physics_summary)
    if return_predictions:
        return (loss_stats, metrics, prediction_rows, physics_summary)
    return (loss_stats, metrics, physics_summary)


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_checkpoint(path, model, config, x_scaler, y_scaler):
    payload = {
        "model_state_dict": dict(model.state_dict()),
        "config": asdict(config),
        "x_scaler": _clone_scaler_to_cpu(x_scaler),
        "y_scaler": _clone_scaler_to_cpu(y_scaler),
    }
    if any((key.endswith("lora_a") for key in payload["model_state_dict"])):
        payload["dora"] = {"rank": 64, "alpha": 16.0, "eps": 5e-06}
    torch.save(payload, path)


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_model_state(path, model, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


def main(output_dir):
    args = SimpleNamespace(
        lambda_pred=1.0,
        lambda_recon=0.025,
        lambda_phy=10.0,
        lambda_mmd=0.0,
        grad_clip=1.0,
        stage="pretrain",
    )
    set_seed(2026)
    device = resolve_device("cuda")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_dataset, val_dataset, test_dataset, split_info = build_datasets()
    split_scalers = prepare_split_scalers(
        train_dataset=train_dataset, val_dataset=val_dataset, test_dataset=test_dataset
    )
    target_align_loader = None
    target_align_x_scaler = None
    train_loader = DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    feature_dim = train_dataset[0]["features"].shape[-1]
    config = build_config(feature_dim)
    print(
        f"[setup] seed=2026 window=50 horizon=10 raw_length=128 split={split_info} config={config}"
    )
    model = PicolNet(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    print(
        f"[pretrain:start] task={infer_pretrain_task_name(args)} variant={infer_pretrain_variant_name(args)} dataset={3} train={split_info.get('train_vehicles')} val={split_info.get('val_vehicles')} test={split_info.get('test_vehicles')} windows={split_info.get('train_windows')}/{split_info.get('val_windows')}/{split_info.get('test_windows')} device={device} out={output_dir}"
    )
    torch.save(
        {
            "split_scalers": {
                split_name: {
                    "x": _clone_scaler_to_cpu(split_scaler["x"]),
                    "y": _clone_scaler_to_cpu(split_scaler["y"]),
                }
                for (split_name, split_scaler) in split_scalers.items()
            },
            "x_scaler": _clone_scaler_to_cpu(split_scalers["train"]["x"]),
            "y_scaler": _clone_scaler_to_cpu(split_scalers["train"]["y"]),
            "mean": (
                None
                if split_scalers["train"]["x"]["scope"] != "train"
                else split_scalers["train"]["x"]["mean"].cpu()
            ),
            "std": (
                None
                if split_scalers["train"]["x"]["scope"] != "train"
                else split_scalers["train"]["x"]["std"].cpu()
            ),
        },
        output_dir / "feature_scaler.pt",
    )
    rows = []
    best_rmse = float("inf")
    best_epoch = 0
    for epoch in range(1, 30 + 1):
        train_loss, train_metrics, train_physics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            x_scaler=split_scalers["train"]["x"],
            y_scaler=split_scalers["train"]["y"],
            device=device,
            args=args,
            target_align_loader=target_align_loader,
            target_align_x_scaler=target_align_x_scaler,
        )
        val_loss, val_metrics, val_physics = evaluate(
            model=model,
            loader=val_loader,
            x_scaler=split_scalers["val"]["x"],
            y_scaler=split_scalers["val"]["y"],
            device=device,
            args=args,
        )
        row = {
            "stage": "pretrain",
            "task": infer_pretrain_task_name(args),
            "variant": infer_pretrain_variant_name(args),
            "epoch": epoch,
            "train_total": train_loss["total"],
            "train_pred": train_loss["pred"],
            "train_recon": train_loss["recon"],
            "train_phy": train_loss["phy"],
            "train_mmd": train_loss["mmd"],
            "train_mae": train_metrics["mae"],
            "train_rmse": train_metrics["rmse"],
            "train_mape": train_metrics["mape"],
            "train_r2": train_metrics["r2"],
            "val_total": val_loss["total"],
            "val_pred": val_loss["pred"],
            "val_recon": val_loss["recon"],
            "val_phy": val_loss["phy"],
            "val_mmd": val_loss["mmd"],
            "val_mae": val_metrics["mae"],
            "val_rmse": val_metrics["rmse"],
            "val_mape": val_metrics["mape"],
            "val_r2": val_metrics["r2"],
        }
        add_weighted_loss_columns(row, "train", train_loss, args)
        add_weighted_loss_columns(row, "val", val_loss, args)
        rows.append(row)
        write_csv(output_dir / "metrics.csv", rows)
        if val_metrics["rmse"] < best_rmse:
            best_rmse = val_metrics["rmse"]
            best_epoch = epoch
            save_checkpoint(
                output_dir / "best.pt",
                model=model,
                config=config,
                x_scaler=split_scalers["train"]["x"],
                y_scaler=split_scalers["train"]["y"],
            )
            print(
                f"[pretrain:best] epoch={epoch:03d} val_rmse={best_rmse:.6f} checkpoint={output_dir / 'best.pt'}"
            )
        save_checkpoint(
            output_dir / "last.pt",
            model=model,
            config=config,
            x_scaler=split_scalers["train"]["x"],
            y_scaler=split_scalers["train"]["y"],
        )
        message = f"[pretrain task={infer_pretrain_task_name(args)} variant={infer_pretrain_variant_name(args)} out={output_dir}] epoch={epoch:03d} {format_loss_weights(args)} tr(total={train_loss['total']:.4f},raw_pred={train_loss['pred']:.4f},raw_rec={train_loss['recon']:.4f},raw_phy={train_loss['phy']:.4f},raw_mmd={train_loss['mmd']:.4f},{format_weighted_losses(train_loss, args)},r2={train_metrics['r2']:.4f}) va(rmse={val_metrics['rmse']:.4f},mae={val_metrics['mae']:.4f},r2={val_metrics['r2']:.4f})"
        if train_physics is not None:
            message += f" phy_tr(r0={train_physics['r0_mean']:.2e},rp={train_physics['rp_mean']:.2e},cp={train_physics['cp_mean']:.2e},tau={train_physics['tau_mean']:.2e})"
        if val_physics is not None:
            message += f" phy_va(r0={val_physics['r0_mean']:.2e},rp={val_physics['rp_mean']:.2e},cp={val_physics['cp_mean']:.2e},tau={val_physics['tau_mean']:.2e})"
        print(message)
    load_model_state(output_dir / "best.pt", model, device)
    val_loss, val_metrics, val_predictions, _ = evaluate(
        model=model,
        loader=val_loader,
        x_scaler=split_scalers["val"]["x"],
        y_scaler=split_scalers["val"]["y"],
        device=device,
        args=args,
        split_name="val",
        return_predictions=True,
    )
    test_loss, test_metrics, test_predictions, _ = evaluate(
        model=model,
        loader=test_loader,
        x_scaler=split_scalers["test"]["x"],
        y_scaler=split_scalers["test"]["y"],
        device=device,
        args=args,
        split_name="test",
        return_predictions=True,
    )
    final_metrics = {
        "best_epoch": best_epoch,
        "best_val": {"loss": val_loss, "regression": val_metrics},
        "test": {"loss": test_loss, "regression": test_metrics},
    }
    write_json(output_dir / "final_metrics.json", final_metrics)
    write_json(output_dir / "test_metrics.json", test_metrics)
    write_csv(output_dir / "val_predictions.csv", val_predictions)
    write_csv(output_dir / "test_predictions.csv", test_predictions)
    print(f"best_epoch={best_epoch} best_val_rmse={best_rmse:.6f}")
    print(
        f"test_rmse={test_metrics['rmse']:.6f} test_mae={test_metrics['mae']:.6f} test_r2={test_metrics['r2']:.6f}"
    )
    print(f"output_dir={output_dir}")
    print(
        "[raw]",
        {
            split: raw_segment_available_rate(dataset)
            for (split, dataset) in [
                ("train", train_dataset),
                ("val", val_dataset),
                ("test", test_dataset),
            ]
        },
    )
