from trainer_offline import raw_segment_available_rate
import math
import sys
from pathlib import Path
from types import SimpleNamespace
import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from models.dora import (
    dora_parameter_report,
    freeze_non_dora_parameters,
    inject_dora_linear_layers,
)
from models.picolnet import PicolNet
from src.config import ECMConfig, ModelConfig, PredictorConfig, SSPConfig
from src.dataloader import (
    BatteryWindowDataset,
    MultiVehicleBatteryDataset,
    list_vehicle_ids,
)
from trainer_offline import (
    evaluate,
    load_scaler_from_checkpoint,
    resolve_device,
    resolve_transfer_split_scalers,
    save_checkpoint,
    set_seed,
    train_one_epoch,
    write_csv,
    write_json,
)


def config_from_checkpoint(checkpoint):
    c = checkpoint["config"]
    return ModelConfig(
        feature_dim=c["feature_dim"],
        window=c["window"],
        ssp=SSPConfig(**c["ssp"]),
        predictor=PredictorConfig(**c["predictor"]),
        ecm=ECMConfig(**c["ecm"]),
    )


def resolve_data_args():
    return {"window": 50, "horizon": 10, "stride": 1, "raw_length": 128}


def split_target_vehicle_ids(all_vehicle_ids):
    known = [1, 15, 19]
    return (known, [vehicle for vehicle in all_vehicle_ids if vehicle not in known])


def split_known_vehicle_dataset(dataset, val_ratio):
    if len(dataset) < 2:
        raise ValueError(
            "Known target vehicle must contain at least 2 windows for train/val split."
        )
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("target_val_ratio must be in (0, 1).")
    n_val = max(1, int(math.ceil(len(dataset) * val_ratio)))
    n_train = len(dataset) - n_val
    if n_train <= 0:
        n_train = 1
        n_val = len(dataset) - 1
    train_indices = list(range(n_train))
    val_indices = list(range(n_train, len(dataset)))
    return (Subset(dataset, train_indices), Subset(dataset, val_indices))


def build_target_datasets(args, data_args):
    all_vehicle_ids = list_vehicle_ids(1)
    known_ids, test_ids = split_target_vehicle_ids(all_vehicle_ids)
    train_parts = []
    val_parts = []
    known_windows = {}
    for vehicle_id in known_ids:
        dataset = BatteryWindowDataset(
            dataset_id=1,
            vehicle_id=vehicle_id,
            window=data_args["window"],
            horizon=data_args["horizon"],
            stride=data_args["stride"],
            raw_length=data_args["raw_length"],
        )
        train_dataset, val_dataset = split_known_vehicle_dataset(dataset, 0.2)
        train_parts.append(train_dataset)
        val_parts.append(val_dataset)
        known_windows[vehicle_id] = {
            "total_windows": len(dataset),
            "train_windows": len(train_dataset),
            "val_windows": len(val_dataset),
        }
    train_dataset = ConcatDataset(train_parts)
    val_dataset = ConcatDataset(val_parts)
    test_dataset = MultiVehicleBatteryDataset(
        dataset_id=1,
        vehicle_ids=test_ids,
        window=data_args["window"],
        horizon=data_args["horizon"],
        stride=data_args["stride"],
        raw_length=data_args["raw_length"],
    )
    split_info = {
        "mode": "target_vehicle_finetune",
        "known_vehicle_ratio": 0.1,
        "target_val_ratio": 0.2,
        "known_vehicles": known_ids,
        "test_vehicles": test_ids,
        "known_vehicle_windows": known_windows,
        "train_windows": len(train_dataset),
        "val_windows": len(val_dataset),
        "test_windows": len(test_dataset),
    }
    return (train_dataset, val_dataset, test_dataset, split_info)


def build_source_alignment_dataset(data_args):
    vehicles = [346, 481, 341, 406, 431, 369, 392]
    dataset = MultiVehicleBatteryDataset(
        dataset_id=3, vehicle_ids=vehicles, **data_args
    )
    return (dataset, {"vehicle_ids": vehicles, "num_windows": len(dataset)})


def build_loader(dataset, batch_size, shuffle, num_workers, device):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )


@torch.no_grad()
def baseline_evaluate(model, loader, x_scaler, y_scaler, device, args):
    loss, metrics, _ = evaluate(
        model=model,
        loader=loader,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        device=device,
        args=args,
    )
    return {"loss": loss, "regression": metrics}


def configure_finetune_mode(model):
    replaced_modules = inject_dora_linear_layers(model)
    freeze_non_dora_parameters(model)
    report = {
        "mode": "dora",
        **dora_parameter_report(model, replaced_modules=replaced_modules),
    }
    return report


def main(output_dir, source_checkpoint):
    args = SimpleNamespace(
        lambda_pred=1.0,
        lambda_recon=0.025,
        lambda_phy=10.0,
        lambda_mmd=0.3,
        grad_clip=0.5,
        stage="finetune",
    )
    set_seed(2026)
    device = resolve_device("cuda")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(source_checkpoint, map_location=device, weights_only=False)
    config = config_from_checkpoint(checkpoint)
    data_args = resolve_data_args()
    train_dataset, val_dataset, test_dataset, split_info = build_target_datasets(
        args, data_args
    )
    source_x_scaler = load_scaler_from_checkpoint(checkpoint, "x")
    source_y_scaler = load_scaler_from_checkpoint(checkpoint, "y")
    split_scalers = resolve_transfer_split_scalers(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        source_x_scaler=source_x_scaler,
        source_y_scaler=source_y_scaler,
    )
    print(
        f"[setup] seed=2026 window=50 horizon=10 raw_length=128 split={split_info} config={config}"
    )
    model = PicolNet(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    loss_args = SimpleNamespace(
        lambda_pred=1.0,
        lambda_recon=0.025,
        lambda_phy=10.0,
        lambda_mmd=0.3,
        grad_clip=0.5,
    )
    train_loader = build_loader(train_dataset, 128, True, 0, device)
    val_loader = build_loader(val_dataset, 128, False, 0, device)
    test_loader = build_loader(test_dataset, 128, False, 0, device)
    align_dataset = align_loader = None
    align_dataset, _ = build_source_alignment_dataset(data_args)
    align_loader = build_loader(align_dataset, 128, True, 0, device)
    baseline_val = baseline_evaluate(
        model,
        val_loader,
        split_scalers["val"]["x"],
        split_scalers["val"]["y"],
        device,
        loss_args,
    )
    baseline_test = baseline_evaluate(
        model,
        test_loader,
        split_scalers["test"]["x"],
        split_scalers["test"]["y"],
        device,
        loss_args,
    )
    finetune_report = configure_finetune_mode(model)
    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not trainable:
        raise RuntimeError(
            f"No trainable parameters found for finetune mode: {'dora'}."
        )
    optimizer = torch.optim.AdamW(trainable, lr=0.001, weight_decay=0.0)
    rows = []
    best_rmse = float("inf")
    best_epoch = 0
    for epoch in range(1, 50 + 1):
        train_loss, _train_metrics, train_physics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            x_scaler=split_scalers["train"]["x"],
            y_scaler=split_scalers["train"]["y"],
            device=device,
            args=loss_args,
            target_align_loader=align_loader,
            target_align_x_scaler=source_x_scaler,
        )
        val_loss, val_metrics, val_physics = evaluate(
            model=model,
            loader=val_loader,
            x_scaler=split_scalers["val"]["x"],
            y_scaler=split_scalers["val"]["y"],
            device=device,
            args=loss_args,
        )
        row = {
            "epoch": epoch,
            "train_total": train_loss["total"],
            "train_pred": train_loss["pred"],
            "train_recon": train_loss["recon"],
            "train_phy": train_loss["phy"],
            "train_mmd": train_loss["mmd"],
            "train_weighted_mmd": 0.3 * train_loss["mmd"],
            "val_total": val_loss["total"],
            "val_pred": val_loss["pred"],
            "val_recon": val_loss["recon"],
            "val_phy": val_loss["phy"],
            "val_mae": val_metrics["mae"],
            "val_rmse": val_metrics["rmse"],
            "val_mape": val_metrics["mape"],
            "val_r2": val_metrics["r2"],
        }
        if train_physics is not None:
            for k, v in train_physics.items():
                row[f"train_{k}"] = v
        if val_physics is not None:
            for k, v in val_physics.items():
                row[f"val_{k}"] = v
        rows.append(row)
        write_csv(output_dir / "metrics.csv", rows)
        if val_metrics["rmse"] < best_rmse:
            best_rmse = val_metrics["rmse"]
            best_epoch = epoch
            save_checkpoint(
                output_dir / "best.pt",
                model=model,
                config=config,
                x_scaler=source_x_scaler,
                y_scaler=source_y_scaler,
            )
        save_checkpoint(
            output_dir / "last.pt",
            model=model,
            config=config,
            x_scaler=source_x_scaler,
            y_scaler=source_y_scaler,
        )
        print(
            f"epoch={epoch:03d} train_total={train_loss['total']:.6f} train_pred={train_loss['pred']:.6f} train_recon={train_loss['recon']:.6f} train_phy={train_loss['phy']:.6f} train_mmd={train_loss['mmd']:.6f} val_rmse={val_metrics['rmse']:.6f} val_mae={val_metrics['mae']:.6f} val_r2={val_metrics['r2']:.6f}"
        )
    best_checkpoint = torch.load(
        output_dir / "best.pt", map_location=device, weights_only=False
    )
    model.load_state_dict(best_checkpoint["model_state_dict"])
    val_loss, val_metrics, val_predictions, _ = evaluate(
        model=model,
        loader=val_loader,
        x_scaler=split_scalers["val"]["x"],
        y_scaler=split_scalers["val"]["y"],
        device=device,
        args=loss_args,
        split_name="val",
        return_predictions=True,
    )
    test_loss, test_metrics, test_predictions, _ = evaluate(
        model=model,
        loader=test_loader,
        x_scaler=split_scalers["test"]["x"],
        y_scaler=split_scalers["test"]["y"],
        device=device,
        args=loss_args,
        split_name="test",
        return_predictions=True,
    )
    final_metrics = {
        "best_epoch": best_epoch,
        "baseline_before_finetune": {"val": baseline_val, "test": baseline_test},
        "finetune_mode": "dora",
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
    print(f"finetune_mode={'dora'}")
    print(f"finetune_trainable_params={finetune_report['trainable_params']}")
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
