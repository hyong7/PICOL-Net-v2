from dataclasses import asdict
from copy import deepcopy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import torch
from torch.utils.data import DataLoader

from models.picolnet import PicolNet
from models.dora import inject_dora_linear_layers, freeze_non_dora_parameters
from models.ecm import ecm_voltage_loss
from models.memory_v2 import MemoryV2
from src.config import ModelConfig, SSPConfig, PredictorConfig
from trainer_offline import train_one_epoch
from exp.online_memory import (
    LABEL_DELAY,
    _stream_vehicle_v2,
    _ema_dora,
    resolve_memory_update_indices,
    resolve_memory_update_seed,
)
from weights import load_model, load_memory


def main():
    torch.set_num_threads(2)
    torch.manual_seed(2026)
    config = ModelConfig(
        43,
        window=4,
        ssp=SSPConfig(d_model=16, num_heads=4, num_layers=1, dropout=0),
        predictor=PredictorConfig(hidden_dim=16, num_layers=1, d_state=4, dropout=0),
    )
    model = PicolNet(config)
    pinn = torch.zeros(4, 8, 5)
    pinn[..., 1] = 70
    pinn[..., 2] = 0.1
    pinn[..., 3] = 3.7
    pinn[..., 4] = 1
    items = [
        {
            "features": torch.randn(4, 43),
            "pinn": pinn,
            "label": torch.tensor(0.9),
            "vehicle_id": torch.tensor(2),
            "sample_idx": torch.tensor(i),
            "cycle_number": torch.tensor(60 + i),
        }
        for i in range(32)
    ]
    out = model(items[0]["features"].unsqueeze(0), pinn.unsqueeze(0))
    ecm_voltage_loss(out["voltage_pred"], out["voltage_true"]).backward()
    assert model.predictor.input_proj.weight.grad.abs().sum() > 0
    assert model.ssp.input_proj.weight.grad.abs().sum() > 0
    replaced = inject_dora_linear_layers(model)
    assert all(
        "token_decoder" not in name and "encoder_layer" not in name for name in replaced
    )
    freeze_non_dora_parameters(model)
    x = {"kind": "x", "scope": "train", "mean": torch.zeros(43), "std": torch.ones(43)}
    y = {"kind": "y", "scope": "train", "mean": torch.zeros(1), "std": torch.ones(1)}
    args = SimpleNamespace(
        lambda_pred=0, lambda_recon=0, lambda_phy=0, lambda_mmd=0.3, grad_clip=0.5
    )
    before = model.ssp.input_proj.lora_b.detach().clone()
    align = [{**item, "features": item["features"] + 2} for item in items[:4]]
    params = [p for p in model.parameters() if p.requires_grad]
    losses, _, _ = train_one_epoch(
        model,
        DataLoader(items[:4], batch_size=4),
        torch.optim.AdamW(params, lr=0.001, weight_decay=0),
        x,
        y,
        torch.device("cpu"),
        args,
        target_align_loader=DataLoader(align, batch_size=4),
        target_align_x_scaler=x,
    )
    assert losses["mmd"] > 0 and abs(losses["total"] - 0.3 * losses["mmd"]) < 1e-6
    assert not torch.equal(before, model.ssp.input_proj.lora_b)
    memory = MemoryV2(16)
    memory.seed_anchor(
        torch.ones(1, 16),
        torch.tensor([0.2]),
        torch.tensor([0.8]),
        torch.tensor([1.0]),
        torch.tensor([1]),
    )
    model.eval()
    counterfactual_model = deepcopy(model)
    counterfactual_memory = deepcopy(memory)
    loss_args = SimpleNamespace(
        lambda_pred=1, lambda_recon=0.025, lambda_phy=0.01, lambda_mmd=0
    )
    optimizer = torch.optim.AdamW(params, lr=0.0005, weight_decay=0)
    rows, metrics, _ = _stream_vehicle_v2(
        model,
        items,
        2,
        memory,
        x,
        y,
        torch.device("cpu"),
        SimpleNamespace(),
        optimizer,
        loss_args,
        0,
    )
    assert len(rows) == 32 and metrics["memory_update_samples"] == 5
    assert metrics["supervised_updates"] > 0
    visible = set(
        resolve_memory_update_indices(32, 0.15, resolve_memory_update_seed(2026, 2))
    )
    arrived = {i for i in visible if i + LABEL_DELAY < len(items)}
    assert sum(row["memory_update_enabled"] for row in rows) == len(visible)
    assert metrics["memory_updates_applied"] == len(arrived)
    assert metrics["supervised_updates"] == sum(i >= 20 for i in arrived)
    for row in rows:
        source = row["update_source_idx"]
        expected = row["stream_idx"] - LABEL_DELAY
        assert source == (expected if expected in arrived else None)
        assert row["experience_size"] == sum(
            i + LABEL_DELAY <= row["stream_idx"] for i in arrived
        )
    hidden_changed = [
        {**item, "label": item["label"] + (100 if i not in arrived else 0)}
        for i, item in enumerate(items)
    ]
    counterfactual_params = [
        p for p in counterfactual_model.parameters() if p.requires_grad
    ]
    counterfactual_rows, _, _ = _stream_vehicle_v2(
        counterfactual_model,
        hidden_changed,
        2,
        counterfactual_memory,
        x,
        y,
        torch.device("cpu"),
        SimpleNamespace(),
        torch.optim.AdamW(counterfactual_params, lr=0.0005, weight_decay=0),
        loss_args,
        0,
    )
    torch.testing.assert_close(model.state_dict(), counterfactual_model.state_dict())
    torch.testing.assert_close(
        torch.tensor([r["y_pred"] for r in rows]),
        torch.tensor([r["y_pred"] for r in counterfactual_rows]),
    )
    assert memory.bias_state == counterfactual_memory.bias_state
    for field in memory.experience.__dataclass_fields__:
        torch.testing.assert_close(
            getattr(memory.experience, field),
            getattr(counterfactual_memory.experience, field),
        )
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "model.pt"
        torch.save(
            {
                "model_state_dict": dict(model.state_dict()),
                "config": asdict(config),
                "dora": {"rank": 64, "alpha": 16.0, "eps": 5e-6},
            },
            path,
        )
        restored, _ = load_model(path)
        with torch.no_grad():
            torch.testing.assert_close(
                model(items[0]["features"].unsqueeze(0))["pred"],
                restored(items[0]["features"].unsqueeze(0))["pred"],
            )
    restored_memory = load_memory(
        {
            "memory": memory.state_dict(),
            "experience": {
                k: getattr(memory.experience, k)
                for k in memory.experience.__dataclass_fields__
            },
        }
    )
    torch.testing.assert_close(
        memory.retrieve(torch.ones(1, 16))["residual_delta"],
        restored_memory.retrieve(torch.ones(1, 16))["residual_delta"],
    )
    torch.testing.assert_close(
        _ema_dora({"w": torch.zeros(1)}, {"w": torch.ones(1)}, 0.1)["w"],
        torch.tensor([0.1]),
    )
    print(
        "PASS: PINN gradients, DoRA, MMD, 15% visible labels, delayed updates, hidden-label independence and weight restoration"
    )


if __name__ == "__main__":
    main()
