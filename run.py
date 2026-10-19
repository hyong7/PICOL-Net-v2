import contextlib
import json
import os
import shutil
from pathlib import Path
import sys
from datetime import datetime

ROOT = Path(__file__).resolve().parent


def write_state(output, **state):
    state.update(
        pid=os.getpid(), output=str(output), updated_at=datetime.now().isoformat()
    )
    path = output / "state.json"
    temporary = path.with_suffix(".new")
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2))
    temporary.replace(path)


def export_weight(source, destination, stage):
    import torch

    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    keys = (
        "model_state_dict",
        "config",
        "x_scaler",
        "y_scaler",
        "dora",
        "memory",
        "experience",
        "dora_ema",
    )
    payload = {key: checkpoint[key] for key in keys if key in checkpoint}
    payload["model_state_dict"] = dict(payload["model_state_dict"])
    if stage == "finetune":
        payload["dora"] = {"rank": 64, "alpha": 16.0, "eps": 5e-6}
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, destination)
    from weights import load_model

    load_model(destination)


def main():
    from trainer_offline import main as pretrain
    from exp.finetune_target_dora import main as finetune
    from exp.online_memory import main_v2 as online

    output = (
        ROOT / "outputs" / ("3to1_fixed_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    )
    output.mkdir(parents=True, exist_ok=False)
    (ROOT / "outputs/latest.txt").write_text(str(output))
    stages = [
        (
            "pretrain",
            lambda: pretrain(output / "pretrain"),
            "pretrain/best.pt",
            "pretrain.pt",
        ),
        (
            "finetune",
            lambda: finetune(output / "finetune", output / "pretrain/best.pt"),
            "finetune/best.pt",
            "finetuned.pt",
        ),
        (
            "online",
            lambda: online(output / "online", output / "finetune/best.pt"),
            "online/after_online.pt",
            "after_online.pt",
        ),
    ]
    try:
        for stage, action, source, weight in stages:
            write_state(output, status="running", stage=stage)
            print(f"Starting {stage}: {output}", flush=True)
            with (output / f"{stage}.log").open("w", buffering=1) as log:
                with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                    action()
            export_weight(output / source, output / "weights" / weight, stage)
            print(f"Saved {weight}", flush=True)
        metrics = json.loads((output / "online/final_metrics.json").read_text())
        results = {key: metrics[key] for key in ("sample_count", "overall", "vehicles")}
        (output / "weights/results.json").write_text(
            json.dumps(results, ensure_ascii=False, indent=2)
        )
        (ROOT / "weights").mkdir(exist_ok=True)
        for name in ("pretrain.pt", "finetuned.pt", "after_online.pt", "results.json"):
            shutil.copy2(output / "weights" / name, ROOT / "weights" / name)
        write_state(
            output, status="complete", stage="online", metrics=metrics["overall"]
        )
        print(json.dumps(metrics["overall"]), flush=True)
        return output
    except BaseException as error:
        write_state(output, status="failed", stage=stage, error=str(error))
        raise


if __name__ == "__main__":
    if len(sys.argv) != 1:
        raise SystemExit("Run python run.py without options.")
    main()
