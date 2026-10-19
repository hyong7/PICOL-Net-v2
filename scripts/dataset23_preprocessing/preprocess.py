#!/usr/bin/env python3
"""将 dataset2 或 dataset3 的原始 PKL 整理到全新数据目录。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .core import (FEATURE_COLUMNS, KALMAN_Q, KALMAN_R, NOMINAL_CAPACITY_AH,
                       SIGNAL_MAP, SOURCE_DATASET, integer_key, load_pickle_tuple,
                       make_soh, pinn_arrays, raw_feature_row)
except ImportError:
    from core import (FEATURE_COLUMNS, KALMAN_Q, KALMAN_R, NOMINAL_CAPACITY_AH,
                      SIGNAL_MAP, SOURCE_DATASET, integer_key, load_pickle_tuple,
                      make_soh, pinn_arrays, raw_feature_row)

PROJECT = Path(__file__).resolve().parents[2]


def validate_output(raw_root: Path, output: Path) -> None:
    protected = (PROJECT / "data", PROJECT / "scripts/online_dualpool_final", raw_root)
    if any(output == root.resolve() or root.resolve() in output.parents for root in protected):
        raise ValueError(f"输出不可位于既有数据、标准结果或输入目录中: {output}")
    if output.exists():
        raise FileExistsError(f"输出根目录必须全新: {output}")


def build(dataset_id: int, raw_root: Path, output: Path, cars: list[int], label_mode: str) -> dict:
    raw_root, output = raw_root.resolve(), output.resolve()
    if dataset_id not in SOURCE_DATASET:
        raise ValueError("仅支持 dataset2、dataset3")
    if label_mode not in ("raw", "zero-phase-kalman"):
        raise ValueError(f"未知标签模式: {label_mode}")
    if not cars or any(car < 0 for car in cars):
        raise ValueError("必须明确提供非负车辆编号")
    validate_output(raw_root, output)
    if not raw_root.is_dir():
        raise FileNotFoundError(raw_root)
    paths = sorted(raw_root.glob("*.pkl"), key=lambda p: integer_key(p.stem, "pkl_id"))
    if not paths:
        raise ValueError(f"没有找到数字编号 PKL: {raw_root}")
    wanted = set(cars)
    rows = []
    output.mkdir(parents=True, exist_ok=False)
    for ordinal, path in enumerate(paths, 1):
        try:
            series, meta = load_pickle_tuple(path)
            car = integer_key(meta["car"], "car")
            if car not in wanted:
                continue
            cycle = integer_key(meta["charge_segment"], "charge_segment")
            try:
                capacity = float(meta.get("capacity"))
            except (TypeError, ValueError):
                capacity = np.nan
            rows.append({"car": car, "cycle_number": cycle,
                         "pkl_id": int(path.stem), "source_path": str(path),
                         "capacity_raw": capacity if np.isfinite(capacity) and capacity > 0 else np.nan,
                         **raw_feature_row(series)})
        except Exception as exc:
            raise ValueError(f"预处理失败: {path}: {exc}") from exc
        if ordinal % 10000 == 0:
            print(f"已读取 {ordinal}/{len(paths)} 个 PKL，已选片段 {len(rows)}", flush=True)
    if not rows:
        raise ValueError("指定车辆没有可用片段")
    segments = pd.DataFrame(rows)
    missing = wanted - set(segments.car)
    if missing:
        raise ValueError(f"未找到指定车辆: {sorted(missing)}")
    segments.to_csv(output / "segments.csv", index=False)
    dataset_root = output / f"dataset{dataset_id}"
    processed = dataset_root / "processed"
    (processed / "vehicles").mkdir(parents=True)
    indices, cycle_indices = [], []
    valid_points = total_points = 0
    for car, group in segments.groupby("car", sort=True):
        vehicle = f"dataset{dataset_id}_car_{car}"
        grouped = group.groupby("cycle_number", sort=True)
        features = grouped[FEATURE_COLUMNS].mean()
        capacity = grouped.capacity_raw.median().reindex(features.index).to_numpy()
        cycles = features.index.to_numpy(dtype=np.int64)
        soh = make_soh(cycles, capacity, label_mode)
        frame = features.reset_index()
        frame.insert(0, "vehicle_id", vehicle)
        frame["soh"] = soh
        frame.to_csv(processed / "vehicles" / f"{vehicle}.csv", index=False)
        representative = group.sort_values("pkl_id").drop_duplicates("cycle_number").set_index("cycle_number")
        index = pd.DataFrame({"dataset_id": f"dataset{dataset_id}",
                              "source_dataset": SOURCE_DATASET[dataset_id],
                              "vehicle_id": vehicle, "cycle_number": cycles,
                              "capacity_raw": capacity, "capacity": soh * NOMINAL_CAPACITY_AH,
                              "capacity_source": label_mode, "soh": soh,
                              "source_path": representative.loc[cycles, "source_path"].to_numpy(),
                              "source_pkl_id": representative.loc[cycles, "pkl_id"].to_numpy()})
        indices.append(index)
        for row in index.itertuples(index=False):
            series, meta = load_pickle_tuple(Path(row.source_path))
            if (integer_key(meta["car"], "car") != car or
                    integer_key(meta["charge_segment"], "charge_segment") != row.cycle_number):
                raise ValueError(f"代表片段元数据不匹配: {row.source_path}")
            arrays = pinn_arrays(series)
            target = dataset_root / "cycles" / vehicle / f"cycle_{row.cycle_number:06d}.npz"
            target.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(target, **arrays)
            n_valid = int(arrays["mask"].sum())
            valid_points += n_valid
            total_points += len(series)
            cycle_indices.append({"vehicle_id": vehicle, "cycle_number": row.cycle_number,
                                  "cycle_path": str(target.relative_to(output)),
                                  "source_path": row.source_path, "num_points": len(series),
                                  "valid_points": n_valid})
    index = pd.concat(indices, ignore_index=True)
    index.to_csv(processed / "index.csv", index=False)
    pd.DataFrame(cycle_indices).to_csv(dataset_root / "cycles/index.csv", index=False)
    report = {"status": "完成", "dataset_id": dataset_id, "source_dataset": SOURCE_DATASET[dataset_id],
              "raw_root": str(raw_root), "output_root": str(output), "cars": sorted(wanted),
              "input_pkl_count": len(paths), "selected_segments": len(segments), "cycles": len(index),
              "feature_columns": FEATURE_COLUMNS, "signal_map": SIGNAL_MAP,
              "voltage_semantics": "mean V，第0列不除以串联数", "std_ddof": 0,
              "label_mode": label_mode, "nominal_capacity_ah": NOMINAL_CAPACITY_AH,
              "kalman_q": KALMAN_Q, "kalman_r": KALMAN_R,
              "uses_future_capacity": label_mode == "zero-phase-kalman",
              "raw_segment_availability": len(cycle_indices) / len(index),
              "raw_valid_point_rate": valid_points / total_points,
              "raw_capacity_cycle_rate": float(index.capacity_raw.notna().mean()),
              "finite_soh_rate": float(np.isfinite(index.soh).mean()),
              "split": None, "seed": None, "horizon": None, "model": None,
              "loss_weights": None, "metrics": None,
              "experiment_fields_note": "确定性预处理阶段，不涉及训练、划分或预测指标"}
    (processed / "build_meta.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=int, choices=(2, 3), required=True)
    parser.add_argument("--raw-root", type=Path, required=True, help="数字编号 PKL 所在目录")
    parser.add_argument("--output-root", type=Path, required=True, help="全新输出目录")
    parser.add_argument("--cars", type=int, nargs="+", required=True, help="THU meta.car 车辆编号")
    parser.add_argument("--label-mode", choices=("raw", "zero-phase-kalman"), required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.dataset, args.raw_root, args.output_root, args.cars, args.label_mode),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
