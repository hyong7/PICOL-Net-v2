"""THU PKL 读取、42 维统计与历史离线标签处理。"""

from __future__ import annotations

import math
import pickle
from collections.abc import Mapping
from pathlib import Path

import numpy as np

SIGNAL_MAP = {"SOC": 2, "current": 1, "pack_V": 0, "min_V": 4,
              "max_V": 3, "min_T": 6, "max_T": 5}
STAT_NAMES = ("mean", "sum", "std", "max", "min", "skew")
FEATURE_COLUMNS = [f"{signal}_{stat}" for signal in SIGNAL_MAP for stat in STAT_NAMES]
SOURCE_DATASET = {2: "battery_dataset1", 3: "battery_dataset2"}
NOMINAL_CAPACITY_AH = 50.0
KALMAN_Q = 2.5e-6
KALMAN_R = 0.0025


def load_pickle_tuple(path: Path) -> tuple[np.ndarray, Mapping]:
    """跳过文件头，连续读取至 (series, meta)，支持历史 Torch pickle 流。"""
    with path.open("rb") as handle:
        while True:
            try:
                obj = pickle.load(handle)
            except EOFError as exc:
                raise ValueError(f"未找到 (series, meta): {path}") from exc
            if isinstance(obj, (tuple, list)) and len(obj) >= 2:
                series, meta = obj[:2]
                if isinstance(series, np.ndarray) and isinstance(meta, Mapping):
                    if series.ndim != 2 or series.shape[1] < 8 or not len(series):
                        raise ValueError(f"THU series 形状异常 {series.shape}: {path}")
                    return series, meta


def integer_key(value, name: str) -> int:
    number = float(value)
    if not math.isfinite(number) or number < 0 or not number.is_integer():
        raise ValueError(f"{name} 必须为非负整数，实际为 {value!r}")
    return int(number)


def unbiased_skew(values: np.ndarray) -> float:
    n = len(values)
    std = float(np.std(values))
    if n < 3 or std == 0:
        return 0.0
    g1 = float(np.mean((values - np.mean(values)) ** 3) / std**3)
    return math.sqrt(n * (n - 1)) / (n - 2) * g1


def raw_feature_row(series: np.ndarray) -> dict[str, float]:
    """逐信号剔除非有限值；std 使用 ddof=0，skew 使用无偏修正。"""
    row = {}
    for signal, column in SIGNAL_MAP.items():
        values = np.asarray(series[:, column], dtype=np.float64)
        values = values[np.isfinite(values)]
        if not len(values):
            raise ValueError(f"信号 {signal} 没有有限值")
        stats = (np.mean(values), np.sum(values), np.std(values), np.max(values),
                 np.min(values), unbiased_skew(values))
        row.update({f"{signal}_{name}": float(value)
                    for name, value in zip(STAT_NAMES, stats)})
    return row


def pinn_arrays(series: np.ndarray) -> dict[str, np.ndarray]:
    """保存 ECM 主输入原始量纲；pack_V 第 0 列已经是 mean V。"""
    arrays = {name: np.asarray(series[:, column], dtype=np.float32)
              for name, column in {"time": 7, "soc": 2, "current": 1, "mean_v": 0}.items()}
    arrays["mask"] = np.logical_and.reduce([np.isfinite(v) for v in arrays.values()])
    return arrays


def kalman_filter(values: np.ndarray) -> np.ndarray:
    estimate, error = float(values[0]), 1.0
    out = np.empty(len(values), dtype=np.float64)
    for i, measurement in enumerate(values):
        error += KALMAN_Q
        gain = error / (error + KALMAN_R)
        estimate += gain * (float(measurement) - estimate)
        error *= 1.0 - gain
        out[i] = estimate
    return out


def make_soh(cycles: np.ndarray, capacity: np.ndarray, mode: str) -> np.ndarray:
    raw = np.asarray(capacity, dtype=np.float64) / NOMINAL_CAPACITY_AH
    valid = np.isfinite(raw) & (raw > 0)
    if mode == "raw":
        return np.where(valid, raw, np.nan)
    if mode != "zero-phase-kalman":
        raise ValueError(f"未知标签模式: {mode}")
    if valid.sum() < 2:
        raise ValueError("零相位 Kalman 至少需要两个有效容量循环")
    measurements = raw[valid]
    smooth = 0.5 * (kalman_filter(measurements) + kalman_filter(measurements[::-1])[::-1])
    return np.interp(cycles, cycles[valid], smooth)
