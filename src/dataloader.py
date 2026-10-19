import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_LENGTH = 128


def resample_pinn(time, current, soc, mean_v, target_length=RAW_LENGTH):

    t_new = np.linspace(time[0], time[-1], target_length).astype(np.float32)
    time_out = t_new - time[0]
    dt = np.diff(time_out, prepend=time_out[:1]).astype(np.float32)

    return {
        "time": time_out,
        "current": np.interp(t_new, time, current).astype(np.float32),
        "soc": np.interp(t_new, time, soc).astype(np.float32),
        "mean_v": np.interp(t_new, time, mean_v).astype(np.float32),
        "dt": dt,
    }


def load_pinn_data(dataset_id, vehicle_id, cycle_id, target_length=RAW_LENGTH):
    npz_path = (
        ROOT
        / f"data/datasets/dataset{dataset_id}/cycles/dataset{dataset_id}_car_{vehicle_id}/cycle_{cycle_id:06d}.npz"
    )
    data = np.load(npz_path)

    time = data["time"]
    current = data["current"]
    soc = data["soc"]
    mean_v = data["mean_v"]

    return resample_pinn(time, current, soc, mean_v, target_length)


def make_window(features, labels, pinn_data, window=50, horizon=30, stride=1):
    X_main = []
    X_pinn = []
    y = []

    for i in range(0, len(features) - window - horizon + 1, stride):
        X_main.append(features[i : i + window])
        X_pinn.append(pinn_data[i : i + window])
        y.append(labels[i + window + horizon - 1])

    return (
        np.asarray(X_main, dtype=np.float32),
        np.asarray(X_pinn),
        np.asarray(y, dtype=np.float32),
    )


def load_main_data(dataset_id, vehicle_id):
    path = (
        ROOT
        / f"data/datasets/dataset{dataset_id}/processed/vehicles/dataset{dataset_id}_car_{vehicle_id}.csv"
    )
    df = pd.read_csv(path)

    features = df.drop(columns=["soh", "vehicle_id", "cycle_number"]).to_numpy(
        dtype=np.float32
    )
    labels = df["soh"].to_numpy(dtype=np.float32)
    cycle_numbers = df["cycle_number"].to_numpy()
    cycle_index = (cycle_numbers - cycle_numbers.min()) / (
        cycle_numbers.max() - cycle_numbers.min() + 1e-8
    )
    features = np.column_stack([features, cycle_index.astype(np.float32)])
    return features, labels, cycle_numbers


def list_vehicle_ids(dataset_id):
    vehicle_dir = ROOT / f"data/datasets/dataset{dataset_id}/processed/vehicles"
    prefix = f"dataset{dataset_id}_car_"
    vehicle_ids = []

    for path in sorted(
        vehicle_dir.glob(f"{prefix}*.csv"), key=lambda p: int(p.stem[len(prefix) :])
    ):
        stem = path.stem
        if stem.startswith(prefix):
            vehicle_ids.append(int(stem[len(prefix) :]))

    return vehicle_ids


class SingleDataset(Dataset):
    def __init__(
        self,
        dataset_id,
        vehicle_id,
        window=50,
        horizon=30,
        stride=1,
        raw_length=RAW_LENGTH,
    ):
        self.dataset_id = dataset_id
        self.vehicle_id = vehicle_id
        self.window = window
        self.horizon = horizon
        self.stride = stride
        self.raw_length = raw_length

        features, labels, cycle_numbers = load_main_data(dataset_id, vehicle_id)

        pinn_data = []
        for cycle_id in cycle_numbers:
            pinn = load_pinn_data(
                dataset_id=dataset_id,
                vehicle_id=vehicle_id,
                cycle_id=int(cycle_id),
                target_length=raw_length,
            )
            pinn_data.append(
                np.stack(
                    [
                        pinn["time"],
                        pinn["soc"],
                        pinn["current"],
                        pinn["mean_v"],
                        pinn["dt"],
                    ],
                    axis=-1,
                )
            )

        pinn_data = np.asarray(pinn_data, dtype=np.float32)

        self.X_main, self.X_pinn, self.y = make_window(
            features=features,
            labels=labels,
            pinn_data=pinn_data,
            window=window,
            horizon=horizon,
            stride=stride,
        )
        label_indices = np.arange(window + horizon - 1, len(cycle_numbers), stride)
        self.label_cycle_numbers = cycle_numbers[label_indices].astype(np.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {
            "features": torch.from_numpy(self.X_main[idx]),
            "pinn": torch.from_numpy(self.X_pinn[idx]),
            "label": torch.tensor(self.y[idx], dtype=torch.float32),
            "vehicle_id": torch.tensor(self.vehicle_id, dtype=torch.long),
            "sample_idx": torch.tensor(idx, dtype=torch.long),
            "cycle_number": torch.tensor(
                self.label_cycle_numbers[idx], dtype=torch.long
            ),
        }


class VehiclesDataset(Dataset):
    def __init__(
        self,
        dataset_id,
        vehicle_ids,
        window=50,
        horizon=30,
        stride=1,
        raw_length=RAW_LENGTH,
    ):
        self.dataset_id = dataset_id
        self.vehicle_ids = list(vehicle_ids)
        self.datasets = []
        self.index = []

        for vehicle_id in self.vehicle_ids:
            dataset = SingleDataset(
                dataset_id=dataset_id,
                vehicle_id=vehicle_id,
                window=window,
                horizon=horizon,
                stride=stride,
                raw_length=raw_length,
            )
            if len(dataset) == 0:
                continue

            dataset_idx = len(self.datasets)
            self.datasets.append(dataset)

            for sample_idx in range(len(dataset)):
                self.index.append((dataset_idx, sample_idx))

        if not self.index:
            raise ValueError("No valid windows found for requested vehicle_ids.")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        dataset_idx, sample_idx = self.index[idx]
        dataset = self.datasets[dataset_idx]
        item = dataset[sample_idx]
        return item


BatteryWindowDataset = SingleDataset
MultiVehicleBatteryDataset = VehiclesDataset
