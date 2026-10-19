# PICOL-Net v2

Battery health prediction and online adaptation with SSP, Mamba, and ECM-PINN.

## Setup

Python 3.10 · PyTorch 2.10 · CUDA 12.8

```bash
uv venv --python 3.10.19
uv pip install torch==2.10.0 setuptools packaging ninja --torch-backend cu128
uv pip install -r requirements.txt --no-build-isolation
.venv/bin/python run.py
```

## Data

- Dataset 1: [on-road charging data](https://github.com/BatICM/battery-charging-data-of-on-road-electric-vehicles), vehicles 1–20 ([paper](https://doi.org/10.1016/j.apenergy.2023.120954)).
- Datasets 2 and 3: [EVBattery](https://doi.org/10.6084/m9.figshare.23301881), `battery_dataset1` and `battery_dataset2`, respectively ([paper](https://arxiv.org/abs/2201.12358)).

Preprocess raw PKL files:

```bash
.venv/bin/python scripts/dataset23_preprocessing/preprocess.py \
  --dataset 2 --raw-root /path/to/battery_dataset1/data \
  --cars 18 29 --label-mode zero-phase-kalman \
  --output-root outputs/preprocessed
```

Use `--dataset 3` with the corresponding files and vehicle IDs for Dataset 3. Place the generated `dataset<N>/` under `data/datasets/`. The output directory must be new. Zero-phase Kalman labels use future capacity observations for offline processing; `raw` labels retain missing values.

## Check

```bash
.venv/bin/python check.py
```

A 120-cycle sample from each dataset passed preprocessing, CUDA training, prediction, and checkpoint reload checks. Full training is required to evaluate accuracy.
