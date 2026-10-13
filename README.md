# PICOL-Net v2

CUDA 12.8 · NVIDIA GeForce RTX 4060 Laptop GPU

```bash
uv venv --python 3.10.19
uv pip install torch==2.10.0 setuptools packaging ninja --torch-backend cu128
uv pip install -r requirements.txt --no-build-isolation
.venv/bin/python run.py
```

数据来源：

- **Dataset 1**：[原始数据](https://github.com/BatICM/battery-charging-data-of-on-road-electric-vehicles)，使用车辆 1–20。Deng et al. (2023). [Prognostics of battery capacity based on charging data and data-driven methods for on-road vehicles](https://doi.org/10.1016/j.apenergy.2023.120954). *Applied Energy*, 339, 120954.
- **Dataset 2 & 3**：[EVBattery 原始数据](https://figshare.com/articles/dataset/EVBattery_A_Large-Scale_Electric_Vehicle_Dataset_for_Battery_Health_and_Capacity_Estimation/23301881)。
- **Dataset 2**：使用 EVBattery 的 `battery_dataset1`，选中 23 辆车：18, 29, 33, 35, 43, 50, 51, 57, 69, 81, 87, 91, 94, 98, 139, 140, 141, 152, 153, 157, 175, 176, 187。
- **Dataset 3**：使用 EVBattery 的 `battery_dataset2`，选中 10 辆车：341, 346, 368, 369, 379, 392, 406, 428, 431, 481。

EVBattery：He et al. (2022). [EVBattery: A Large-Scale Electric Vehicle Dataset for Battery Health and Capacity Estimation](https://arxiv.org/abs/2201.12358)；数据 DOI：[10.6084/m9.figshare.23301881](https://doi.org/10.6084/m9.figshare.23301881)。

使用数据时请引用对应原始论文，并遵守原发布方的使用条款。
