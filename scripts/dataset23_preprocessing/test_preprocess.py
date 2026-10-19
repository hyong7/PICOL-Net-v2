"""验证多对象 PKL、循环聚合、物理量纲与输出隔离。"""

import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import skew

from scripts.dataset23_preprocessing.core import (
    FEATURE_COLUMNS, load_pickle_tuple, make_soh, raw_feature_row,
)
from scripts.dataset23_preprocessing.preprocess import PROJECT, build, validate_output


def write_sample(path, cycle, capacity, voltage):
    series = np.tile([voltage, -10, 40, 4.1, 3.0, 30, 20, 0], (5, 1)).astype(float)
    series[:, 7] = np.arange(5) * 10
    series[:, 2] += np.arange(5)
    with path.open("wb") as handle:
        for obj in (119547037146038801333356, 1001, {"protocol_version": 1001},
                    (series, {"car": 18, "charge_segment": cycle, "capacity": capacity})):
            pickle.dump(obj, handle)
    return series


class PreprocessingTests(unittest.TestCase):
    def test_header_and_invalid_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "0.pkl"
            expected = write_sample(path, 1, 40, 3.5)
            actual, meta = load_pickle_tuple(path)
            np.testing.assert_array_equal(actual, expected)
            self.assertEqual(meta["car"], 18)
            with path.open("wb") as handle:
                pickle.dump(1001, handle)
            with self.assertRaisesRegex(ValueError, "未找到"):
                load_pickle_tuple(path)

    def test_statistics_match_independent_reference(self):
        series = np.tile(np.array([1., 2., 3., 9., np.nan])[:, None], (1, 8))
        row = raw_feature_row(series)
        self.assertEqual(list(row), FEATURE_COLUMNS)
        self.assertEqual(row["pack_V_mean"], 3.75)
        self.assertEqual(row["pack_V_sum"], 15)
        self.assertAlmostEqual(row["pack_V_std"], np.std([1, 2, 3, 9]))
        self.assertAlmostEqual(row["pack_V_skew"], skew([1, 2, 3, 9], bias=False))

    def test_labels_and_interpolation(self):
        cycles = np.array([1, 2, 3])
        capacity = np.array([40., np.nan, 35.])
        np.testing.assert_allclose(make_soh(cycles, capacity, "raw"), [0.8, np.nan, 0.7])
        smooth = make_soh(cycles, capacity, "zero-phase-kalman")
        self.assertAlmostEqual(smooth[1], (smooth[0] + smooth[2]) / 2)
        with self.assertRaisesRegex(ValueError, "至少"):
            make_soh(cycles, np.array([40., 0, np.nan]), "zero-phase-kalman")

    def test_build_aggregates_all_fragments_and_exports_representative(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw"
            raw.mkdir()
            write_sample(raw / "10.pkl", 1, 44, 3.8)
            representative = write_sample(raw / "2.pkl", 1, 40, 3.4)
            write_sample(raw / "11.pkl", 2, 0, 3.6)
            out = root / "output"
            report = build(2, raw, out, [18], "raw")
            frame = pd.read_csv(out / "dataset2/processed/vehicles/dataset2_car_18.csv")
            index = pd.read_csv(out / "dataset2/processed/index.csv")
            self.assertAlmostEqual(frame.pack_V_mean.iloc[0], 3.6)
            self.assertEqual(index.capacity_raw.iloc[0], 42)
            self.assertAlmostEqual(frame.soh.iloc[0], 0.84)
            self.assertTrue(np.isnan(frame.soh.iloc[1]))
            self.assertEqual(index.source_pkl_id.iloc[0], 2)
            with np.load(out / "dataset2/cycles/dataset2_car_18/cycle_000001.npz") as z:
                np.testing.assert_allclose(z["mean_v"], representative[:, 0])
                np.testing.assert_array_equal(z["time"], representative[:, 7])
                np.testing.assert_array_equal(z["soc"], representative[:, 2])
                self.assertTrue(z["mask"].all())
            self.assertEqual(report["selected_segments"], 3)
            self.assertEqual(report["cycles"], 2)
            self.assertEqual(report["raw_segment_availability"], 1)
            with self.assertRaises(FileExistsError):
                build(2, raw, out, [18], "raw")

    def test_output_protection(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp)
            for output in (PROJECT / "data/new", PROJECT / "scripts/online_dualpool_final/new", raw / "new"):
                with self.assertRaises(ValueError):
                    validate_output(raw.resolve(), output.resolve())


if __name__ == "__main__":
    unittest.main()
