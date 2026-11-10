"""Fast protocol tests; the CLI smoke command separately checks model fitting."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

from mecg_star_tabm.data import FEATURES, TARGET, TIMESTAMP, create_split_manifest
from reproduce_ulb import ARMS, RECIPE, average, freeze_rules, load_data, report


class ProtocolTests(unittest.TestCase):
    def test_calibration_caps_and_confusion_counts(self):
        raw = np.linspace(0.01, 0.99, 10_000)
        y = (np.arange(len(raw)) % 10 == 0).astype(np.int8)
        raw = np.clip(raw + y * 0.01, 0, 1)
        f = pd.DataFrame(dict(source_index=np.arange(len(raw)), target=y, raw_probability=raw))
        platt, rules = freeze_rules(f)
        probs = platt.predict(raw)
        for cap in (0.001, 0.005):
            pred = probs >= rules[f"calibration_FPR_le_{cap:g}"]
            assert pred[y == 0].mean() <= cap
            row = report(y, raw, {"cap": pred})[0]
            self.assertEqual(row["tp"]+row["fp"]+row["tn"]+row["fn"], len(y))

    def test_seed_misalignment_fails(self):
        f = pd.DataFrame(dict(source_index=[1, 2], target=[0, 1], raw_probability=[0.1, 0.8]))
        with self.assertRaises(ValueError):
            average([f, f.iloc[::-1].reset_index(drop=True)])

    def test_manifest_reuse_and_change_detection(self):
        rng = np.random.default_rng(7)
        f = pd.DataFrame(rng.normal(size=(600, len(FEATURES))), columns=FEATURES)
        f[TIMESTAMP] = "2013-09-01"
        f[TARGET] = (np.arange(len(f)) % 5 == 0).astype(int)
        with TemporaryDirectory() as temp:
            root = Path(temp)
            csv, split = root / "test.csv", root / "split"
            f.to_csv(csv, index=False)
            create_split_manifest(csv, split, allow_synthetic=True)
            meta = load_data(csv, split, synthetic=True)[-1]
            create_split_manifest(csv, split, allow_synthetic=True)
            self.assertEqual(load_data(csv, split, synthetic=True)[-1], meta)
            f.iloc[::-1].to_csv(csv, index=False)
            with self.assertRaises(ValueError):
                load_data(csv, split, synthetic=True)

    def test_frozen_controls(self):
        self.assertEqual(ARMS["nptrbr"], (False, True, 1.0))
        self.assertEqual(ARMS["wbce_w2"], (False, False, 2.0))
        self.assertEqual(RECIPE["members"], 32)
        self.assertFalse(RECIPE["amp"])
        self.assertEqual(RECIPE["nptrbr_lambda"], 0.02)


if __name__ == "__main__":
    unittest.main()
