#!/usr/bin/env python3
"""Re-evaluate saved predictions; never train or choose thresholds on test labels.

All input prediction files remain local. Outputs contain aggregate statistics only.
Three prespecified operating points are reported without selecting a winner:
calibration-budget-1%-frozen, calibrated probability 0.5, and calibration-max-F1.
The last two are supplementary analyses of an already inspected test set.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

VERSION = "1.0.0"
RULES = ("calibration_budget_1pct_frozen", "probability_0p5", "calibration_max_f1")


def read_predictions(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    required = {"source_index", "target", "raw_probability"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{Path(path).name}: missing columns {sorted(missing)}")
    frame = frame.loc[:, [c for c in ("source_index", "target", "raw_probability", "timestamp") if c in frame]]
    source = pd.to_numeric(frame["source_index"], errors="raise")
    if source.isna().any() or not np.isfinite(source).all() or (source != np.floor(source)).any():
        raise ValueError("source_index must contain finite integers")
    frame["source_index"] = source.astype(np.int64)
    if frame["source_index"].duplicated().any():
        raise ValueError("Duplicate source_index; refusing ambiguous alignment")
    target = pd.to_numeric(frame["target"], errors="raise")
    if not target.isin([0, 1]).all():
        raise ValueError("Target must be binary 0/1")
    frame["target"] = target.astype(np.int8)
    score = pd.to_numeric(frame["raw_probability"], errors="raise").to_numpy(float)
    if not np.isfinite(score).all() or np.any((score < 0) | (score > 1)):
        raise ValueError("raw_probability must be finite and in [0,1]")
    frame["raw_probability"] = score
    if len(frame) == 0:
        raise ValueError("Empty prediction file")
    return frame.sort_values("source_index", kind="stable").reset_index(drop=True)


def mean_predictions(paths: Sequence[str | Path]) -> pd.DataFrame:
    if not paths:
        raise ValueError("At least one prediction file is required")
    canonical = read_predictions(paths[0])
    scores = [canonical["raw_probability"].to_numpy(float)]
    for path in paths[1:]:
        frame = read_predictions(path)
        assert_aligned(canonical, frame)
        scores.append(frame["raw_probability"].to_numpy(float))
    canonical["raw_probability"] = np.mean(scores, axis=0)
    return canonical


def assert_aligned(left: pd.DataFrame, right: pd.DataFrame) -> None:
    for column in ("source_index", "target"):
        if not np.array_equal(left[column].to_numpy(), right[column].to_numpy()):
            raise ValueError(f"Models/seeds do not have identical {column}; no paired comparison")
    if "timestamp" in left and "timestamp" in right:
        a = pd.to_datetime(left["timestamp"], errors="raise").to_numpy()
        b = pd.to_datetime(right["timestamp"], errors="raise").to_numpy()
        if not np.array_equal(a, b):
            raise ValueError("Timestamps disagree at matched source_index")


def split_fingerprint(frame: pd.DataFrame) -> str:
    source = np.sort(frame["source_index"].to_numpy(dtype="<i8"))
    return hashlib.sha256(source.tobytes()).hexdigest()


def classification_metrics(y: np.ndarray, prediction: np.ndarray) -> dict:
    y = np.asarray(y, dtype=np.int8)
    prediction = np.asarray(prediction, dtype=bool)
    tp = int(np.sum((y == 1) & prediction))
    fp = int(np.sum((y == 0) & prediction))
    fn = int(np.sum((y == 1) & ~prediction))
    tn = int(np.sum((y == 0) & ~prediction))
    div = lambda a, b: float(a / b) if b else 0.0
    precision, recall, specificity = div(tp, tp + fp), div(tp, tp + fn), div(tn, tn + fp)
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {
        "rows": len(y), "positives": int(y.sum()), "prevalence": float(y.mean()),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": precision, "recall": recall, "sensitivity": recall,
        "specificity": specificity, "npv": div(tn, tn + fn),
        "f1": div(2 * tp, 2 * tp + fp + fn), "accuracy": div(tp + tn, len(y)),
        "balanced_accuracy": 0.5 * (recall + specificity),
        "false_positive_rate": div(fp, fp + tn), "false_negative_rate": div(fn, fn + tp),
        "mcc": div(tp * tn - fp * fn, denominator),
        "predicted_positives": tp + fp, "alert_fraction": div(tp + fp, len(y)),
    }


class PlattCalibration:
    def fit(self, score: np.ndarray, target: np.ndarray) -> "PlattCalibration":
        if len(np.unique(target)) != 2:
            raise ValueError("Calibration requires both classes; no test-dependent fallback")
        self.model = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
        self.model.fit(self._logits(score), target)
        return self

    @staticmethod
    def _logits(score: np.ndarray) -> np.ndarray:
        score = np.clip(np.asarray(score, dtype=float), 1e-6, 1 - 1e-6)
        return (np.log(score) - np.log1p(-score)).reshape(-1, 1)

    def predict(self, score: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(self._logits(score))[:, 1]

    def metadata(self) -> dict:
        return {
            "method": "Platt_on_logit_of_mean_raw_probability",
            "coef": float(self.model.coef_[0, 0]), "intercept": float(self.model.intercept_[0]),
            "C": 1.0, "fit_split": "calibration_only",
        }


def freeze_thresholds(target: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    """No test inputs. Ties in max-F1 prefer the higher threshold/fewer alerts."""
    target = np.asarray(target, dtype=np.int8)
    probability = np.asarray(probability, dtype=float)
    if len(np.unique(target)) != 2:
        raise ValueError("Both classes required for threshold selection")
    order = np.argsort(-probability, kind="stable")
    p, y = probability[order], target[order]
    ends = np.r_[np.flatnonzero(p[:-1] != p[1:]), len(p) - 1]
    hits = np.cumsum(y, dtype=np.int64)[ends]
    selected = ends + 1
    f1 = 2 * hits / (selected + int(y.sum()))
    # Candidate thresholds are in descending order; first maximum wins.
    best = int(np.argmax(f1))
    k = max(1, min(len(p), math.ceil(len(p) * 0.01)))
    return {
        "calibration_budget_1pct_frozen": float(p[k - 1]),
        "probability_0p5": 0.5,
        "calibration_max_f1": float(p[ends[best]]),
    }


def evaluate_arrays(calibration: pd.DataFrame, test: pd.DataFrame, protocol: str, model: str) -> tuple[list[dict], list[dict], dict]:
    overlap = np.intersect1d(calibration["source_index"], test["source_index"])
    if overlap.size:
        raise ValueError("Calibration and test overlap; refusing evaluation")
    y_cal = calibration["target"].to_numpy(np.int8)
    cal_raw = calibration["raw_probability"].to_numpy(float)
    # Freeze all rules BEFORE accessing test labels.
    calibrator = PlattCalibration().fit(cal_raw, y_cal)
    cal_probability = calibrator.predict(cal_raw)
    thresholds = freeze_thresholds(y_cal, cal_probability)
    test_raw = test["raw_probability"].to_numpy(float)
    test_probability = calibrator.predict(test_raw)
    y = test["target"].to_numpy(np.int8)
    if len(np.unique(y)) != 2:
        raise ValueError("Test set must contain both classes for AP/AUROC")
    common = {
        "protocol": protocol, "model": model, "evaluation_set": "test",
        "auprc": float(average_precision_score(y, test_raw)),
        "roc_auc": float(roc_auc_score(y, test_raw)),
        "calibrated_auprc": float(average_precision_score(y, test_probability)),
        "calibrated_roc_auc": float(roc_auc_score(y, test_probability)),
        "brier": float(brier_score_loss(y, test_probability)),
        "test_split_sha256": split_fingerprint(test),
        "calibration_split_sha256": split_fingerprint(calibration),
    }
    common["auprc_lift"] = common["auprc"] / float(y.mean())
    rows = []
    for rule, threshold in thresholds.items():
        rows.append({**common, "operating_point": rule, "threshold": threshold,
                     **classification_metrics(y, test_probability >= threshold)})
    budgets = []
    order = np.argsort(-test_raw, kind="stable")
    for budget in (0.001, 0.005, 0.01, 0.02, 0.05, 0.1):
        k = max(1, min(len(y), math.ceil(len(y) * budget)))
        prediction = np.zeros(len(y), dtype=bool)
        prediction[order[:k]] = True
        budgets.append({**common, "operating_point": f"test_top_{100 * budget:g}pct",
                        "requested_fraction": budget, **classification_metrics(y, prediction)})
    metadata = {"version": VERSION, "thresholds": thresholds, "calibrator": calibrator.metadata(),
                "threshold_tie_policy": "higher_threshold_fewer_alerts",
                "thresholds_selected_on_test": False,
                "test_already_inspected": True,
                "interpretation": "supplementary_reanalysis_not_a_new_untouched_test",
                "raw_ranking_used_for_AP_and_AUROC": True}
    return rows, budgets, metadata


def evaluate_paths(calibration_paths: Sequence[str | Path], test_paths: Sequence[str | Path], protocol: str, model: str) -> tuple[list[dict], list[dict], dict]:
    if len(calibration_paths) != len(test_paths):
        raise ValueError("Calibration and test must use the same number of ensemble members")
    return evaluate_arrays(mean_predictions(calibration_paths), mean_predictions(test_paths), protocol, model)


def write_summary(rows: list[dict], budgets: list[dict], metadata: dict, output: str | Path) -> None:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output / "standard_test_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(budgets).to_csv(output / "budget_diagnostics.csv", index=False, encoding="utf-8-sig")
    (output / "evaluation_protocol.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-files", required=True, help="Comma-separated saved prediction CSVs")
    parser.add_argument("--test-files", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    rows, budgets, metadata = evaluate_paths(args.calibration_files.split(","), args.test_files.split(","), args.protocol, args.model)
    write_summary(rows, budgets, metadata, args.output_dir)
    print(pd.DataFrame(rows)[["model", "operating_point", "auprc", "precision", "recall", "f1", "accuracy", "tp", "fp", "tn", "fn"]].to_string(index=False))


if __name__ == "__main__":
    main()
