"""Exact-budget metrics, explicit threshold metrics, and aligned seed averaging."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from .data import dump_json

BUDGETS = (0.001, 0.005, 0.01, 0.02, 0.05, 0.10)


def counts(target, selected):
    y = np.asarray(target)
    selected = np.asarray(selected, dtype=bool)
    tp = int(np.sum((y == 1) & selected))
    fp = int(np.sum((y == 0) & selected))
    fn = int(np.sum(y == 1)) - tp
    tn = int(np.sum(y == 0)) - fp
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "precision": precision,
        "recall": recall,
        "sensitivity": recall,
        "specificity": specificity,
        "npv": tn / max(tn + fn, 1),
        "f1": 2 * tp / max(2 * tp + fp + fn, 1),
        "accuracy": (tn + tp) / len(y),
        "balanced_accuracy": (recall + specificity) / 2,
        "false_positive_rate": fp / max(fp + tn, 1),
        "false_negative_rate": fn / max(fn + tp, 1),
        "false_discovery_rate": fp / max(fp + tp, 1),
        "mcc": (tp * tn - fp * fn) / max(denominator, np.finfo(float).eps),
    }


def evaluate(target, score, source_index):
    y = np.asarray(target)
    score = np.asarray(score, dtype=np.float64)
    ids = np.asarray(source_index)
    if y.ndim != 1 or score.shape != y.shape or ids.shape != y.shape or len(y) == 0:
        raise ValueError("Labels, scores and identities must be aligned nonempty vectors")
    if not np.isin(y, [0, 1]).all() or np.unique(y).size != 2:
        raise ValueError("Both binary classes are required")
    if (
        np.unique(ids).size != len(ids)
        or not np.isfinite(score).all()
        or np.any((score < 0) | (score > 1))
    ):
        raise ValueError("Unique identities and finite probabilities in [0,1] are required")
    order = np.lexsort((ids, -score))
    ap = float(average_precision_score(y, score))
    prev = float(y.mean())
    result = {
        "rows": len(y),
        "positives": int(y.sum()),
        "prevalence": prev,
        "average_precision": ap,
        "auprc": ap,
        "lift": ap / prev,
        "roc_auc": float(roc_auc_score(y, score)),
        "brier_score": float(brier_score_loss(y, score)),
        "score_source": "raw_probability",
        "tie_break": "ascending source_index",
        "topk": {},
        "threshold_0.5": counts(y, score >= 0.5),
    }
    for budget in BUDGETS:
        k = min(len(y), max(1, math.ceil(budget * len(y))))
        selected = np.zeros(len(y), dtype=bool)
        selected[order[:k]] = True
        result["topk"][f"top_{budget * 100:g}pct"] = {
            "budget": budget,
            "selected": k,
            **counts(y, selected),
        }
    return result


def read_predictions(path):
    frame = pd.read_csv(path, float_precision="round_trip")
    if not {"source_index", "target", "raw_probability"}.issubset(frame):
        raise ValueError("Prediction CSV needs source_index, target and raw_probability")
    if frame.source_index.duplicated().any():
        raise ValueError("Duplicate prediction identities")
    return frame.sort_values("source_index", kind="stable").reset_index(drop=True)


def save_evaluation(frame, output, prefix, **metadata):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    m = {
        **metadata,
        **evaluate(
            frame.target.to_numpy(), frame.raw_probability.to_numpy(), frame.source_index.to_numpy()
        ),
    }
    dump_json(output / f"{prefix}_metrics.json", m)
    rows = []
    for name, values in [*m["topk"].items(), ("threshold_0.5", m["threshold_0.5"])]:
        rows.append(
            {
                "operating_point": name,
                "auprc": m["auprc"],
                "roc_auc": m["roc_auc"],
                **{k: v for k, v in values.items() if k != "confusion_matrix"},
            }
        )
    pd.DataFrame(rows).to_csv(
        output / f"{prefix}_complete_metrics_and_confusion_matrices.csv", index=False
    )
    return m


def print_evaluation(m, name):
    print(
        f"{name}: rows={m['rows']:,}, positives={m['positives']}, AP={m['auprc']:.6f}, AUROC={m['roc_auc']:.6f}",
        flush=True,
    )
    for label in ("top_1pct", "top_5pct", "top_10pct"):
        c = m["topk"][label]
        print(
            f"  {label}: TP={c['tp']} FP={c['fp']} TN={c['tn']} FN={c['fn']} "
            f"P={c['precision']:.6f} R={c['recall']:.6f} F1={c['f1']:.6f} Acc={c['accuracy']:.6f}",
            flush=True,
        )


def evaluate_artifact(directory, run_mode):
    directory = Path(directory)
    for split, filename in (
        ("early", "early_stopping_validation_predictions.csv"),
        ("validation", "validation_predictions.csv"),
        ("test", "test_predictions.csv"),
    ):
        if split == "test" and run_mode != "final":
            continue
        frame = read_predictions(directory / filename)
        m = save_evaluation(frame, directory, f"{split}_ranking", evaluation_set=split)
        if split != "early":
            print_evaluation(m, f"{split}/raw fixed-budget ranking")


def ensemble(directories, output, split="test"):
    directories = [Path(p) for p in directories]
    if not directories or len({p.resolve() for p in directories}) != len(directories):
        raise ValueError("At least one distinct completed artifact is required")
    filename = {"test": "test_predictions.csv", "validation": "validation_predictions.csv"}[split]
    canonical = None
    probabilities = []
    for directory in directories:
        frame = read_predictions(directory / filename)
        if canonical is None:
            canonical = frame[["source_index", "target"]].copy()
        elif not canonical.equals(frame[["source_index", "target"]]):
            raise ValueError("Ensemble identities or targets differ; never average unmatched rows")
        probabilities.append(frame.raw_probability.to_numpy(dtype=np.float64))
    stack = np.vstack(probabilities)
    canonical["raw_probability"] = stack.mean(axis=0)
    canonical["seed_std"] = stack.std(axis=0)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    canonical.to_csv(
        output / f"ensemble_{split}_predictions.csv", index=False, encoding="utf-8-sig"
    )
    m = save_evaluation(
        canonical,
        output,
        f"ensemble_{split}",
        evaluation_set=split,
        n_models=len(directories),
        predictor="arithmetic mean of seed probabilities",
    )
    print_evaluation(m, f"{len(directories)}-seed ensemble/{split}")
    return m
