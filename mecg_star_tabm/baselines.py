"""The evaluated logistic and CatBoost public baselines; same shared manifest."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ._baseline_estimators import fit_catboost, fit_logistic
from .data import FEATURES, TIMESTAMP, dump_json, verify_manifest
from .metrics import save_evaluation


def engineer_features(frame):
    ts = pd.to_datetime(frame[TIMESTAMP])
    result = pd.DataFrame(index=frame.index)
    result["__month"] = ts.dt.month.astype(np.float32)
    result["__day_of_year"] = ts.dt.dayofyear.astype(np.float32)
    result["__hour"] = (ts.dt.hour + ts.dt.minute / 60.0 + ts.dt.second / 3600.0).astype(np.float32)
    result["__month_sin"] = np.sin(2 * np.pi * (result["__month"] - 1) / 12)
    result["__month_cos"] = np.cos(2 * np.pi * (result["__month"] - 1) / 12)
    result["__doy_sin"] = np.sin(2 * np.pi * (result["__day_of_year"] - 1) / 365.25)
    result["__doy_cos"] = np.cos(2 * np.pi * (result["__day_of_year"] - 1) / 365.25)
    result["__hour_sin"] = np.sin(2 * np.pi * result["__hour"] / 24)
    result["__hour_cos"] = np.cos(2 * np.pi * result["__hour"] / 24)
    for name in FEATURES:
        result[name] = pd.to_numeric(frame[name], errors="raise").astype(np.float32)
    names = FEATURES + [
        "__month",
        "__day_of_year",
        "__hour",
        "__month_sin",
        "__month_cos",
        "__doy_sin",
        "__doy_cos",
        "__hour_sin",
        "__hour_cos",
    ]
    return result[names]


def train_baseline(args):
    frame, y, indices, _ = verify_manifest(
        args.data, args.split_dir, allow_synthetic=bool(getattr(args, "_synthetic", False))
    )
    features = engineer_features(frame)
    train, early = indices["train"], indices["early_validation"]
    cfg = Namespace(
        seed=args.seed,
        category_min_frequency=5,
        logistic_c=1.0,
        logistic_max_iter=3000,
        logistic_tol=1e-3,
        task_type="GPU" if args.device.startswith("cuda") else "CPU",
        devices=args.device.split(":", 1)[-1] if args.device.startswith("cuda") else "0",
        workers=-1,
        catboost_iterations=2000,
        catboost_depth=8,
        catboost_learning_rate=0.05,
        catboost_l2_leaf_reg=5.0,
        catboost_early_stopping_rounds=100,
        catboost_verbose=50,
    )
    if args.model == "logistic":
        model = fit_logistic(
            features.iloc[train],
            y[train],
            np.ones(len(train), dtype=np.float32),
            list(features),
            [],
            cfg,
        )
        joblib.dump(model, Path(args.output_dir) / "model.joblib")
    else:
        model, _, trace = fit_catboost(
            features.iloc[train],
            y[train],
            np.ones(len(train), dtype=np.float32),
            features.iloc[early],
            y[early],
            [],
            cfg,
        )
        model.save_model(str(Path(args.output_dir) / "model.cbm"))
        pd.DataFrame(trace).to_csv(
            Path(args.output_dir) / "catboost_iteration_selection.csv", index=False
        )
    import shutil

    shutil.copy2(
        Path(args.split_dir) / "split_manifest.csv", Path(args.output_dir) / "split_manifest.csv"
    )
    summaries = {}
    for split, name in (
        ("early_validation", "early_stopping_validation"),
        ("calibration", "validation"),
        ("test", "test"),
    ):
        if split == "test" and args.run_mode != "final":
            continue
        idx = indices[split]
        probability = np.asarray(model.predict_proba(features.iloc[idx])[:, 1], dtype=np.float64)
        predictions = pd.DataFrame(
            {"source_index": idx, "target": y[idx], "raw_probability": probability}
        )
        predictions.to_csv(
            Path(args.output_dir) / f"{name}_predictions.csv", index=False, encoding="utf-8-sig"
        )
        summaries[split] = save_evaluation(
            predictions, args.output_dir, f"{name}_ranking", evaluation_set=split
        )
    dump_json(
        Path(args.output_dir) / "metrics.json",
        {
            "model": args.model,
            "run_mode": args.run_mode,
            "features": list(features),
            "split_metrics": summaries,
            "estimator_configuration": vars(cfg),
        },
    )
    return summaries
