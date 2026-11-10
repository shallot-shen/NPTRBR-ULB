#!/usr/bin/env python3
"""Public-only ULB baselines, using an EXISTING split manifest. No private imports.

Official architectures: tabm.TabM and rtdl_revisiting_models.FTTransformer.
AP means sklearn average_precision_score, NOT trapezoidal PR area.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import random
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

VERSION = "1.0.0"
FEATURES = [f"V{i}" for i in range(1, 29)] + ["Amount"]
MODELS = ("random_forest", "xgboost", "lightgbm", "ft_transformer", "official_tabm")
BUDGETS = (0.001, 0.005, 0.01, 0.02, 0.05, 0.10)


def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    os.replace(temporary, path)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def versions():
    result = {"runner": VERSION}
    for package in (
        "numpy",
        "pandas",
        "scikit-learn",
        "torch",
        "xgboost",
        "xgboost-cpu",
        "lightgbm",
        "tabm",
        "rtdl-revisiting-models",
        "rtdl-num-embeddings",
    ):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


def load_data(data_path, reference_dir):
    """Read only the known public schema; source_index is positional after prepare-ulb."""
    data_path, reference_dir = Path(data_path), Path(reference_dir)
    header = pd.read_csv(data_path, nrows=0).columns
    missing = set(FEATURES + ["__target__"]) - set(header)
    if missing:
        raise ValueError(f"Not the prepared ULB public schema. Missing: {sorted(missing)}")
    frame = pd.read_csv(data_path, usecols=FEATURES + ["__target__"])
    y = pd.to_numeric(frame["__target__"], errors="raise").to_numpy()
    if not np.isin(y, [0, 1]).all():
        raise ValueError("__target__ must contain only 0 and 1")
    x = frame[FEATURES].apply(pd.to_numeric, errors="raise").replace([np.inf, -np.inf], np.nan)
    manifest_path = reference_dir / "split_manifest.csv"
    manifest = pd.read_csv(manifest_path)
    if not {"source_index", "split"}.issubset(manifest):
        raise ValueError("Reference manifest needs source_index and split")
    numeric_ids = pd.to_numeric(manifest["source_index"], errors="raise").to_numpy()
    if not np.isfinite(numeric_ids).all() or not np.equal(numeric_ids, np.floor(numeric_ids)).all():
        raise ValueError("Non-integer source_index")
    manifest["source_index"] = numeric_ids.astype(np.int64)
    if len(manifest) != len(frame) or manifest["source_index"].duplicated().any():
        raise ValueError("Reference manifest must cover each prepared row exactly once")
    manifest = manifest.sort_values("source_index", kind="stable").reset_index(drop=True)
    if not np.array_equal(manifest["source_index"], np.arange(len(frame))):
        raise ValueError("Reference source_index does not match prepared CSV row positions")
    manifest["split"] = manifest["split"].replace({"locked_test": "test"})
    if set(manifest["split"]) - {"train", "early_validation", "calibration", "test", "unused"}:
        raise ValueError("Unrecognized split names")
    indices = {
        name: np.flatnonzero(manifest["split"].to_numpy() == name)
        for name in ("train", "early_validation", "calibration", "test")
    }
    for name, idx in indices.items():
        if not len(idx):
            raise ValueError(f"Empty {name} split")
        if name != "test" and np.unique(y[idx]).size != 2:
            raise ValueError(f"{name} must have both classes")
    # A fingerprinted manifest alone cannot prove that a CSV was not reordered.
    # Verify identities/targets against the existing CALIBRATION predictions too.
    ref_predictions = reference_dir / "validation_predictions.csv"
    if not ref_predictions.exists():
        raise FileNotFoundError(f"Need existing public calibration predictions: {ref_predictions}")
    ref = pd.read_csv(ref_predictions, usecols=["source_index", "target"])
    ref = ref.sort_values("source_index", kind="stable")
    if not np.array_equal(ref["source_index"], indices["calibration"]):
        raise ValueError("Calibration identity mismatch with existing public reference")
    if not np.array_equal(ref["target"], y[indices["calibration"]]):
        raise ValueError("Target mismatch: wrong/reordered public CSV")
    evidence = {
        "data_sha256": sha256(data_path),
        "manifest_sha256": sha256(manifest_path),
        "features": FEATURES,
        "reference_dir": str(reference_dir.resolve()),
        "split_rows": {k: int(len(v)) for k, v in indices.items()},
        "ordering_caveat": "Reuse prepared order; missing Time is NOT verified chronology.",
    }
    return x, y.astype(np.int64), indices, evidence


def counts(y, pred):
    tp = int(np.sum((y == 1) & pred))
    fp = int(np.sum((y == 0) & pred))
    fn = int(y.sum()) - tp
    tn = len(y) - tp - fp - fn
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
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
        "accuracy": (tp + tn) / len(y),
        "balanced_accuracy": (recall + specificity) / 2,
        "false_positive_rate": fp / max(fp + tn, 1),
        "false_negative_rate": fn / max(fn + tp, 1),
        "false_discovery_rate": fp / max(fp + tp, 1),
        "mcc": (tp * tn - fp * fn)
        / max(math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)), 1),
    }


def evaluate(y, score, ids):
    if not np.isfinite(score).all() or np.any((score < 0) | (score > 1)):
        raise ValueError("Invalid probability predictions")
    if np.unique(y).size != 2:
        raise ValueError("Evaluation requires both classes")
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
        "topk": {},
    }
    for b in BUDGETS:
        k = min(len(y), max(1, math.ceil(len(y) * b)))
        pred = np.zeros(len(y), dtype=bool)
        pred[order[:k]] = True
        result["topk"][f"top_{b * 100:g}pct"] = {"budget": b, "selected": k, **counts(y, pred)}
    result["threshold_0.5"] = counts(y, score >= 0.5)
    return result


def export_predictions(out, split, y, score, ids, metadata):
    out.mkdir(parents=True, exist_ok=True)
    metrics = {**metadata, "evaluation_set": split, **evaluate(y, score, ids)}
    name = {
        "calibration": "validation",
        "early_validation": "early_stopping_validation",
        "test": "test",
    }[split]
    pred = np.zeros(len(y), dtype=np.int8)
    pred[np.lexsort((ids, -score))[: math.ceil(len(y) * 0.01)]] = 1
    pd.DataFrame(
        {"source_index": ids, "target": y, "raw_probability": score, "prediction_top1": pred}
    ).to_csv(out / f"{name}_predictions.csv", index=False, encoding="utf-8-sig")
    dump(out / f"{name}_metrics.json", metrics)
    records = []
    for label, values in list(metrics["topk"].items()) + [
        ("threshold_0.5", metrics["threshold_0.5"])
    ]:
        records.append(
            {
                "evaluation_set": split,
                "operating_point": label,
                **{
                    k: metrics[k]
                    for k in ("rows", "positives", "prevalence", "auprc", "lift", "roc_auc")
                },
                **{k: v for k, v in values.items() if k != "confusion_matrix"},
            }
        )
    pd.DataFrame(records).to_csv(
        out / f"{name}_complete_metrics_and_confusion_matrices.csv",
        index=False,
        encoding="utf-8-sig",
    )
    c = metrics["topk"]["top_1pct"]
    print(
        f"{split}: AP={metrics['auprc']:.6f} ROC-AUC={metrics['roc_auc']:.6f} "
        f"Top1% [[TN={c['tn']},FP={c['fp']}],[FN={c['fn']},TP={c['tp']}]] "
        f"P={c['precision']:.6f} R={c['recall']:.6f} F1={c['f1']:.6f}",
        flush=True,
    )
    return metrics


def make_neural(name, x_train, spec, bins=None):
    import torch

    if name == "official_tabm":
        from tabm import TabM
        from rtdl_num_embeddings import compute_bins, PiecewiseLinearEmbeddings

        if bins is None:
            bins = compute_bins(torch.as_tensor(x_train), n_bins=spec["ple_bins"])
        embedding = PiecewiseLinearEmbeddings(bins, d_embedding=8, activation=False, version="B")
        model = TabM.make(
            n_num_features=len(FEATURES),
            d_out=1,
            num_embeddings=embedding,
            n_blocks=2,
            d_block=512,
            dropout=0.1,
            k=32,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.0003)
    else:
        from rtdl_revisiting_models import FTTransformer

        model = FTTransformer(
            n_cont_features=len(FEATURES),
            cat_cardinalities=[],
            d_out=1,
            n_blocks=3,
            d_block=192,
            attention_n_heads=8,
            attention_dropout=0.2,
            ffn_d_hidden=None,
            ffn_d_hidden_multiplier=4 / 3,
            ffn_dropout=0.1,
            residual_dropout=0.0,
        )
        optimizer = torch.optim.AdamW(model.make_parameter_groups(), lr=1e-4, weight_decay=1e-5)
    return model, optimizer, bins


def torch_predict(model, x, name, device, batch):
    import torch

    model.eval()
    output = []
    with torch.no_grad():
        for start in range(0, len(x), batch):
            xb = torch.as_tensor(x[start : start + batch], device=device)
            logits = model(xb) if name == "official_tabm" else model(xb, None)
            prob = logits.sigmoid()
            if name == "official_tabm":
                prob = prob.mean(dim=1)  # average PROBABILITIES, not logits
            output.append(prob.flatten().cpu().numpy())
    return np.concatenate(output).astype(np.float64)


def train_neural(out, name, x, y, ix, spec, positive_weight):
    import torch

    seed = spec["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(spec["threads"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device(spec["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable. Choose --device cpu explicitly.")
    model, optimizer, bins = make_neural(name, x[ix["train"]], spec)
    model.to(device)
    pos_weight = torch.tensor(positive_weight, device=device)
    best_ap = -1.0
    best_epoch = 0
    start = 1
    history = []
    # Epoch-boundary continuation; never load an untrusted checkpoint.
    last_path = out / "last.pt"
    if last_path.exists():
        state = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        best_ap = state["best_ap"]
        best_epoch = state["best_epoch"]
        start = state["epoch"] + 1
        history = state["history"]
        torch.set_rng_state(state["torch_rng"].cpu())
        np.random.set_state(state["numpy_rng"])
        random.setstate(state["python_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda_rng"]])
        print(f"Resume after epoch {start - 1}", flush=True)
    print(
        f"Official architecture={name}; parameters={sum(p.numel() for p in model.parameters()):,}",
        flush=True,
    )
    for epoch in range(start, spec["epochs"] + 1):
        if epoch - best_epoch > spec["patience"]:
            break
        tick = time.monotonic()
        model.train()
        total_loss = 0.0
        order = np.random.permutation(ix["train"])
        for begin in range(0, len(order), spec["batch_size"]):
            ids = order[begin : begin + spec["batch_size"]]
            xb = torch.as_tensor(x[ids], device=device)
            yb = torch.as_tensor(y[ids], dtype=torch.float32, device=device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb) if name == "official_tabm" else model(xb, None)
            target = yb.reshape(-1, *([1] * (logits.ndim - 1))).expand_as(logits)
            # A separate BCE for every TabM member; no collapse before the loss.
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, target, pos_weight=pos_weight
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(ids)
        prob = torch_predict(
            model, x[ix["early_validation"]], name, device, spec["prediction_batch_size"]
        )
        ap = float(average_precision_score(y[ix["early_validation"]], prob))
        improved = ap > best_ap + 1e-12
        if improved:
            best_ap, best_epoch = ap, epoch
            torch.save({"model": model.state_dict(), "bins": bins}, out / "best.pt.tmp")
            os.replace(out / "best.pt.tmp", out / "best.pt")
        record = {
            "epoch": epoch,
            "train_loss": total_loss / len(order),
            "early_validation_ap": ap,
            "seconds": time.monotonic() - tick,
            "best_epoch": best_epoch,
        }
        history.append(record)
        pd.DataFrame(history).to_csv(out / "training_history.csv", index=False)
        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_ap": best_ap,
            "best_epoch": best_epoch,
            "history": history,
            "torch_rng": torch.get_rng_state(),
            "numpy_rng": np.random.get_state(),
            "python_rng": random.getstate(),
            "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
        }
        torch.save(state, out / "last.pt.tmp")
        os.replace(out / "last.pt.tmp", last_path)
        print(
            f"Epoch {epoch:03d}: loss={record['train_loss']:.6f} EV_AP={ap:.6f} "
            f"best={best_epoch} seconds={record['seconds']:.1f}",
            flush=True,
        )
        if epoch - best_epoch >= spec["patience"]:
            break
    best = torch.load(out / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model"])
    return model, {
        "best_epoch": best_epoch,
        "best_early_validation_ap": best_ap,
        "checkpoint_policy": "single_best_early_validation_AP",
    }


def train_tree(out, name, x, y, ix, spec, positive_weight):
    xt, yt = x[ix["train"]], y[ix["train"]]
    xv, yv = x[ix["early_validation"]], y[ix["early_validation"]]
    seed = spec["seed"]
    threads = spec["threads"]
    if name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier

        model = RandomForestClassifier(
            n_estimators=spec["trees"],
            max_features="sqrt",
            min_samples_leaf=1,
            class_weight="balanced_subsample" if spec["class_weight"] == "balanced" else None,
            random_state=seed,
            n_jobs=threads,
        )
        model.fit(xt, yt)
        joblib.dump(model, out / "model.joblib")
        return model, {"best_trees": spec["trees"], "early_stopping": False}
    if name == "xgboost":
        import xgboost as xgb

        if spec["tree_device"].startswith("cuda") and not xgb.build_info().get("USE_CUDA", False):
            raise RuntimeError(
                "This XGBoost build lacks CUDA. Use --tree-device cpu in a NEW tagged run."
            )
        train = xgb.DMatrix(xt, label=yt)
        val = xgb.DMatrix(xv, label=yv)
        params = {
            "objective": "binary:logistic",
            "disable_default_eval_metric": 1,
            "tree_method": "hist",
            "device": spec["tree_device"],
            "max_depth": 6,
            "eta": 0.05,
            "min_child_weight": 1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "lambda": 1.0,
            "scale_pos_weight": positive_weight,
            "seed": seed,
            "nthread": threads,
        }
        history = {}
        model = xgb.train(
            params,
            train,
            num_boost_round=spec["boost_rounds"],
            evals=[(val, "early_validation")],
            custom_metric=lambda p, d: ("AP", float(average_precision_score(d.get_label(), p))),
            maximize=True,
            early_stopping_rounds=spec["tree_patience"],
            evals_result=history,
            verbose_eval=50,
        )
        model.save_model(out / "model.ubj")
        dump(out / "training_history.json", history)
        return model, {"best_trees": int(model.best_iteration) + 1, "tree_parameters": params}
    import lightgbm as lgb

    params = {
        "objective": "binary",
        "metric": "None",
        "verbosity": -1,
        "num_leaves": 31,
        "learning_rate": 0.05,
        "min_data_in_leaf": 20,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "scale_pos_weight": positive_weight,
        "seed": seed,
        "num_threads": threads,
        "device_type": "cpu",
        "deterministic": True,
        "force_col_wise": True,
    }
    train = lgb.Dataset(xt, label=yt)
    val = lgb.Dataset(xv, label=yv, reference=train)
    history = {}
    model = lgb.train(
        params,
        train,
        num_boost_round=spec["boost_rounds"],
        valid_sets=[val],
        valid_names=["early_validation"],
        feval=lambda p, d: ("AP", float(average_precision_score(d.get_label(), p)), True),
        callbacks=[
            lgb.early_stopping(spec["tree_patience"]),
            lgb.log_evaluation(50),
            lgb.record_evaluation(history),
        ],
    )
    model.save_model(str(out / "model.txt"), num_iteration=model.best_iteration)
    dump(out / "training_history.json", history)
    return model, {"best_trees": int(model.best_iteration), "tree_parameters": params}


def predict(model, x, name, spec, training_info):
    if name in ("official_tabm", "ft_transformer"):
        return torch_predict(model, x, name, spec["device"], spec["prediction_batch_size"])
    if name == "random_forest":
        return model.predict_proba(x)[:, 1]
    if name == "xgboost":
        import xgboost as xgb

        return model.predict(xgb.DMatrix(x), iteration_range=(0, training_info["best_trees"]))
    return model.predict(x, num_iteration=training_info["best_trees"])


def load_model(out, name, x, ix, spec):
    if name == "random_forest":
        return joblib.load(out / "model.joblib")
    if name == "xgboost":
        import xgboost as xgb

        model = xgb.Booster()
        model.load_model(out / "model.ubj")
        model.set_param({"device": spec["tree_device"], "nthread": spec["threads"]})
        return model
    if name == "lightgbm":
        import lightgbm as lgb

        return lgb.Booster(model_file=str(out / "model.txt"))
    import torch

    state = torch.load(out / "best.pt", map_location=spec["device"], weights_only=False)
    model, _, _ = make_neural(name, x[ix["train"]], spec, state["bins"])
    model.load_state_dict(state["model"])
    return model.to(spec["device"])


def run(args):
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    xraw, y, ix, evidence = load_data(args.data, args.reference_dir)
    spec = {
        k: getattr(args, k)
        for k in (
            "seed",
            "device",
            "threads",
            "tree_device",
            "epochs",
            "patience",
            "batch_size",
            "prediction_batch_size",
            "trees",
            "boost_rounds",
            "tree_patience",
            "ple_bins",
            "class_weight",
        )
    }
    signature = {
        "runner_version": VERSION,
        "model": args.model,
        "spec": spec,
        "data_sha256": evidence["data_sha256"],
        "manifest_sha256": evidence["manifest_sha256"],
    }
    config_path = out / "config.json"
    if config_path.exists():
        old = json.loads(config_path.read_text(encoding="utf-8"))
        if old["signature"] != signature:
            raise ValueError(
                "Existing directory has a different configuration/data/split. Use a NEW output directory."
            )
    else:
        if args.command == "test":
            raise FileNotFoundError("Train this configuration first")
        dump(
            config_path,
            {
                "signature": signature,
                "evidence": evidence,
                "versions": versions(),
                "test_used_for_selection": False,
                "preprocessing_fit": "train_only",
                "baseline_scope": "official architecture, local fixed preset; not published-score reproduction",
            },
        )
    complete = out / "training_complete.json"
    if args.command == "train" and complete.exists():
        print(f"Skip completed, matching signature: {out}", flush=True)
        return
    if args.command == "test" and not args.unlock_test:
        raise ValueError(
            "Final evaluation requires --unlock-test. No training is performed by test."
        )
    if args.command == "test" and not complete.exists():
        raise FileNotFoundError("No completed training artifact")
    if args.command == "test" and (out / "test_metrics.json").exists():
        print(f"Skip existing final evaluation: {out}", flush=True)
        return
    tick = time.monotonic()
    preprocessor_path = out / "preprocessor.joblib"
    if preprocessor_path.exists():
        preprocessor = joblib.load(preprocessor_path)
    else:
        if args.command == "test":
            raise FileNotFoundError(preprocessor_path)
        preprocessor = make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler()
        )
        preprocessor.fit(xraw.iloc[ix["train"]])
        joblib.dump(preprocessor, preprocessor_path)
    x = preprocessor.transform(xraw).astype(np.float32)
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite transformed features")
    if args.command == "train":
        for label in ("train", "early_validation", "calibration"):
            print(
                f"{label}: rows={len(ix[label]):,} positives={int(y[ix[label]].sum())}", flush=True
            )
        print(f"test LOCKED: rows={len(ix['test']):,}; features={len(FEATURES)}", flush=True)
        weight = (
            (len(ix["train"]) - y[ix["train"]].sum()) / y[ix["train"]].sum()
            if args.class_weight == "balanced"
            else 1.0
        )
        trainer = train_neural if args.model in ("official_tabm", "ft_transformer") else train_tree
        model, info = trainer(out, args.model, x, y, ix, spec, float(weight))
        dump(out / "training_info.json", info)
        for split in ("early_validation", "calibration"):
            ids = ix[split]
            score = predict(model, x[ids], args.model, spec, info)
            export_predictions(
                out, split, y[ids], score, ids, {"model": args.model, "seed": args.seed}
            )
        dump(complete, {"elapsed_seconds": time.monotonic() - tick, "signature": signature})
    else:
        info = json.loads((out / "training_info.json").read_text())
        model = load_model(out, args.model, x, ix, spec)
        ids = ix["test"]
        score = predict(model, x[ids], args.model, spec, info)
        export_predictions(
            out,
            "test",
            y[ids],
            score,
            ids,
            {
                "model": args.model,
                "seed": args.seed,
                "selection_policy": "frozen before supplementary test evaluation",
            },
        )


def ensemble(args):
    dirs = [Path(v) for v in args.artifact_dirs.split(",") if v]
    if not dirs or len(set(dirs)) != len(dirs):
        raise ValueError("Need nonempty distinct artifact directories")
    split = args.evaluation_set
    filename = "test_predictions.csv" if split == "test" else "validation_predictions.csv"
    if split == "test" and not args.unlock_test:
        raise ValueError("Test ensemble requires --unlock-test")
    canonical = None
    scores = []
    signatures = []
    for d in dirs:
        signature = json.loads((d / "config.json").read_text(encoding="utf-8"))["signature"]
        signatures.append(signature)
        f = (
            pd.read_csv(d / filename, float_precision="round_trip")
            .sort_values("source_index", kind="stable")
            .reset_index(drop=True)
        )
        if f["source_index"].duplicated().any():
            raise ValueError("Duplicate source_index")
        if canonical is None:
            canonical = f[["source_index", "target"]]
        elif not canonical.equals(f[["source_index", "target"]]):
            raise ValueError("Different prediction identities/labels")
        s = f["raw_probability"].to_numpy(float)
        if not np.isfinite(s).all() or np.any((s < 0) | (s > 1)):
            raise ValueError("Invalid seed probabilities")
        scores.append(s)
    for key in ("data_sha256", "manifest_sha256", "model"):
        if len({s[key] for s in signatures}) != 1:
            raise ValueError(f"Ensemble mismatch: {key}")
    if len({s["spec"]["seed"] for s in signatures}) != len(signatures):
        raise ValueError("Duplicate ensemble seeds")
    base = {k: v for k, v in signatures[0]["spec"].items() if k != "seed"}
    if any({k: v for k, v in s["spec"].items() if k != "seed"} != base for s in signatures):
        raise ValueError("Only seeds may differ within a baseline ensemble")
    out = Path(args.output_dir)
    metrics = export_predictions(
        out,
        split,
        canonical.target.to_numpy(),
        np.mean(scores, axis=0),
        canonical.source_index.to_numpy(),
        {
            "model": signatures[0]["model"],
            "n_models": len(dirs),
            "ensemble_policy": "equal_seed_probability_mean"
            if len(dirs) > 1
            else "single_seed_identity",
            "artifact_dirs": [str(d.resolve()) for d in dirs],
        },
    )
    # Same filenames as the existing project's paired statistical tool.
    stem = "test" if split == "test" else "validation"
    import shutil

    shutil.copyfile(out / f"{stem}_predictions.csv", out / f"ensemble_{stem}_predictions.csv")
    dump(out / f"ensemble_{stem}_metrics.json", metrics)


def collect(args):
    root = Path(args.runs_dir)
    summary = []
    complete = []
    for file in sorted(
        root.glob(f"public_ulb_extra_*_{args.tag}_ensemble/ensemble_test_metrics.json")
    ):
        m = json.loads(file.read_text(encoding="utf-8"))
        summary.append(
            {
                "model": m["model"],
                "tag": args.tag,
                "n_seeds": m["n_models"],
                "rows": m["rows"],
                "positives": m["positives"],
                "AUPRC": m["auprc"],
                "ROC_AUC": m["roc_auc"],
                "Recall1": m["topk"]["top_1pct"]["recall"],
                "Recall5": m["topk"]["top_5pct"]["recall"],
                "Recall10": m["topk"]["top_10pct"]["recall"],
            }
        )
        table = pd.read_csv(file.parent / "test_complete_metrics_and_confusion_matrices.csv")
        table.insert(0, "model", m["model"])
        complete.append(table)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary).to_csv(
        output / "public_extra_comparison.csv", index=False, encoding="utf-8-sig"
    )
    seed_rows = []
    for file in sorted(root.glob(f"public_ulb_extra_*_{args.tag}_seed*/test_metrics.json")):
        m = json.loads(file.read_text(encoding="utf-8"))
        seed_rows.append(
            {
                "model": m["model"],
                "seed": m["seed"],
                "AUPRC": m["auprc"],
                "ROC_AUC": m["roc_auc"],
                "Recall1": m["topk"]["top_1pct"]["recall"],
                "Recall5": m["topk"]["top_5pct"]["recall"],
                "Recall10": m["topk"]["top_10pct"]["recall"],
            }
        )
    if seed_rows:
        seed_table = pd.DataFrame(seed_rows)
        seed_table.to_csv(output / "public_extra_per_seed.csv", index=False, encoding="utf-8-sig")
        aggregate = seed_table.groupby("model").agg(
            {k: ["mean", "std"] for k in ("AUPRC", "ROC_AUC", "Recall1", "Recall5", "Recall10")}
        )
        aggregate.columns = ["_".join(col) for col in aggregate.columns]
        aggregate.to_csv(output / "public_extra_seed_mean_std.csv", encoding="utf-8-sig")
    if complete:
        pd.concat(complete, ignore_index=True).to_csv(
            output / "public_extra_all_confusion_matrices.csv", index=False, encoding="utf-8-sig"
        )
    import zipfile

    archive = output / "ULB_extra_baseline_results.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        # Public predictions enable paired tests; no models or enterprise data.
        for pattern in (
            f"public_ulb_extra_*_{args.tag}_*/*.json",
            f"public_ulb_extra_*_{args.tag}_*/*.csv",
        ):
            for f in sorted(root.glob(pattern)):
                z.write(f, str(f.relative_to(root)))
        for name in (
            "public_extra_comparison.csv",
            "public_extra_all_confusion_matrices.csv",
            "public_extra_per_seed.csv",
            "public_extra_seed_mean_std.csv",
            "paired_public_test_bootstrap.json",
        ):
            if (output / name).exists():
                z.write(output / name, name)
    print(pd.DataFrame(summary).to_string(index=False), flush=True)
    print(f"Saved: {archive}", flush=True)


def doctor(args):
    print(json.dumps(versions(), indent=2), flush=True)
    import torch

    torch.set_num_threads(2)
    print(f"torch CUDA available: {torch.cuda.is_available()}", flush=True)
    for name in args.models.split(","):
        if name in ("official_tabm", "ft_transformer"):
            x = np.random.default_rng(42).normal(size=(80, len(FEATURES))).astype(np.float32)
            model, _, _ = make_neural(name, x, {"ple_bins": 8})
            p = torch_predict(model, x[:3], name, "cpu", 2)
            assert p.shape == (3,) and np.isfinite(p).all()
        else:
            __import__(
                {"random_forest": "sklearn", "lightgbm": "lightgbm", "xgboost": "xgboost"}[name]
            )
        print(f"OK: {name}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    for command in ("train", "test"):
        q = sub.add_parser(command)
        q.add_argument("--data", required=True)
        q.add_argument("--reference-dir", required=True)
        q.add_argument("--output-dir", required=True)
        q.add_argument("--model", choices=MODELS, required=True)
        q.add_argument("--seed", type=int, default=42)
        q.add_argument("--device", default="cuda:0")
        q.add_argument("--threads", type=int, default=8)
        q.add_argument("--tree-device", default="cuda")
        q.add_argument("--epochs", type=int, default=100)
        q.add_argument("--patience", type=int, default=20)
        q.add_argument("--batch-size", type=int, default=512)
        q.add_argument("--prediction-batch-size", type=int, default=1024)
        q.add_argument("--trees", type=int, default=600)
        q.add_argument("--boost-rounds", type=int, default=3000)
        q.add_argument("--tree-patience", type=int, default=100)
        q.add_argument("--ple-bins", type=int, default=16)
        q.add_argument("--class-weight", choices=["none", "balanced"], default="none")
        q.add_argument("--unlock-test", action="store_true")
    q = sub.add_parser("check")
    q.add_argument("--data", required=True)
    q.add_argument("--reference-dir", required=True)
    q = sub.add_parser("doctor")
    q.add_argument("--models", default=",".join(MODELS))
    q = sub.add_parser("ensemble")
    q.add_argument("--artifact-dirs", required=True)
    q.add_argument("--output-dir", required=True)
    q.add_argument("--evaluation-set", choices=["calibration", "test"], required=True)
    q.add_argument("--unlock-test", action="store_true")
    q = sub.add_parser("collect")
    q.add_argument("--runs-dir", required=True)
    q.add_argument("--output-dir", required=True)
    q.add_argument("--tag", default="default")
    args = p.parse_args()
    if args.command in ("train", "test"):
        if (
            min(
                args.epochs,
                args.patience,
                args.batch_size,
                args.threads,
                args.trees,
                args.boost_rounds,
                args.tree_patience,
                args.prediction_batch_size,
            )
            < 1
        ):
            p.error("Counts must be positive")
        run(args)
    elif args.command == "check":
        _, _, _, evidence = load_data(args.data, args.reference_dir)
        print(json.dumps(evidence, indent=2), flush=True)
        print(
            "PASS: existing public manifest + calibration identities; no split regenerated.",
            flush=True,
        )
    elif args.command == "doctor":
        doctor(args)
    elif args.command == "ensemble":
        ensemble(args)
    else:
        collect(args)


if __name__ == "__main__":
    main()
