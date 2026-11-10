"""Portable public-only command line interface."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import logging
import sys
import tempfile
from importlib.resources import files
from pathlib import Path

from .data import create_split_manifest, dump_json, prepare_ulb, verify_manifest
from .metrics import ensemble, evaluate_artifact, read_predictions, save_evaluation

NEURAL = ("full", "mecg_bce", "no_mecg_nptrbr", "vanilla_tabm")
MODELS = NEURAL + ("logistic", "catboost")


def load_config(path=None):
    default = json.loads(
        files("mecg_star_tabm").joinpath("configs/ulb_final.json").read_text(encoding="utf-8")
    )
    if path:
        changes = json.loads(Path(path).read_text(encoding="utf-8"))
        if set(changes) - set(default):
            raise ValueError(
                f"Unknown public configuration keys: {sorted(set(changes) - set(default))}"
            )
        default.update(changes)
    for key in (
        "feature_modulation_scale",
        "feature_modulation_lambda_range",
        "modulation_member_rank",
    ):
        if default[key] != 0:
            raise ValueError("Input FAM is disabled in this frozen public release")
    if any(
        default[k] != value
        for k, value in (
            ("target_col", "__target__"),
            ("timestamp_col", "__timestamp__"),
            ("time_fourier_harmonics", 4),
            ("positive_values", "1"),
        )
    ):
        raise ValueError("Public schema and constant compatibility encodings are fixed")
    if default["mix_region"] != 0 or default["mix_prototype"] != 0:
        raise ValueError("The public steady-state sampler has no region/prototype quota")
    if default["ensemble_init"] not in {"legacy", "feature_normal", "feature_signs"}:
        raise ValueError("Unknown ensemble initialization")
    for key in (
        "members",
        "hidden_dim",
        "blocks",
        "batch_size",
        "prediction_batch_size",
        "patience",
        "epochs",
        "checkpoint_ensemble_top_k",
        "early_stop_smoothing",
        "nptrbr_ramp_epochs",
    ):
        if not isinstance(default[key], int) or default[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if default["member_negative_ratio"] <= 0 or default["nptrbr_lambda"] < 0:
        raise ValueError("Invalid sampling ratio or recall coefficient")
    if not 0 <= default["dropout"] < 1 or default["learning_rate"] <= 0:
        raise ValueError("Invalid dropout or learning rate")
    return default


def environment():
    result = {"python": sys.version.split()[0]}
    for package in ("torch", "numpy", "pandas", "scikit-learn", "joblib", "catboost"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


def run_training(options, *, synthetic=False):
    from ._engine import train_pipeline

    cfg = load_config(options.config)
    for name in ("epochs", "patience", "batch_size", "prediction_batch_size"):
        value = getattr(options, name, None)
        if value is not None:
            if value < 1:
                raise ValueError(f"{name} must be positive")
            cfg[name] = value
    cfg["model_variant"] = "vanilla_tabm" if options.model == "vanilla_tabm" else "mecg_star_tabm"
    cfg["mecg_enabled"] = options.model not in ("no_mecg_nptrbr", "vanilla_tabm")
    if options.model in ("mecg_bce", "vanilla_tabm"):
        cfg["nptrbr_lambda"] = 0.0
    if cfg["epochs"] < cfg["early_stop_warmup"]:
        raise ValueError(
            "epochs must be >= early_stop_warmup; use smoke for a tiny functional test"
        )
    if cfg["nptrbr_lambda"] > 0 and cfg["epochs"] <= cfg["nptrbr_warmup"]:
        raise ValueError("NP-TRBR needs epochs > nptrbr_warmup")
    _, _, _, meta = verify_manifest(options.data, options.split_dir, allow_synthetic=synthetic)
    args = argparse.Namespace(
        **cfg,
        data=str(Path(options.data).resolve()),
        output_dir=str(Path(options.output_dir).resolve()),
        split_dir=str(Path(options.split_dir).resolve()),
        experiment_label=f"public_ulb_{options.model}",
        run_mode=options.run_mode,
        seed=options.seed,
        device=options.device,
        _synthetic=synthetic,
    )
    request = {
        "model": options.model,
        "seed": options.seed,
        "run_mode": options.run_mode,
        "config": cfg,
        "data_sha256": meta["data_sha256"],
        "manifest_sha256": meta["manifest_sha256"],
        "device": options.device,
        "synthetic_fixture": synthetic,
    }
    from .training_state import source_fingerprint
    request["training_code_sha256"] = source_fingerprint()
    request_hash = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
    out = Path(args.output_dir)
    complete = out / "completed.json"
    args._resume_training = False
    if out.exists() and any(out.iterdir()):
        if options.resume and complete.exists():
            saved = json.loads(complete.read_text(encoding="utf-8"))
            prediction = out / (
                "test_predictions.csv"
                if options.run_mode == "final"
                else "validation_predictions.csv"
            )
            if saved["request_sha256"] == request_hash and prediction.exists():
                print(f"Skip matching completed public run: {out}", flush=True)
                return out
        previous_request = out/"run_request.json"
        epoch_state = out/"training_state.pt"
        if (options.resume and options.model in NEURAL and previous_request.is_file()
                and epoch_state.is_file() and not complete.exists()
                and json.loads(previous_request.read_text(encoding="utf-8"))["request_sha256"] == request_hash):
            args._resume_training = True
            print(f"Resume matching last completed epoch: {out}", flush=True)
        else:
            raise FileExistsError(
                f"Nonempty output: {out}. Resume requires a matching training_state.pt; "
                "old-version failures without epoch state cannot resume. Preserve the directory."
            )
    out.mkdir(parents=True, exist_ok=True)
    dump_json(out / "run_request.json", {**request, "request_sha256": request_hash})
    dump_json(out / "environment.json", environment())
    if options.model in NEURAL:
        train_pipeline(args)
    else:
        from .baselines import train_baseline

        args.model = options.model
        train_baseline(args)
    evaluate_artifact(out, options.run_mode)
    dump_json(
        complete,
        {
            "request_sha256": request_hash,
            "model": options.model,
            "seed": options.seed,
            "run_mode": options.run_mode,
        },
    )
    return out


def suite(options):
    models = [x.strip() for x in options.models.split(",") if x.strip()]
    seeds = [int(x.strip()) for x in options.seeds.split(",") if x.strip()]
    if not models or len(set(models)) != len(models) or set(models) - set(MODELS):
        raise ValueError(f"Choose distinct supported models: {MODELS}")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Choose distinct integer seeds")
    for model in models:
        directories = []
        for seed in seeds:
            opts = argparse.Namespace(**vars(options))
            opts.model, opts.seed = model, seed
            opts.output_dir = str(Path(options.output_root) / model / f"seed{seed}")
            directories.append(run_training(opts))
        output = Path(options.output_root) / model / "ensemble"
        ensemble(directories, output, "validation")
        if options.run_mode == "final":
            ensemble(directories, output, "test")
    summary = []
    for model in models:
        split = "test" if options.run_mode == "final" else "validation"
        m = json.loads(
            (
                Path(options.output_root) / model / "ensemble" / f"ensemble_{split}_metrics.json"
            ).read_text()
        )
        row = {
            "model": model,
            "split": split,
            "seeds": len(seeds),
            "rows": m["rows"],
            "positives": m["positives"],
            "auprc": m["auprc"],
            "roc_auc": m["roc_auc"],
        }
        for budget in (1, 5, 10):
            c = m["topk"][f"top_{budget}pct"]
            row.update(
                {
                    f"{k}_at_{budget}pct": c[k]
                    for k in ("tp", "fp", "tn", "fn", "precision", "recall", "f1", "accuracy")
                }
            )
        summary.append(row)
    import pandas as pd

    pd.DataFrame(summary).to_csv(
        Path(options.output_root) / "public_suite_summary.csv", index=False
    )


def smoke(output):
    """Exercise evidence gates, teacher, constraints, checkpointing and outputs."""
    import numpy as np
    import pandas as pd
    import torch
    from .data import FEATURES, GROUP, TARGET, TIMESTAMP

    torch.set_num_threads(1)
    out = Path(output)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Use an empty smoke output directory")
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    n = 2400
    x = rng.normal(size=(n, len(FEATURES)))
    # Regularly spaced labels keep every small validation segment evaluable.
    y = (np.arange(n) % 25 == 0).astype(np.int64)
    x[:, 0] += y * 1.5
    frame = pd.DataFrame(x, columns=FEATURES)
    frame[TARGET], frame[TIMESTAMP], frame[GROUP] = y, "2013-09-01", "public_fixture"
    data = out / "synthetic_ulb_shape.csv"
    frame.to_csv(data, index=False)
    create_split_manifest(data, out / "split", allow_synthetic=True)
    cfg = load_config()
    cfg.update(
        epochs=8,
        patience=8,
        hidden_dim=16,
        members=2,
        blocks=1,
        context_dim=8,
        member_adapter_rank=2,
        ple_bins=4,
        numeric_embedding_dim=2,
        batch_size=128,
        prediction_batch_size=256,
        early_stop_warmup=1,
        early_stop_smoothing=2,
        checkpoint_ensemble_top_k=2,
        nptrbr_warmup=1,
        nptrbr_ramp_epochs=1,
        nptrbr_pool_size=1000,
        mecg_warmup_epochs=1,
        mecg_ramp_epochs=1,
        mecg_evidence_max_rows=1000,
        hard_start_epoch=2,
        hard_pool_size=1000,
        hard_bank_size=100,
        hard_refresh=2,
        importance_max_rows=100,
    )
    config = out / "smoke_config.json"
    dump_json(config, cfg)
    options = argparse.Namespace(
        model="full",
        config=str(config),
        data=str(data),
        split_dir=str(out / "split"),
        output_dir=str(out / "full_seed42"),
        run_mode="final",
        seed=42,
        device="cpu",
        resume=False,
        epochs=None,
        patience=None,
        batch_size=None,
        prediction_batch_size=None,
    )
    run_training(options, synthetic=True)
    ensemble([options.output_dir], out / "ensemble", "test")
    hist = json.loads((Path(options.output_dir) / "training_history.json").read_text())
    if not any(h["nptrbr_lambda"] > 0 for h in hist) or not any(h["mecg_scale"] > 0 for h in hist):
        raise AssertionError("Smoke test did not exercise both contributions")
    print("Synthetic functional smoke passed; these are NOT benchmark performance results.")


def predict(options):
    import pickle
    from ._engine import choose_device, predict_checkpoint_ensemble, safe_torch_load
    from .data import load_training_frame
    from .model import ModelConfig, build_model

    run = Path(options.run_dir)
    device = choose_device(options.device)
    checkpoint = safe_torch_load(run / "model.pt", device)
    # Pickle is used ONLY for locally trained, trusted preprocessing artifacts.
    with (run / "preprocessor.pkl").open("rb") as handle:
        preprocessing = pickle.load(handle)
    request = argparse.Namespace(data=options.data, _synthetic=False)
    frame, y = load_training_frame(request)
    arrays = preprocessing.transform(frame)
    model = build_model(ModelConfig(**checkpoint["model_config"])).to(device)
    score, _, _ = predict_checkpoint_ensemble(
        model, checkpoint["checkpoint_state_dicts"], arrays, device, options.prediction_batch_size
    )
    import pandas as pd

    result = pd.DataFrame(
        {"source_index": arrays.source_index, "target": y, "raw_probability": score}
    )
    target = Path(options.output)
    if target.exists():
        raise FileExistsError("Prediction output already exists")
    target.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(target, index=False)


def build_parser():
    parser = argparse.ArgumentParser(description="Public-only ULB MECG-STaR-TabM reproduction")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser(
        "prepare", help="Download/prepare public ULB and freeze its split"
    )
    prepare.add_argument("--input", help="Optional canonical public creditcard.csv")
    prepare.add_argument("--data", default="data/ulb_prepared.csv")
    prepare.add_argument("--split-dir", default="artifacts/ulb_split")
    split = commands.add_parser(
        "split", help="Freeze splits for an existing prepared PUBLIC ULB CSV"
    )
    split.add_argument("--data", required=True)
    split.add_argument("--split-dir", default="artifacts/ulb_split")
    for name in ("train", "suite"):
        cmd = commands.add_parser(name)
        cmd.add_argument("--data", default="data/ulb_prepared.csv")
        cmd.add_argument("--split-dir", default="artifacts/ulb_split")
        cmd.add_argument("--config")
        cmd.add_argument("--device", default="cuda:0")
        cmd.add_argument("--run-mode", choices=["tune", "final"], default="tune")
        cmd.add_argument("--epochs", type=int)
        cmd.add_argument("--patience", type=int)
        cmd.add_argument("--batch-size", type=int)
        cmd.add_argument("--prediction-batch-size", type=int)
        cmd.add_argument(
            "--resume",
            action="store_true",
            help="Skip matching completed runs or restore the last completed epoch; not a partial minibatch",
        )
        if name == "train":
            cmd.add_argument("--model", choices=MODELS, default="full")
            cmd.add_argument("--seed", type=int, default=42)
            cmd.add_argument("--output-dir", required=True)
        else:
            cmd.add_argument("--models", default=",".join(MODELS))
            cmd.add_argument("--seeds", default="42,7,2027")
            cmd.add_argument("--output-root", default="runs/ulb")
    ens = commands.add_parser("ensemble")
    ens.add_argument("--artifact-dirs", nargs="+", required=True)
    ens.add_argument("--output-dir", required=True)
    ens.add_argument("--split", choices=["validation", "test"], default="test")
    ev = commands.add_parser("evaluate")
    ev.add_argument("--predictions", required=True)
    ev.add_argument("--output-dir", required=True)
    tiny = commands.add_parser("smoke")
    tiny.add_argument("--output-dir")
    pred = commands.add_parser(
        "predict", help="Score with trusted, locally produced neural artifacts"
    )
    pred.add_argument("--run-dir", required=True)
    pred.add_argument("--data", required=True)
    pred.add_argument("--output", required=True)
    pred.add_argument("--device", default="cuda:0")
    pred.add_argument("--prediction-batch-size", type=int, default=4096)
    return parser


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S"
    )
    args = build_parser().parse_args()
    if args.command == "prepare":
        prepare_ulb(args.data, args.input)
        create_split_manifest(args.data, args.split_dir)
    elif args.command == "split":
        create_split_manifest(args.data, args.split_dir)
    elif args.command == "train":
        run_training(args)
    elif args.command == "suite":
        suite(args)
    elif args.command == "ensemble":
        ensemble(args.artifact_dirs, args.output_dir, args.split)
    elif args.command == "evaluate":
        save_evaluation(read_predictions(args.predictions), args.output_dir, "evaluation")
    elif args.command == "predict":
        predict(args)
    elif args.output_dir:
        smoke(args.output_dir)
    else:
        with tempfile.TemporaryDirectory(prefix="mecg_public_smoke_") as folder:
            smoke(folder)


if __name__ == "__main__":
    main()
