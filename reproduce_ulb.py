"""Public ULB-only reproduction of the manuscript's matched TabM experiments."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import pickle
import sys
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from filelock import FileLock
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from decision_metrics import PlattCalibration, classification_metrics, freeze_thresholds
from mecg_star_tabm.data import (
    EXPECTED_SPLIT_POSITIVES, EXPECTED_SPLIT_ROWS, FEATURES, TARGET, TIMESTAMP,
    create_split_manifest, prepare_ulb, sha256, verify_manifest,
)
from paired_stats import paired_bootstrap

ROOT = Path(__file__).resolve().parent
DEFAULT_SEEDS = (42, 7, 2027)
# Order matches the supplement, including the order of bootstrap inputs.
ARMS = {
    "bce": (False, False, 1.0),
    "mecg_bce": (True, False, 1.0),
    "nptrbr": (False, True, 1.0),
    "mecg_nptrbr": (True, True, 1.0),
    "wbce_w2": (False, False, 2.0),
}
FILES = ("model.pt", "preprocessor.pkl", "validation_predictions.csv", "metrics.json", "split_manifest.csv")
RECIPE_PATH = ROOT / "mecg_star_tabm" / "configs" / "ulb_final.json"
RECIPE = json.loads(RECIPE_PATH.read_text(encoding="utf-8"))
RECIPE.update(
    members=32, hidden_dim=128, blocks=3, ple_bins=16, numeric_embedding_dim=8,
    early_stop_warmup=5, grad_clip=5.0, threshold_strategy="f1", sparse_lambda=0.0, amp=False,
)


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame(rows).to_csv(tmp, index=False, encoding="utf-8-sig")
    os.replace(tmp, path)


def environment():
    keys = ("torch", "tabm", "rtdl-num-embeddings", "numpy", "pandas", "scikit-learn", "scipy")
    return {"python": sys.version.split()[0], **{k: importlib.metadata.version(k) for k in keys}}


def load_data(data, split_dir, *, synthetic=False):
    frame, y, indices, meta = verify_manifest(data, split_dir, allow_synthetic=synthetic)
    for name, ids in indices.items():
        if not synthetic and (len(ids) != EXPECTED_SPLIT_ROWS[name] or
                              int(y[ids].sum()) != EXPECTED_SPLIT_POSITIVES[name]):
            raise ValueError(f"The {name} partition differs from the frozen ULB protocol")
    # The split is ordered and contiguous; refuse any manifest that mixes the roles.
    if not np.array_equal(np.concatenate(list(indices.values())), np.arange(len(frame))):
        raise ValueError("The four public partitions must be contiguous in original source order")
    frame[TIMESTAMP] = pd.to_datetime(frame[TIMESTAMP])
    frame["_source_index"] = np.arange(len(frame), dtype=np.int64)
    return frame, y, indices, meta


def request(args, arm, seed, loaded, recipe):
    from mecg_star_tabm.training_state import source_fingerprint

    meta = loaded[3]
    val = {
        "dataset": "openml_1597", "data_sha256": meta["data_sha256"],
        "manifest_sha256": meta["manifest_sha256"], "versions": environment(),
        "code": {**source_fingerprint(), **{n: sha256(ROOT / n) for n in
                   ("reproduce_ulb.py", "official_model.py", "decision_metrics.py", "paired_stats.py")}},
        "recipe": recipe, "device": args.device, "seed": seed, "variant": arm,
        "gates": ARMS[arm][0], "nptrbr": ARMS[arm][1], "positive_loss_weight": ARMS[arm][2],
    }
    val["request_sha256"] = hashlib.sha256(json.dumps(val, sort_keys=True).encode()).hexdigest()
    return val


def run_dir(args, arm, seed):
    return args.output / arm / f"seed{seed}"


def check_fit(args, arm, seed, loaded, recipe):
    path = run_dir(args, arm, seed)
    if read(path / "run_request.json") != request(args, arm, seed, loaded, recipe):
        raise ValueError(f"Run signature mismatch: {path}. Use the original settings or a new output root.")
    hashes = read(path / "complete.json")
    if set(hashes) != set(FILES) or any(sha256(path / n) != value for n, value in hashes.items()):
        raise ValueError(f"Incomplete or modified artifacts: {path}")
    return path


def fit(args, arm, seed, loaded, recipe):
    from mecg_star_tabm import _engine as engine
    from mecg_star_tabm.preprocessing import PublicPreprocessor
    from official_model import InputView, OfficialTransfer

    frame, y, ix, _ = loaded
    out = run_dir(args, arm, seed)
    out.mkdir(parents=True, exist_ok=True)
    with FileLock(str(out / ".training.lock"), timeout=0):
        req = request(args, arm, seed, loaded, recipe)
        identity = out / "run_request.json"
        if identity.exists() and read(identity) != req:
            raise ValueError(f"Run signature mismatch: {out}. Use another output root.")
        if (out / "complete.json").exists():
            check_fit(args, arm, seed, loaded, recipe)
            print(f"Skip completed {arm}/seed{seed}", flush=True)
            return
        write_json(identity, req)
        gate, loss, weight = ARMS[arm]
        settings = dict(recipe)
        settings.update(
            mecg_enabled=gate, nptrbr_lambda=recipe["nptrbr_lambda"] if loss else 0.0,
            positive_loss_weight=weight, model_variant="mecg_star_tabm", run_mode="tune",
            device=args.device, seed=seed, output_dir=str(out), experiment_label=f"ulb_{arm}",
            data="public_ulb", _transfer_protocol="preserved_source_row_order",
            _resume_training=(out / "training_state.pt").exists(),
        )

        def fixed_split(*_):
            early, cal = ix["early_validation"], ix["calibration"]
            return ix["train"], np.r_[early, cal], ix["test"], {
                "_early_indices": early, "_calibration_indices": cal,
                "_calibration_info": {"fixed": True}, "strategy": "fixed_original_manifest",
            }

        patches = {
            "load_training_frame": lambda _: (frame, y),
            "fixed_split_for_training": fixed_split,
            "PublicPreprocessor": lambda **kw: InputView(PublicPreprocessor(**kw)),
            "build_model": OfficialTransfer,
        }
        old = {n: getattr(engine, n) for n in (*patches, "QUICK_CHECK_NUMERIC")}
        try:
            for n, val in patches.items():
                setattr(engine, n, val)
            engine.train_pipeline(argparse.Namespace(**settings))
        finally:
            for n, val in old.items():
                setattr(engine, n, val)
        assert not (out / "test_predictions.csv").exists()
        write_json(out / "complete.json", {n: sha256(out / n) for n in FILES})


def validated_cal(path, loaded):
    table = pd.read_csv(path / "validation_predictions.csv", float_precision="round_trip")
    table = table.sort_values("source_index").reset_index(drop=True)
    ids = loaded[2]["calibration"]
    if not np.array_equal(table.source_index, ids) or not np.array_equal(table.target, loaded[1][ids]):
        raise ValueError("Calibration identity/target mismatch")
    check_probability(table.raw_probability.to_numpy(float), len(ids))
    return table


def check_probability(raw, n):
    if raw.shape != (n,) or not np.isfinite(raw).all() or np.any((raw < 0) | (raw > 1)):
        raise ValueError("Probabilities are not finite, in range and aligned")


def average(frames):
    keys = ["source_index", "target"]
    if any(not f[keys].equals(frames[0][keys]) for f in frames[1:]):
        raise ValueError("Seed prediction identity/target mismatch")
    return frames[0].assign(raw_probability=np.mean(
        [f.raw_probability.to_numpy(float) for f in frames], axis=0))


def freeze_rules(cal):
    y = cal.target.to_numpy()
    raw = cal.raw_probability.to_numpy(float)
    platt = PlattCalibration().fit(raw, y)
    p = platt.predict(raw)
    rules = {
        "probability_0p5": 0.5,
        "calibration_max_f1": freeze_thresholds(y, p)["calibration_max_f1"],
    }
    neg = np.sort(p[y == 0])
    for cap in (0.001, 0.005):
        # Advance above the next excluded negative. Equal scores are excluded together.
        k = int(cap * len(neg))
        rules[f"calibration_FPR_le_{cap:g}"] = float(np.nextafter(neg[-1-k], np.inf))
    return platt, rules


def report(y, raw, decisions, **common):
    core = dict(common, dataset="ulb", global_ap=float(average_precision_score(y, raw)),
                auroc=float(roc_auc_score(y, raw)))
    return [dict(core, operating_point=rule, **classification_metrics(y, pred))
            for rule, pred in decisions.items()]


def development(args, loaded, recipe):
    result = []
    for arm in args.arms:
        paths = [check_fit(args, arm, s, loaded, recipe) for s in args.seeds]
        cal = average([validated_cal(p, loaded) for p in paths])
        raw, y = cal.raw_probability.to_numpy(float), cal.target.to_numpy()
        decisions = {}
        for train, test in StratifiedKFold(3, shuffle=True, random_state=20261109).split(raw, y):
            platt, cuts = freeze_rules(cal.iloc[train])
            p = platt.predict(raw[test])
            for name, cutoff in cuts.items():
                decisions.setdefault(name, np.zeros(len(cal), dtype=bool))[test] = p >= cutoff
        result.extend(report(y, raw, decisions, variant=arm, seed="ensemble",
                             split="development_oof_calibration", ensemble_seed_count=len(args.seeds)))
    write_csv(args.output / "development_metrics.csv", result)


def final_test(args, loaded, recipe):
    import torch
    from mecg_star_tabm._engine import predict_checkpoint_ensemble
    from mecg_star_tabm.model import ModelConfig
    from official_model import OfficialTransfer

    if not args.unlock_test:
        raise ValueError("Evaluation requires --unlock-test. Test is not used for training or calibration.")
    # Validate all requested completed runs before inferring any test prediction.
    fits = {(a, s): check_fit(args, a, s, loaded, recipe) for a in args.arms for s in args.seeds}
    frame, y, ix, meta = loaded
    out = getattr(args, "report_dir", None) or args.output / "evaluation"
    provenance = {"data_sha256": meta["data_sha256"], "manifest_sha256": meta["manifest_sha256"],
                  "seeds": list(args.seeds), "arms": list(args.arms), "test_used_for_selection": False,
                  "bootstrap_repeats": args.bootstrap_repeats, "intervals": "not requested",
                  "fits": {f"{a}/seed{s}": sha256(path / "complete.json") for (a, s), path in fits.items()}}
    settings = out / "evaluation_request.json"
    if settings.exists() and read(settings) != provenance:
        raise ValueError("Existing evaluation settings differ; use a different --report-dir.")
    write_json(settings, provenance)
    ids = ix["test"]
    targets = y[ids]
    if not np.array_equal(np.unique(targets), [0, 1]):
        raise ValueError("Test must contain both classes")
    rows, ensemble_scores, ensemble_decisions = [], {}, {}
    device = torch.device(args.device)
    for arm in args.arms:
        cals, raw_seeds, checkpoints, params = [], [], [], []
        for seed in args.seeds:
            path = fits[arm, seed]
            cals.append(validated_cal(path, loaded))
            # Only files produced by this run and hash-checked above should be loaded.
            cp = torch.load(path / "model.pt", map_location="cpu", weights_only=False)
            with (path / "preprocessor.pkl").open("rb") as handle:
                prep = pickle.load(handle)
            model = OfficialTransfer(ModelConfig(**cp["model_config"])).to(device)
            states = cp["checkpoint_state_dicts"]
            arrays = prep.transform(frame.iloc[ids].copy())
            if not np.array_equal(arrays.source_index, ids):
                raise ValueError("Preprocessing reordered the fixed test rows")
            raw, _, _ = predict_checkpoint_ensemble(model, states, arrays, device, recipe["prediction_batch_size"])
            raw_seeds.append(np.asarray(raw, dtype=np.float64))
            checkpoints.append(len(states))
            params.append(int(read(path / "metrics.json")["parameter_count"]))
            del model
        if len(set(params)) != 1:
            raise ValueError("Parameter count changed across seeds")
        runs = [(str(s), c, raw, 1, n) for s, c, raw, n in zip(args.seeds, cals, raw_seeds, checkpoints)]
        # The supplement casts each checkpoint-ensemble result to float64 before seed averaging.
        runs.append(("ensemble", average(cals), np.mean(raw_seeds, axis=0), len(args.seeds), sum(checkpoints)))
        for label, cal, raw, n_seeds, n_ckpt in runs:
            check_probability(np.asarray(raw), len(ids))
            if np.intersect1d(cal.source_index, ids).size:
                raise ValueError("Calibration/test overlap")
            platt, cuts = freeze_rules(cal)
            p = platt.predict(np.asarray(raw, dtype=np.float64))
            decisions = {k: p >= v for k, v in cuts.items()}
            local = report(targets, raw, decisions, variant=arm, seed=label, split="test",
                           positive_weight=ARMS[arm][2], single_network_parameters=params[0],
                           ensemble_seed_count=n_seeds, checkpoint_forwards=n_ckpt)
            rows.extend(dict(r, calibration_threshold=cuts[r["operating_point"]]) for r in local)
            if label == "ensemble":
                ensemble_scores[arm], ensemble_decisions[arm] = raw, decisions
    write_csv(out / "test_metrics.csv", rows)
    write_csv(out / "test_ensemble_metrics.csv", [r for r in rows if r["seed"] == "ensemble"])
    if args.bootstrap_repeats:
        contrasts = [(a, b) for a, b in (("nptrbr", "bce"), ("nptrbr", "wbce_w2"), ("wbce_w2", "bce"))
                     if a in ensemble_scores and b in ensemble_scores]
        if not contrasts:
            raise ValueError("Bootstrap requires at least one of the matched BCE/weighted BCE contrasts")
        relevant = {n for pair in contrasts for n in pair}
        ci, meta = paired_bootstrap(
            y=targets, scores={a: v for a, v in ensemble_scores.items() if a in relevant},
            decisions={a: v for a, v in ensemble_decisions.items() if a in relevant}, contrasts=contrasts,
            scheme="ulb/class_stratified_rows", state_path=out / "_local_bootstrap.npz",
            repeats=args.bootstrap_repeats, seed=20261109,
        )
        write_csv(out / "paired_intervals.csv", [dict(dataset="ulb", **r) for r in ci])
        provenance["intervals"] = meta
    write_json(out / "evaluation_protocol.json", provenance)
    print(pd.DataFrame([r for r in rows if r["seed"] == "ensemble" and
                        r["operating_point"] == "calibration_FPR_le_0.001"])[
          ["variant", "global_ap", "auroc", "recall", "tp", "fp", "tn", "fn"]].to_string(index=False))


def smoke():
    import torch
    from official_model import OfficialTransfer
    from mecg_star_tabm import _engine as engine
    from mecg_star_tabm.model import ModelConfig

    torch.set_num_threads(2)
    recipe = dict(RECIPE, members=4, hidden_dim=16, blocks=2, epochs=2, batch_size=64,
                  early_stop_warmup=1, nptrbr_warmup=1, mecg_warmup=1, checkpoint_ensemble_top_k=2,
                  member_negative_ratio=2.0, ple_bins=4)
    with TemporaryDirectory(prefix="ulb_smoke_") as td:
        root = Path(td)
        data, split = root / "synthetic.csv", root / "split"
        rng = np.random.default_rng(74)
        f = pd.DataFrame(rng.normal(size=(600, 29)), columns=FEATURES)
        f["Amount"] = f.Amount.abs() + 1
        f[TIMESTAMP] = "2013-09-01"
        f[TARGET] = (np.arange(len(f)) % 5 == 0).astype(np.int64)
        f.to_csv(data, index=False)
        create_split_manifest(data, split, allow_synthetic=True)
        loaded = load_data(data, split, synthetic=True)
        a = argparse.Namespace(output=root / "runs", device="cpu", seeds=(42, 7),
                               arms=tuple(ARMS), unlock_test=True, bootstrap_repeats=100)
        for arm in a.arms:
            for seed in a.seeds:
                fit(a, arm, seed, loaded, recipe)
        fit(a, "bce", 42, loaded, recipe)  # completed-run reuse
        # Simulate an interruption just after saving epoch 1.
        restart = argparse.Namespace(output=root / "restart", device="cpu")
        original_save = engine.save_epoch_checkpoint
        class InterruptAfterSave(RuntimeError):
            pass
        def interrupt(path, state):
            original_save(path, state)
            raise InterruptAfterSave
        try:
            engine.save_epoch_checkpoint = interrupt
            try:
                fit(restart, "nptrbr", 42, loaded, recipe)
                raise AssertionError("Expected a simulated interruption")
            except InterruptAfterSave:
                pass
        finally:
            engine.save_epoch_checkpoint = original_save
        fit(restart, "nptrbr", 42, loaded, recipe)
        pd.testing.assert_frame_equal(
            validated_cal(run_dir(a, "nptrbr", 42), loaded),
            validated_cal(run_dir(restart, "nptrbr", 42), loaded),
        )
        cp = torch.load(run_dir(a, "bce", 42) / "model.pt", map_location="cpu", weights_only=False)
        model = OfficialTransfer(ModelConfig(**cp["model_config"])).eval()
        model.load_state_dict(cp["model_state_dict"])
        xn, empty = torch.ones(5, 29), torch.empty(5, 0)
        with torch.no_grad():
            assert torch.equal(model.network(xn).squeeze(-1), model(xn, empty, empty, empty)[0])
        development(a, loaded, recipe)
        final_test(a, loaded, recipe)
        rows = pd.read_csv(a.output / "evaluation" / "test_metrics.csv")
        assert len(rows) == 60 and (rows.tp + rows.fp + rows.tn + rows.fn == 120).all()
        assert len(pd.read_csv(a.output / "evaluation" / "test_ensemble_metrics.csv")) == 20
        assert (a.output / "evaluation" / "paired_intervals.csv").is_file()
    print("PASS: five arms, two seeds, reuse and epoch resume, calibration, 60 confusion matrices and paired intervals")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    cmds = p.add_subparsers(dest="command", required=True)
    for name in ("prepare", "split", "audit", "train", "development", "evaluate"):
        cmd = cmds.add_parser(name)
        cmd.add_argument("--data", type=Path, default=Path("data/ulb_prepared.csv"))
        cmd.add_argument("--split-dir", type=Path, default=Path("artifacts/ulb_split"))
        if name == "prepare":
            cmd.add_argument("--input", type=Path, help="Optional local public creditcard.csv in the original order")
        if name in ("train", "development", "evaluate"):
            cmd.add_argument("--output", type=Path, default=Path("runs/ulb_paper"))
            cmd.add_argument("--device", default="cuda:0")
            cmd.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
            cmd.add_argument("--arms", nargs="+", choices=tuple(ARMS), default=tuple(ARMS))
        if name == "evaluate":
            cmd.add_argument("--unlock-test", action="store_true", required=True)
            cmd.add_argument("--bootstrap-repeats", type=int, default=2000, help="Use 0 to skip intervals")
            cmd.add_argument("--report-dir", type=Path, help="Default: <output>/evaluation")
    cmds.add_parser("smoke")
    return p


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    p = parser()
    args = p.parse_args()
    if args.command == "smoke":
        smoke()
        return
    if args.command == "prepare":
        prepare_ulb(args.data, args.input)
    if args.command in ("prepare", "split"):
        create_split_manifest(args.data, args.split_dir)
    loaded = load_data(args.data, args.split_dir)
    if args.command in ("prepare", "split", "audit"):
        print(json.dumps(loaded[3]["counts"], indent=2))
        return
    if len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0:
        p.error("Seeds must be distinct nonnegative integers")
    if len(set(args.arms)) != len(args.arms):
        p.error("Arms must be distinct")
    # Use a canonical arm order for frozen bootstrapping.
    args.arms = tuple(a for a in ARMS if a in args.arms)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.command == "train":
        for arm in args.arms:
            for seed in args.seeds:
                fit(args, arm, seed, loaded, RECIPE)
    elif args.command == "development":
        development(args, loaded, RECIPE)
    else:
        if args.bootstrap_repeats != 0 and args.bootstrap_repeats < 100:
            p.error("Use 0 to skip intervals, or at least 100 bootstrap repeats")
        if args.bootstrap_repeats and not any(a in args.arms and b in args.arms for a,b in
             (("nptrbr", "bce"), ("nptrbr", "wbce_w2"), ("wbce_w2", "bce"))):
            p.error("Select a matched contrast or use --bootstrap-repeats 0")
        report_dir = args.report_dir or args.output / "evaluation"
        report_dir.mkdir(parents=True, exist_ok=True)
        with FileLock(str(report_dir / ".evaluation.lock"), timeout=0):
            final_test(args, loaded, RECIPE)


if __name__ == "__main__":
    main()
