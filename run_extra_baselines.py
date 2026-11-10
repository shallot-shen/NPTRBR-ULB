"""Train the optional ULB comparators on the paper split without changing it."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

import pandas as pd

from reproduce_ulb import DEFAULT_SEEDS, load_data
from mecg_star_tabm.extra_baselines import MODELS


def invoke(*tokens):
    subprocess.run([sys.executable, "-m", "mecg_star_tabm.extra_baselines", *map(str, tokens)], check=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("data/ulb_prepared.csv"))
    p.add_argument("--split-dir", type=Path, default=Path("artifacts/ulb_split"))
    p.add_argument("--output", type=Path, default=Path("runs/ulb_extra"))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--tree-device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    p.add_argument("--models", choices=MODELS, nargs="+", default=MODELS)
    p.add_argument("--unlock-test", action="store_true")
    args = p.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0:
        p.error("Seeds must be distinct nonnegative integers")
    _, y, indices, meta = load_data(args.data, args.split_dir)
    # The archived runner needs an identity witness in addition to the manifest.
    # These are calibration row IDs/targets from the verified public split, not predictions.
    ref = args.output / "reference" / meta["manifest_sha256"]
    ref.mkdir(parents=True, exist_ok=True)
    (ref / "split_manifest.csv").write_bytes((args.split_dir / "split_manifest.csv").read_bytes())
    ids = indices["calibration"]
    pd.DataFrame({"source_index": ids, "target": y[ids]}).to_csv(
        ref / "validation_predictions.csv", index=False)
    for name in args.models:
        paths = []
        for seed in args.seeds:
            out = args.output / name / f"seed{seed}"
            options = ("--data", args.data, "--reference-dir", ref, "--output-dir", out,
                       "--model", name, "--seed", seed, "--device", args.device, "--tree-device", args.tree_device)
            invoke("train", *options)
            if args.unlock_test:
                invoke("test", *options, "--unlock-test")
            paths.append(str(out))
        spec = ("--artifact-dirs", ",".join(paths), "--output-dir", args.output / name / "ensemble")
        invoke("ensemble", *spec, "--evaluation-set", "calibration")
        if args.unlock_test:
            invoke("ensemble", *spec, "--evaluation-set", "test", "--unlock-test")


if __name__ == "__main__":
    main()
