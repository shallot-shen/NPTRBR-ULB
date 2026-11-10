"""ULB-only input allowlist and a single fingerprinted, order-preserving split."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = [f"V{i}" for i in range(1, 29)] + ["Amount"]
TARGET = "__target__"
TIMESTAMP = "__timestamp__"
GROUP = "__row_group__"
EXPECTED_ROWS = 284807
EXPECTED_POSITIVES = 492
EXPECTED_SPLIT_ROWS = {
    "train": 182277,
    "early_validation": 30839,
    "calibration": 14730,
    "test": 56961,
}
EXPECTED_SPLIT_POSITIVES = {"train": 365, "early_validation": 32, "calibration": 20, "test": 75}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dump_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False) + "\n", encoding="utf-8"
    )


def validate_public_frame(frame, *, allow_synthetic=False):
    required = set(FEATURES + [TARGET, TIMESTAMP])
    allowed = required | {GROUP}
    if set(frame) - allowed or required - set(frame):
        raise ValueError(
            "Not the allowlisted prepared ULB schema; arbitrary tables are not accepted"
        )
    values = frame[FEATURES].apply(pd.to_numeric, errors="raise").to_numpy()
    if not np.isfinite(values).all():
        raise ValueError("ULB features must be finite, without dropped/reordered rows")
    target = pd.to_numeric(frame[TARGET], errors="raise").to_numpy()
    if not np.isin(target, [0, 1]).all() or np.unique(target).size != 2:
        raise ValueError("ULB target must contain both binary 0/1 classes")
    ts = pd.to_datetime(frame[TIMESTAMP], errors="raise")
    if ts.nunique() != 1 or ts.iloc[0] != pd.Timestamp("2013-09-01"):
        raise ValueError("The frozen ULB recipe uses a constant 2013-09-01 placeholder")
    if not allow_synthetic and (
        len(frame) != EXPECTED_ROWS or int(target.sum()) != EXPECTED_POSITIVES
    ):
        raise ValueError("Expected the complete ULB dataset: 284807 rows and 492 positives")
    return target.astype(np.int64)


def read_prepared(path, *, allow_synthetic=False):
    if Path(path).suffix.lower() != ".csv":
        raise ValueError("The evaluated public preparation uses CSV")
    # Preserve the original trainer's numeric parser; round-trip reading is
    # used for exported prediction CSVs, not to change training inputs.
    frame = pd.read_csv(path, low_memory=False)
    y = validate_public_frame(frame, allow_synthetic=allow_synthetic)
    return frame, y


def prepare_ulb(output, input_path=None):
    if input_path:
        raw = pd.read_csv(input_path, low_memory=False)
    else:
        from sklearn.datasets import fetch_openml

        print("Preparing public OpenML 1597 (ULB); no credentials are required.", flush=True)
        cache = Path(output).resolve().parent / "openml_cache"
        bunch = fetch_openml(data_id=1597, as_frame=True, parser="auto", data_home=str(cache))
        raw = bunch.frame.copy()
        if "Class" not in raw and bunch.target is not None:
            raw["Class"] = bunch.target
    aliases = {str(c).strip().casefold(): c for c in raw.columns}
    expected = FEATURES + ["Class"]
    missing = [c for c in expected if c.casefold() not in aliases]
    if missing:
        raise ValueError(f"Missing ULB fields: {missing}")
    raw = raw.rename(columns={aliases[c.casefold()]: c for c in expected})
    frame = raw[FEATURES].apply(pd.to_numeric, errors="raise").copy()
    frame[TARGET] = pd.to_numeric(raw["Class"], errors="raise").to_numpy()
    # Always reproduce the evaluated OpenML row-order protocol, even if a
    # local canonical CSV supplies Time. It is neither a feature nor a sorter.
    frame[TIMESTAMP] = "2013-09-01"
    frame[GROUP] = (np.arange(len(frame), dtype=np.int64) // 10000).astype(str)
    validate_public_frame(frame)
    out = Path(output)
    if out.exists():
        raise FileExistsError(f"Refusing to replace prepared data: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False, encoding="utf-8-sig")
    dump_json(
        str(out) + ".schema.json",
        {
            "dataset": "ULB/Worldline credit-card fraud",
            "openml_data_id": 1597,
            "features": FEATURES,
            "target": TARGET,
            "time_is_feature": False,
            "order": "preserved source row order",
            "chronology_verified": False,
            "timestamp": "constant compatibility placeholder",
            "rows": len(frame),
            "positives": int(frame[TARGET].sum()),
            "sha256": sha256(out),
            "source": "https://www.openml.org/d/1597",
        },
    )
    print(
        f"Prepared public ULB: {len(frame):,} rows, {int(frame[TARGET].sum())} positives -> {out}",
        flush=True,
    )
    return out


def create_split_manifest(data, split_dir, *, allow_synthetic=False):
    frame, y = read_prepared(data, allow_synthetic=allow_synthetic)
    dest = Path(split_dir)
    dest.mkdir(parents=True, exist_ok=True)
    manifest_path = dest / "split_manifest.csv"
    if manifest_path.exists():
        verify_manifest(data, dest, allow_synthetic=allow_synthetic)
        return manifest_path
    from ._engine import split_validation_for_calibration

    n = len(frame)
    test_start = int(round(n * 0.8))
    val_start = int(round(test_start * 0.8))
    train = np.arange(val_start, dtype=np.int64)
    validation = np.arange(val_start, test_start, dtype=np.int64)
    early, calibration, info = split_validation_for_calibration(
        validation, y, frame[TIMESTAMP], 0.25, 20
    )
    if np.intersect1d(early, calibration).size:
        raise ValueError("Disjoint early-validation and calibration partitions are required")
    split = np.full(n, "test", dtype=object)
    split[train] = "train"
    split[early] = "early_validation"
    split[calibration] = "calibration"
    counts = {
        s: {"rows": int(np.sum(split == s)), "positives": int(y[split == s].sum())}
        for s in EXPECTED_SPLIT_ROWS
    }
    if not allow_synthetic:
        if {s: m["rows"] for s, m in counts.items()} != EXPECTED_SPLIT_ROWS or {
            s: m["positives"] for s, m in counts.items()
        } != EXPECTED_SPLIT_POSITIVES:
            raise ValueError(
                "ULB row order does not match the evaluated partition counts; do not silently reshuffle"
            )
    pd.DataFrame({"source_index": np.arange(n), "split": split}).to_csv(
        manifest_path, index=False, encoding="utf-8-sig"
    )
    dump_json(
        dest / "split_metadata.json",
        {
            "protocol": "fixed_order_preserving_ULB",
            "data_sha256": sha256(data),
            "manifest_sha256": sha256(manifest_path),
            "counts": counts,
            "validation_subsplit": info,
            "chronology_verified": False,
            "calibration_cut_uses_only_validation_labels": True,
        },
    )
    return manifest_path


def verify_manifest(data, split_dir, *, allow_synthetic=False):
    frame, y = read_prepared(data, allow_synthetic=allow_synthetic)
    folder = Path(split_dir)
    meta = json.loads((folder / "split_metadata.json").read_text(encoding="utf-8"))
    path = folder / "split_manifest.csv"
    if meta["data_sha256"] != sha256(data) or meta["manifest_sha256"] != sha256(path):
        raise ValueError("Data/manifest fingerprint changed; use the original public preparation")
    manifest = pd.read_csv(path)
    if list(manifest.columns) != ["source_index", "split"] or not np.array_equal(
        manifest.source_index.to_numpy(), np.arange(len(frame))
    ):
        raise ValueError("Manifest must cover every prepared row exactly once, in source order")
    if set(manifest["split"]) != set(EXPECTED_SPLIT_ROWS):
        raise ValueError("Exactly four disjoint partitions are required")
    indices = {s: np.flatnonzero(manifest["split"].to_numpy() == s) for s in EXPECTED_SPLIT_ROWS}
    for name, idx in indices.items():
        if name != "test" and np.unique(y[idx]).size != 2:
            raise ValueError(f"Both classes required in {name}")
    return frame, y, indices, meta


def load_training_frame(args):
    frame, y = read_prepared(args.data, allow_synthetic=bool(getattr(args, "_synthetic", False)))
    frame["_source_index"] = np.arange(len(frame), dtype=np.int64)
    frame[TIMESTAMP] = pd.to_datetime(frame[TIMESTAMP])
    return frame, y


def fixed_split_for_training(args, frame, y):
    _, _, idx, meta = verify_manifest(
        args.data, args.split_dir, allow_synthetic=bool(getattr(args, "_synthetic", False))
    )
    info = {
        "strategy": "fixed_order_preserving_ULB",
        "chronology_verified": False,
        "data_sha256": meta["data_sha256"],
        "manifest_sha256": meta["manifest_sha256"],
        "_early_indices": idx["early_validation"],
        "_calibration_indices": idx["calibration"],
        "_calibration_info": meta["validation_subsplit"],
    }
    return (
        idx["train"],
        np.sort(np.concatenate([idx["early_validation"], idx["calibration"]])),
        idx["test"],
        info,
    )
