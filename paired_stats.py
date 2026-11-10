"""Paired percentile intervals at already-frozen calibration thresholds."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import pandas as pd


ENDPOINTS = ("recall", "false_positive_rate", "precision", "f1")


def write_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.stem + "_", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            np.savez_compressed(handle, **arrays)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


class WeightedAP:
    """Step AP with resampling multiplicities, treating tied scores as one cut."""

    def __init__(self, target: np.ndarray, score: np.ndarray):
        score = np.asarray(score, dtype=np.float64)
        if score.shape != target.shape or not np.isfinite(score).all():
            raise ValueError("AP score is non-finite or not aligned")
        self.order = np.argsort(-score, kind="stable")
        positive_ranks = np.flatnonzero(target[self.order] == 1)
        if positive_ranks.size == 0:
            raise ValueError("AP requires positives")
        self.positive_ids = self.order[positive_ranks]
        ordered_keys = -score[self.order]
        tie_ends = np.searchsorted(ordered_keys, ordered_keys[positive_ranks], side="right") - 1
        self.starts = np.r_[0, np.flatnonzero(np.diff(tie_ends)) + 1]
        self.ends = tie_ends[self.starts]

    def __call__(self, multiplicity: np.ndarray) -> float:
        new_tp = np.add.reduceat(multiplicity[self.positive_ids], self.starts)
        total_tp = np.cumsum(new_tp, dtype=np.int64)
        all_selected = np.cumsum(multiplicity[self.order], dtype=np.int64)[self.ends]
        precision = np.divide(
            total_tp, all_selected, out=np.zeros(len(total_tp)), where=all_selected > 0
        )
        return float(np.dot(new_tp, precision) / total_tp[-1])


def encode_blocks(values) -> np.ndarray:
    values = pd.Series(values).astype("string").str.strip()
    if values.isna().any() or values.eq("").any():
        raise ValueError("A test block key is missing; do not silently merge unidentified blocks.")
    codes, unique = pd.factorize(values, sort=True)
    if len(unique) < 2:
        raise ValueError("At least two distinct blocks are required.")
    return codes.astype(np.int64)


def _counts(tp: int, fp: int, positives: int, negatives: int) -> tuple:
    return (
        tp / positives,
        fp / negatives,
        tp / max(tp + fp, 1),
        2 * tp / (positives + tp + fp),
    )


def paired_bootstrap(
    *,
    y: np.ndarray,
    scores: dict[str, np.ndarray],
    decisions: dict[str, dict[str, np.ndarray]],
    contrasts: list[tuple[str, str]],
    scheme: str,
    state_path: Path,
    repeats: int = 2000,
    seed: int = 20261109,
    codes: np.ndarray | None = None,
) -> tuple[list[dict], dict]:
    """Same draw for all models; do not re-estimate thresholds or fit models."""
    if repeats < 100 or seed < 0:
        raise ValueError("Use >=100 draws and a nonnegative bootstrap seed.")
    y = np.asarray(y)
    if y.ndim != 1 or not np.array_equal(np.unique(y), [0, 1]):
        raise ValueError("Paired inference requires one aligned binary target with both classes.")
    y = y.astype(np.int8)
    names = list(scores)
    if len(names) < 2 or set(decisions) != set(names) or not contrasts:
        raise ValueError("Provide scores/decisions for the same >=2 models and explicit contrasts.")
    if any(a == b or a not in scores or b not in scores for a, b in contrasts):
        raise ValueError("Unknown/identical model in contrast.")
    rules = tuple(decisions[names[0]])
    if not rules or any(tuple(decisions[n]) != rules for n in names):
        raise ValueError("Operating points differ between models.")
    endpoints = ["global_ap"] + [f"{r}/{m}" for r in rules for m in ENDPOINTS]
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    aps = {n: WeightedAP(y, scores[n]) for n in names}

    # Keep positive/negative alert indices rather than recalculating masks per draw.
    alert_indices = {}
    for n in names:
        alert_indices[n] = {}
        for rule in rules:
            d = np.asarray(decisions[n][rule])
            if d.dtype != np.bool_ or d.shape != y.shape:
                raise ValueError("Decisions must be aligned boolean vectors.")
            alert_indices[n][rule] = (pos[d[pos]], neg[d[neg]])

    n_groups = None
    n_positive_groups = None
    if codes is not None:
        codes = np.asarray(codes, dtype=np.int64)
        if codes.shape != y.shape or not np.array_equal(np.unique(codes), np.arange(codes.max() + 1)):
            raise ValueError("Block codes are not aligned/contiguous.")
        n_groups = int(codes.max() + 1)
        if n_groups < 2:
            raise ValueError("At least two bootstrap blocks are required.")
        group_pos = np.bincount(codes[pos], minlength=n_groups)
        group_neg = np.bincount(codes[neg], minlength=n_groups)
        n_positive_groups = int(np.count_nonzero(group_pos))

    signature = hashlib.sha256(json.dumps(
        [2, names, endpoints, scheme, repeats, seed, codes is not None], ensure_ascii=True
    ).encode())
    signature.update(y.tobytes())
    if codes is not None:
        signature.update(codes.astype("<i8", copy=False).tobytes())
    for n in names:
        signature.update(np.asarray(scores[n], dtype="<f8").tobytes())
        for r in rules:
            signature.update(decisions[n][r].tobytes())
    fingerprint = signature.hexdigest()

    values = np.full((repeats, len(names), len(endpoints)), np.nan)
    done = np.zeros(repeats, dtype=bool)
    if state_path.is_file():
        with np.load(state_path, allow_pickle=False) as state:
            if state["fingerprint"].item() != fingerprint:
                raise ValueError("Bootstrap inputs changed; use a new output root or the same settings.")
            values[:] = state["values"]
            done[:] = state["done"]
        print(f"{scheme}: resume {done.sum()}/{repeats} draws", flush=True)

    def measure(w, n_pos, n_neg):
        result = np.empty((len(names), len(endpoints)), dtype=np.float64)
        for row, name in enumerate(names):
            result[row, 0] = aps[name](w)
            for j, rule in enumerate(rules):
                hits, false_alerts = alert_indices[name][rule]
                start = 1 + j * len(ENDPOINTS)
                result[row, start:start + len(ENDPOINTS)] = _counts(
                    int(w[hits].sum()), int(w[false_alerts].sum()), n_pos, n_neg
                )
        return result

    def save_progress():
        write_npz(state_path, fingerprint=fingerprint, values=values, done=done)

    t0 = time.perf_counter()
    initial = int(done.sum())
    try:
        for j, b in enumerate(np.flatnonzero(~done), start=1):
            rng = np.random.default_rng(np.random.SeedSequence([seed, int(b)]))
            if codes is not None:
                block_weights = np.bincount(rng.integers(n_groups, size=n_groups), minlength=n_groups)
                p, n = int(block_weights @ group_pos), int(block_weights @ group_neg)
                if p and n:
                    values[b] = measure(block_weights[codes], p, n)
            else:
                # ULB has no card/customer ID: stratify individual rows by class.
                p, n = len(pos), len(neg)
                weights = np.empty(len(y), dtype=np.int64)
                weights[pos] = np.bincount(rng.integers(p, size=p), minlength=p)
                weights[neg] = np.bincount(rng.integers(n, size=n), minlength=n)
                values[b] = measure(weights, p, n)
            done[b] = True
            if j % 50 == 0 or j + initial == repeats:
                save_progress()
                elapsed = time.perf_counter() - t0
                eta = elapsed / j * (repeats - j - initial)
                print(f"{scheme}: {j + initial}/{repeats}, estimated remaining {eta:.0f} s", flush=True)
    finally:
        if int(done.sum()) != initial:
            save_progress()

    valid = np.isfinite(values).all(axis=(1, 2))
    valid_draws = int(valid.sum())
    if valid_draws < 100:
        raise ValueError(f"{scheme}: fewer than 100 resamples retain both classes.")
    point = measure(np.ones(len(y), dtype=np.int64), len(pos), len(neg))
    rows = []
    for left, right in contrasts:
        a, b = names.index(left), names.index(right)
        low, high = np.quantile(values[valid, a] - values[valid, b], [.025, .975], axis=0)
        for k, endpoint in enumerate(endpoints):
            rows.append(dict(
                scheme=scheme, contrast=f"{left}_minus_{right}", endpoint=endpoint,
                point=float(point[a, k] - point[b, k]),
                ci_low=float(low[k]), ci_high=float(high[k]), valid_draws=valid_draws,
            ))
    warnings = []
    if n_positive_groups is not None and n_positive_groups < 10:
        warnings.append("Fewer than ten blocks contain positives.")
    if valid_draws < 0.9 * repeats:
        warnings.append("More than 10% of resamples lose a class.")
    return rows, dict(
        scheme=scheme, draws=repeats, valid_draws=valid_draws, seed=seed, blocks=n_groups,
        positive_blocks=n_positive_groups, warnings=warnings,
        interval="paired percentile 95%; nominal, not multiplicity-adjusted",
        conditioning="fixed fitted seed ensembles, Platt mappings and calibration thresholds",
    )
