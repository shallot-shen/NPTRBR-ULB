"""Generate public ranking, confusion-matrix and matched factorial tables."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ARMS = {
    "bce": (False, False, "Matched BCE"), "wbce_w2": (False, False, "Weighted BCE (w+=2)"),
    "nptrbr": (False, True, "NP-TRBR"), "mecg_bce": (True, False, "MECG + BCE"),
    "mecg_nptrbr": (True, True, "MECG + NP-TRBR"),
}
RULE = "calibration_FPR_le_0.001"
RULES = (RULE, "calibration_FPR_le_0.005", "probability_0p5", "calibration_max_f1")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metrics", type=Path, default=Path("runs/ulb_paper/evaluation/test_ensemble_metrics.csv"))
    p.add_argument("--output", type=Path, default=Path("tables/ulb"))
    args = p.parse_args()
    d = pd.read_csv(args.metrics)
    if not d.dataset.eq("ulb").all() or not d.seed.eq("ensemble").all() or not d.split.eq("test").all():
        raise ValueError("Expected an ensemble-only test table for public ULB.")
    if not set(d.variant) <= set(ARMS) or d.duplicated(["variant", "operating_point"]).any():
        raise ValueError("Unknown/duplicate arm or operating point.")
    if not d.rows.eq(56_961).all() or not d.positives.eq(75).all():
        raise ValueError("Expected the manuscript's ULB test split.")
    if not (d.tp+d.fp+d.tn+d.fn).eq(d.rows).all() or not (d.tp+d.fn).eq(d.positives).all():
        raise ValueError("Confusion counts disagree with the test cohort.")
    counts = d.groupby("variant").operating_point.agg(set)
    if not all(v == set(RULES) for v in counts):
        raise ValueError("Each arm must report all four operating points.")
    d["model"] = d.variant.map(lambda x: ARMS[x][2])
    d["mecg"] = d.variant.map(lambda x: ARMS[x][0])
    d["nptrbr"] = d.variant.map(lambda x: ARMS[x][1])
    d["variant"] = pd.Categorical(d.variant, categories=ARMS, ordered=True)
    d["operating_point"] = pd.Categorical(d.operating_point, categories=RULES, ordered=True)
    d = d.sort_values(["operating_point", "variant"])
    rank = d.loc[d.operating_point.eq(RULE), ["model", "global_ap", "auroc"]]
    metrics = ["tp", "fp", "tn", "fn", "precision", "recall", "false_positive_rate"]
    decisions = d.loc[:, ["operating_point", "model", *metrics]]
    fac = d.loc[d.operating_point.eq(RULE) & ~d.variant.eq("wbce_w2"),
                ["mecg", "nptrbr", "model", "global_ap", "auroc", *metrics]]
    args.output.mkdir(parents=True, exist_ok=True)
    for filename, data in (("matched_ranking.csv", rank), ("matched_decisions.csv", decisions),
                           ("matched_factorial.csv", fac)):
        data.to_csv(args.output / filename, index=False, encoding="utf-8-sig")
    print(rank.to_string(index=False))


if __name__ == "__main__":
    main()
