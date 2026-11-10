from __future__ import annotations
import logging
import math
from typing import Any, Dict, List, Sequence
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression

LOGGER = logging.getLogger("mecg_star_tabm")
EPS = 1e-6
MISSING_TOKEN = "__MISSING__"

from .preprocessing import PreparedArrays

MECG_GROUP_DESCRIPTIONS = {
    "calendar": "Constant compatibility context; no temporal information",
    "missingness": "Train-observed missing-value indicators",
    "other_numeric": "ULB anonymous PCA components and Amount",
    "other_categorical": "Categorical fields (unused on ULB)",
}


def mecg_semantic_group(feature_name, categorical=False):
    if categorical:
        return "other_categorical"
    if feature_name.startswith("MISS__"):
        return "missingness"
    if feature_name.startswith(("_month_", "_doy_", "_hour_", "_annual_")):
        return "calendar"
    return "other_numeric"


def _rank_unit_interval(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.size <= 1:
        return np.full(values.shape, 0.5, dtype=np.float64)
    finite = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    order = np.argsort(finite, kind="stable")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(order), dtype=np.float64)
    return ranks / float(max(len(order) - 1, 1))


def _categorical_rate_dispersion(values: np.ndarray, y: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.int64)
    y = np.asarray(y, dtype=np.float64)
    if len(values) == 0 or y.sum() == 0 or y.sum() == len(y):
        return 0.0
    categories, inverse = np.unique(values, return_inverse=True)
    count = np.bincount(inverse, minlength=len(categories)).astype(np.float64)
    positive = np.bincount(inverse, weights=y, minlength=len(categories)).astype(np.float64)
    global_rate = float((y.sum() + 0.5) / (len(y) + 1.0))
    strength = 20.0
    rate = (positive + strength * global_rate) / (count + strength)
    dispersion = np.sum(count * np.abs(rate - global_rate)) / max(count.sum(), 1.0)
    return float(dispersion / max(math.sqrt(global_rate * (1.0 - global_rate)), EPS))


def compute_mecg_train_evidence(
    arrays: PreparedArrays,
    y: np.ndarray,
    numeric_names: Sequence[str],
    categorical_names: Sequence[str],
    seed: int,
    max_rows: int,
    prior_min: float,
    prior_max: float,
    redundancy_threshold: float,
    use_evidence_prior: bool,
) -> Dict[str, Any]:
    """Fit MECG evidence exclusively on the training partition.

    Four complementary signals are deliberately used instead of declaring a
    feature relevant from one p-value or one fitted model: standardized class
    separation, mutual information, joint logistic coefficients and temporal
    stability.  Categorical fields use smoothed rate dispersion in place of a
    one-hot logistic coefficient, avoiding a target-encoded feature leak.
    """
    y = np.asarray(y, dtype=np.int8)
    if len(y) != len(arrays) or y.sum() == 0 or y.sum() == len(y):
        raise ValueError("MECG")
    if not 0.0 < prior_min < prior_max < 1.0:
        raise ValueError("MECG 0 < min < max < 1")
    rng = np.random.default_rng(seed)
    positive = np.flatnonzero(y == 1)
    negative = np.flatnonzero(y == 0)
    negative_budget = max(int(max_rows) - len(positive), len(positive))
    if len(negative) > negative_budget:
        negative = rng.choice(negative, size=negative_budget, replace=False)
    sample = np.sort(np.concatenate([positive, negative])).astype(np.int64)
    sample_y = y[sample]
    x_num = np.asarray(arrays.x_numeric[sample], dtype=np.float64)
    x_cat = np.asarray(arrays.x_categorical[sample], dtype=np.int64)
    time_group = np.asarray(arrays.time_group[sample], dtype=np.int64)
    n_numeric = x_num.shape[1]
    n_categorical = x_cat.shape[1]
    if n_numeric != len(numeric_names) or n_categorical != len(categorical_names):
        raise ValueError("MECG")
    numeric_effect = np.zeros(n_numeric, dtype=np.float64)
    if n_numeric:
        pos_mean = x_num[sample_y == 1].mean(axis=0)
        neg_mean = x_num[sample_y == 0].mean(axis=0)
        pooled_std = x_num.std(axis=0)
        numeric_effect = np.abs(pos_mean - neg_mean) / np.maximum(pooled_std, EPS)
        numeric_effect = np.nan_to_num(numeric_effect, nan=0.0, posinf=0.0)
        numeric_mi = mutual_info_classif(
            x_num, sample_y, discrete_features=False, n_neighbors=3, random_state=seed
        )
        logistic = LogisticRegression(
            C=0.25,
            l1_ratio=0.0,
            class_weight="balanced",
            solver="liblinear",
            max_iter=500,
            random_state=seed,
        )
        logistic.fit(x_num, sample_y)
        numeric_logistic = np.abs(logistic.coef_.reshape(-1))
    else:
        numeric_mi = np.empty(0, dtype=np.float64)
        numeric_logistic = np.empty(0, dtype=np.float64)
    categorical_effect = np.asarray(
        [_categorical_rate_dispersion(x_cat[:, index], sample_y) for index in range(n_categorical)],
        dtype=np.float64,
    )
    categorical_mi = (
        mutual_info_classif(x_cat, sample_y, discrete_features=True, random_state=seed)
        if n_categorical
        else np.empty(0, dtype=np.float64)
    )
    environment_values = np.unique(time_group)
    numeric_environment_effects: List[np.ndarray] = []
    categorical_environment_effects: List[np.ndarray] = []
    for environment in environment_values:
        mask = time_group == environment
        local_y = sample_y[mask]
        if local_y.sum() < 3 or len(local_y) - local_y.sum() < 30:
            continue
        local_num = x_num[mask]
        if n_numeric:
            local_difference = (
                local_num[local_y == 1].mean(axis=0) - local_num[local_y == 0].mean(axis=0)
            ) / np.maximum(local_num.std(axis=0), EPS)
            numeric_environment_effects.append(
                np.nan_to_num(local_difference, nan=0.0, posinf=0.0, neginf=0.0)
            )
        if n_categorical:
            categorical_environment_effects.append(
                np.asarray(
                    [
                        _categorical_rate_dispersion(x_cat[mask, index], local_y)
                        for index in range(n_categorical)
                    ],
                    dtype=np.float64,
                )
            )
    if numeric_environment_effects:
        environment_matrix = np.stack(numeric_environment_effects, axis=0)
        global_sign = np.sign(x_num[sample_y == 1].mean(axis=0) - x_num[sample_y == 0].mean(axis=0))
        sign_agreement = np.mean(np.sign(environment_matrix) == global_sign[None, :], axis=0)
        magnitude = np.median(np.abs(environment_matrix), axis=0)
        numeric_stability = magnitude * sign_agreement
    else:
        numeric_stability = numeric_effect.copy()
    if categorical_environment_effects:
        environment_matrix = np.stack(categorical_environment_effects, axis=0)
        categorical_stability = environment_matrix.mean(axis=0) / (
            environment_matrix.mean(axis=0) + environment_matrix.std(axis=0) + EPS
        )
    else:
        categorical_stability = categorical_effect.copy()
    effect = np.concatenate([numeric_effect, categorical_effect])
    mutual_information = np.concatenate([numeric_mi, categorical_mi])
    embedded_model = np.concatenate(
        [numeric_logistic, 0.5 * categorical_effect + 0.5 * categorical_mi]
    )
    stability = np.concatenate([numeric_stability, categorical_stability])
    evidence_matrix = np.column_stack(
        [
            _rank_unit_interval(effect),
            _rank_unit_interval(mutual_information),
            _rank_unit_interval(embedded_model),
            _rank_unit_interval(stability),
        ]
    )
    evidence_weights = np.asarray([0.3, 0.25, 0.2, 0.25], dtype=np.float64)
    consensus = evidence_matrix @ evidence_weights
    if use_evidence_prior:
        prior = prior_min + (prior_max - prior_min) * consensus
    else:
        prior = np.full(len(consensus), 0.5, dtype=np.float64)
    field_names = list(numeric_names) + list(categorical_names)
    field_types = ["numeric"] * n_numeric + ["categorical"] * n_categorical
    group_names_by_field = [
        mecg_semantic_group(name, categorical=field_type == "categorical")
        for name, field_type in zip(field_names, field_types)
    ]
    unique_groups = list(dict.fromkeys(group_names_by_field))
    group_lookup = {name: index for index, name in enumerate(unique_groups)}
    group_ids = np.asarray([group_lookup[name] for name in group_names_by_field], dtype=np.int64)
    group_prior = np.asarray(
        [float(np.mean(prior[group_ids == index])) for index in range(len(unique_groups))],
        dtype=np.float64,
    )
    redundancy = np.zeros((n_numeric, n_numeric), dtype=np.float32)
    if n_numeric > 1:
        corr_rows = sample
        if len(corr_rows) > max_rows:
            corr_rows = rng.choice(corr_rows, size=max_rows, replace=False)
        correlation_values = np.asarray(arrays.x_numeric[corr_rows], dtype=np.float64)
        correlation = np.zeros((n_numeric, n_numeric), dtype=np.float64)
        variable_columns = np.flatnonzero(
            np.isfinite(correlation_values).all(axis=0) & (correlation_values.std(axis=0) > EPS)
        )
        if len(variable_columns) > 1:
            variable_correlation = np.corrcoef(
                correlation_values[:, variable_columns], rowvar=False
            )
            correlation[np.ix_(variable_columns, variable_columns)] = np.nan_to_num(
                np.abs(variable_correlation), nan=0.0
            )
        np.fill_diagonal(correlation, 0.0)
        redundancy = np.clip(
            (correlation - redundancy_threshold) / max(1.0 - redundancy_threshold, EPS), 0.0, 1.0
        ).astype(np.float32)
    records = []
    for index, (name, field_type, group_name) in enumerate(
        zip(field_names, field_types, group_names_by_field)
    ):
        records.append(
            {
                "field_index": index,
                "feature": name,
                "type": field_type,
                "semantic_group": group_name,
                "class_separation": float(effect[index]),
                "mutual_information": float(mutual_information[index]),
                "embedded_model_evidence": float(embedded_model[index]),
                "temporal_stability": float(stability[index]),
                "consensus_score": float(consensus[index]),
                "initial_open_probability": float(prior[index]),
            }
        )
    return {
        "field_names": field_names,
        "field_types": field_types,
        "group_names": unique_groups,
        "group_ids": group_ids.tolist(),
        "feature_prior": prior.astype(np.float32).tolist(),
        "group_prior": group_prior.astype(np.float32).tolist(),
        "numeric_redundancy": redundancy.tolist(),
        "records": records,
        "fit_rows": int(len(sample)),
        "fit_positives": int(sample_y.sum()),
        "environments_used": int(len(numeric_environment_effects)),
        "uses_only_training_partition": True,
        "evidence_prior_enabled": bool(use_evidence_prior),
        "redundancy_threshold": float(redundancy_threshold),
    }
