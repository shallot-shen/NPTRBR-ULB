from __future__ import annotations
import argparse
import copy
import json
import logging
import math
import pickle
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from torch.utils.data import DataLoader

LOGGER = logging.getLogger("mecg_star_tabm")
EPS = 1e-6
MISSING_TOKEN = "__MISSING__"

from .preprocessing import PublicPreprocessor, PreparedArrays
from .model import ModelConfig, MECGSTaRTabM, build_model
from .sampling import ArrayDataset, MemberBaggingDataset, DynamicNegativeSampler
from .losses import (
    temporal_robustness,
    topk_budget_recall_loss,
    nptrbr_augmented_loss,
    topk_recall_lambda_for_epoch,
)
from .evidence import compute_mecg_train_evidence
from .data import load_training_frame, fixed_split_for_training
from .training_state import ensure_finite, save_epoch_checkpoint, load_epoch_checkpoint, restore_rng

QUICK_CHECK_NUMERIC = [f"V{i}" for i in range(1, 29)] + ["Amount"]
FULL_LAB_NUMERIC = []
CLIMATE_AVAILABILITY_COLUMN = "__unavailable_context__"


@dataclass(frozen=True)
class DisabledHistoryConfig:
    enabled: bool = False
    label_delay_days: float = 1.0


def add_history_to_training_splits(
    args,
    train_df,
    y_train,
    early_val_df,
    y_early_val,
    calibration_df,
    y_calibration,
    test_df,
    y_test,
):
    return (train_df, early_val_df, calibration_df, test_df, DisabledHistoryConfig(), "disabled")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_grad_scaler(enabled: bool) -> Any:
    """Use the current AMP API while retaining compatibility with older PyTorch."""
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def autocast_context(device: torch.device, enabled: bool) -> Any:
    try:
        return torch.amp.autocast(device_type=device.type, enabled=enabled)
    except AttributeError:
        return torch.cuda.amp.autocast(enabled=enabled)


def cpu_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def parse_float_list(text: str, name: str) -> List[float]:
    try:
        values = [float(item.strip()) for item in text.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError(f"{name}Public benchmark configuration error") from exc
    if not values:
        raise ValueError(f"{name}Public benchmark configuration error")
    return values


def split_validation_for_calibration(
    validation_indices: np.ndarray,
    y: np.ndarray,
    timestamps: pd.Series,
    calibration_fraction: float,
    min_positives: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Chronologically separate early stopping from calibration/thresholding.

    The requested minimum is relaxed only when the validation period does not
    contain enough positives to satisfy it in both windows. If a chronological
    two-class split is impossible, the complete validation period is reused and
    the fallback is recorded explicitly in the returned metadata.
    """
    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError("--calibration-fraction-of-validation   (0, 1)")
    if min_positives < 1:
        raise ValueError("--calibration-min-positives  1")
    ts = pd.to_datetime(timestamps, errors="coerce")
    ordered = np.asarray(validation_indices, dtype=np.int64)
    ordered = ordered[np.argsort(ts.iloc[ordered].to_numpy(), kind="stable")]
    n = len(ordered)
    if n < 4:
        return (
            ordered,
            ordered,
            {
                "strategy": "shared_validation_fallback",
                "reason": "validation_too_small",
                "calibration_fraction_requested": calibration_fraction,
            },
        )
    labels = y[ordered].astype(np.int64, copy=False)
    total_positive = int(labels.sum())
    target_cut = int(round(n * (1.0 - calibration_fraction)))
    target_cut = min(max(target_cut, 1), n - 1)
    cuts = np.arange(1, n, dtype=np.int64)
    positive_prefix = np.cumsum(labels)[:-1]
    left_positive = positive_prefix
    right_positive = total_positive - positive_prefix
    left_negative = cuts - left_positive
    right_negative = n - cuts - right_positive
    feasible_minimum = min(min_positives, total_positive // 2)
    valid = (
        (left_positive >= feasible_minimum)
        & (right_positive >= feasible_minimum)
        & (left_negative > 0)
        & (right_negative > 0)
        if feasible_minimum >= 1
        else np.zeros_like(cuts, dtype=bool)
    )
    if np.any(valid):
        candidate_cuts = cuts[valid]
        cut = int(candidate_cuts[np.argmin(np.abs(candidate_cuts - target_cut))])
        selection_idx, calibration_idx = (ordered[:cut], ordered[cut:])
        info = {
            "strategy": "chronological_early_stop_then_calibration",
            "calibration_fraction_requested": calibration_fraction,
            "calibration_fraction_actual": len(calibration_idx) / n,
            "minimum_positives_requested": min_positives,
            "minimum_positives_used": feasible_minimum,
            "early_stop_start": str(ts.iloc[selection_idx].min()),
            "early_stop_end": str(ts.iloc[selection_idx].max()),
            "calibration_start": str(ts.iloc[calibration_idx].min()),
            "calibration_end": str(ts.iloc[calibration_idx].max()),
        }
        return (selection_idx, calibration_idx, info)
    LOGGER.warning("Public benchmark configuration error")
    return (
        ordered,
        ordered,
        {
            "strategy": "shared_validation_fallback",
            "reason": "insufficient_class_support",
            "calibration_fraction_requested": calibration_fraction,
            "validation_positives": total_positive,
        },
    )


def validate_split_labels(y: np.ndarray, indices: Mapping[str, np.ndarray]) -> None:
    for name, idx in indices.items():
        positives = int(y[idx].sum())
        negatives = int(len(idx) - positives)
        LOGGER.info(
            "%s: rows=%s positives=%s (%.4f%%) negatives=%s",
            name,
            f"{len(idx):,}",
            f"{positives:,}",
            100.0 * positives / max(len(idx), 1),
            f"{negatives:,}",
        )
        if positives == 0 or negatives == 0:
            raise ValueError(f"{name}Public benchmark configuration error")


@torch.no_grad()
def predict_probabilities(
    model: nn.Module,
    arrays: PreparedArrays,
    device: torch.device,
    batch_size: int,
    indices: Optional[np.ndarray] = None,
    return_masks: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    model.eval()
    if indices is None:
        indices = np.arange(len(arrays), dtype=np.int64)
    mean_prob: List[np.ndarray] = []
    std_prob: List[np.ndarray] = []
    mask_sum: Optional[np.ndarray] = None
    mask_count = 0
    for start in range(0, len(indices), batch_size):
        idx = indices[start : start + batch_size]
        x_num = torch.from_numpy(arrays.x_numeric[idx]).to(device)
        x_cat = torch.from_numpy(arrays.x_categorical[idx]).to(device)
        x_ctx = torch.from_numpy(arrays.x_context[idx]).to(device)
        x_history = torch.from_numpy(arrays.x_history[idx]).to(device)
        logits, _, masks = model(x_num, x_cat, x_ctx, x_history, return_masks=return_masks)
        ensure_finite("inference logits", logits)
        member_prob = torch.sigmoid(logits)
        mean_prob.append(member_prob.mean(dim=1).cpu().numpy())
        std_prob.append(member_prob.std(dim=1, unbiased=False).cpu().numpy())
        if return_masks and masks is not None:
            current = masks.sum(dim=0).cpu().numpy()
            mask_sum = current if mask_sum is None else mask_sum + current
            mask_count += masks.size(0)
    mask_mean = None if mask_sum is None else mask_sum / max(mask_count, 1)
    return (np.concatenate(mean_prob), np.concatenate(std_prob), mask_mean)


@torch.no_grad()
def predict_checkpoint_ensemble(
    model: nn.Module,
    state_dicts: Sequence[Mapping[str, torch.Tensor]],
    arrays: PreparedArrays,
    device: torch.device,
    batch_size: int,
    indices: Optional[np.ndarray] = None,
    return_masks: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Average checkpoint predictions without interpolating neural-network weights.

    The returned uncertainty combines TabM member disagreement within each
    checkpoint and disagreement between checkpoints (law of total variance).
    """
    if not state_dicts:
        raise ValueError("checkpoint")
    checkpoint_probabilities: List[np.ndarray] = []
    within_checkpoint_variances: List[np.ndarray] = []
    checkpoint_masks: List[np.ndarray] = []
    for state_dict in state_dicts:
        model.load_state_dict(state_dict)
        probability, member_std, mask_mean = predict_probabilities(
            model=model,
            arrays=arrays,
            device=device,
            batch_size=batch_size,
            indices=indices,
            return_masks=return_masks,
        )
        checkpoint_probabilities.append(probability)
        within_checkpoint_variances.append(np.square(member_std))
        if mask_mean is not None:
            checkpoint_masks.append(mask_mean)
    probability_stack = np.stack(checkpoint_probabilities, axis=0)
    within_variance_stack = np.stack(within_checkpoint_variances, axis=0)
    probability_mean = probability_stack.mean(axis=0)
    total_variance = within_variance_stack.mean(axis=0) + probability_stack.var(axis=0)
    total_std = np.sqrt(np.maximum(total_variance, 0.0))
    mask_mean = np.stack(checkpoint_masks, axis=0).mean(axis=0) if checkpoint_masks else None
    return (probability_mean, total_std, mask_mean)


def checkpoint_state_dicts(checkpoint: Mapping[str, Any]) -> List[Mapping[str, torch.Tensor]]:
    """Read checkpoint prediction ensembles while retaining older artifacts."""
    states = checkpoint.get("checkpoint_state_dicts")
    if states:
        return list(states)
    if "model_state_dict" not in checkpoint:
        raise KeyError("checkpoint_state_dicts model_state_dict")
    return [checkpoint["model_state_dict"]]


@torch.no_grad()
def export_mecg_checkpoint_selection(
    model: nn.Module,
    state_dicts: Sequence[Mapping[str, torch.Tensor]],
    config: ModelConfig,
    output_path: Path,
    selection_threshold: float,
) -> Optional[Dict[str, Any]]:
    if not isinstance(model, MECGSTaRTabM):
        return None
    feature_probabilities: List[np.ndarray] = []
    group_probabilities: List[np.ndarray] = []
    combined_probabilities: List[np.ndarray] = []
    for state_dict in state_dicts:
        model.load_state_dict(state_dict)
        feature, group, combined = model.mecg_gate.probabilities()
        feature_probabilities.append(feature.detach().cpu().numpy())
        group_probabilities.append(group.detach().cpu().numpy())
        combined_probabilities.append(combined.detach().cpu().numpy())
    feature_stack = np.stack(feature_probabilities, axis=0)
    group_stack = np.stack(group_probabilities, axis=0)
    combined_stack = np.stack(combined_probabilities, axis=0)
    numeric_names = config.mecg_numeric_names or []
    categorical_names = config.mecg_categorical_names or []
    field_names = numeric_names + categorical_names
    field_types = ["numeric"] * len(numeric_names) + ["categorical"] * len(categorical_names)
    group_names = config.mecg_group_names or []
    group_ids = config.mecg_group_ids or []
    prior = np.asarray(config.mecg_feature_prior or [], dtype=np.float64)
    mean_probability = combined_stack.mean(axis=0)
    std_probability = combined_stack.std(axis=0)
    records: List[Dict[str, Any]] = []
    for index, (name, field_type, group_id) in enumerate(zip(field_names, field_types, group_ids)):
        records.append(
            {
                "field_index": index,
                "feature": name,
                "type": field_type,
                "semantic_group": group_names[int(group_id)],
                "evidence_prior": float(prior[index]),
                "checkpoint_gate_mean": float(mean_probability[index]),
                "checkpoint_gate_std": float(std_probability[index]),
                "selected": bool(mean_probability[index] >= selection_threshold),
            }
        )
    pd.DataFrame(records).sort_values(
        ["selected", "checkpoint_gate_mean"], ascending=[False, False]
    ).to_csv(output_path, index=False, encoding="utf-8-sig")
    group_mean = group_stack.mean(axis=0)
    group_std = group_stack.std(axis=0)
    group_records = [
        {
            "semantic_group": name,
            "checkpoint_gate_mean": float(group_mean[index]),
            "checkpoint_gate_std": float(group_std[index]),
        }
        for index, name in enumerate(group_names)
    ]
    return {
        "selection_threshold": float(selection_threshold),
        "selected_fields": int(np.sum(mean_probability >= selection_threshold)),
        "total_fields": int(len(mean_probability)),
        "active_probability_mean": float(mean_probability.mean()),
        "active_probability_min": float(mean_probability.min()),
        "active_probability_max": float(mean_probability.max()),
        "checkpoint_stability_mean_std": float(std_probability.mean()),
        "groups": group_records,
        "records": records,
    }


def refresh_hard_bank(
    model: nn.Module,
    arrays: PreparedArrays,
    negative_indices: np.ndarray,
    device: torch.device,
    prediction_batch_size: int,
    pool_size: int,
    bank_size: int,
    seed: int,
) -> np.ndarray:
    if pool_size <= 0 or bank_size <= 0:
        return np.empty(0, dtype=np.int64)
    rng = np.random.default_rng(seed)
    candidate = rng.choice(
        negative_indices, size=min(pool_size, len(negative_indices)), replace=False
    )
    score, _, _ = predict_probabilities(
        model, arrays, device, prediction_batch_size, indices=candidate
    )
    keep = min(bank_size, len(candidate))
    top = (
        np.argpartition(score, -keep)[-keep:]
        if keep < len(candidate)
        else np.arange(len(candidate))
    )
    return candidate[top[np.argsort(score[top])[::-1]]]


@torch.no_grad()
def estimate_topk_budget_state(
    model: nn.Module,
    arrays: PreparedArrays,
    y_train: np.ndarray,
    budgets: Sequence[float],
    base_weights: Sequence[float],
    device: torch.device,
    prediction_batch_size: int,
    pool_size: int,
    seed: int,
) -> Dict[str, Any]:
    """Estimate score cutoffs and deficit-adaptive weights on training rows."""
    if len(budgets) != len(base_weights):
        raise ValueError("Top-K")
    if pool_size < 1:
        raise ValueError("Top-K 1")
    rng = np.random.default_rng(seed)
    all_indices = np.arange(len(arrays), dtype=np.int64)
    reference_indices = rng.choice(
        all_indices, size=min(pool_size, len(all_indices)), replace=False
    )
    reference_probability, _, _ = predict_probabilities(
        model, arrays, device, prediction_batch_size, indices=reference_indices
    )
    positive_indices = np.flatnonzero(np.asarray(y_train) == 1).astype(np.int64)
    positive_probability, _, _ = predict_probabilities(
        model, arrays, device, prediction_batch_size, indices=positive_indices
    )
    thresholds = np.asarray(
        [float(np.quantile(reference_probability, 1.0 - budget)) for budget in budgets],
        dtype=np.float64,
    )
    recalls = np.asarray(
        [float(np.mean(positive_probability >= threshold)) for threshold in thresholds],
        dtype=np.float64,
    )
    base = np.asarray(base_weights, dtype=np.float64)
    adaptive = base * np.maximum(1.0 - recalls, 0.05)
    adaptive = adaptive / max(float(adaptive.sum()), EPS)
    threshold_logits = np.log(
        np.clip(thresholds, EPS, 1.0 - EPS) / np.clip(1.0 - thresholds, EPS, 1.0 - EPS)
    )
    return {
        "budgets": [float(value) for value in budgets],
        "threshold_probabilities": thresholds.tolist(),
        "threshold_logits": threshold_logits.tolist(),
        "training_positive_recalls": recalls.tolist(),
        "adaptive_weights": adaptive.tolist(),
        "reference_rows": int(len(reference_indices)),
        "positive_rows": int(len(positive_indices)),
    }


@torch.no_grad()
def estimate_nptrbr_budget_state(
    model: nn.Module,
    arrays: PreparedArrays,
    y_train: np.ndarray,
    budgets: Sequence[float],
    base_weights: Sequence[float],
    fpr_slack: float,
    device: torch.device,
    prediction_batch_size: int,
    pool_size: int,
    seed: int,
) -> Dict[str, Any]:
    """Estimate NP-TRBR cutoffs from a training-only negative reference pool."""
    if len(budgets) != len(base_weights):
        raise ValueError("NP-TRBR budgets and weights must have equal length.")
    negative_indices = np.flatnonzero(np.asarray(y_train) == 0).astype(np.int64)
    positive_indices = np.flatnonzero(np.asarray(y_train) == 1).astype(np.int64)
    if len(negative_indices) < 2 or len(positive_indices) < 1:
        raise ValueError("NP-TRBR requires both positive and negative training rows.")
    rng = np.random.default_rng(seed)
    reference_indices = rng.choice(
        negative_indices, size=min(int(pool_size), len(negative_indices)), replace=False
    )
    negative_probability, _, _ = predict_probabilities(
        model, arrays, device, prediction_batch_size, indices=reference_indices
    )
    positive_probability, _, _ = predict_probabilities(
        model, arrays, device, prediction_batch_size, indices=positive_indices
    )
    thresholds = np.asarray(
        [float(np.quantile(negative_probability, 1.0 - budget)) for budget in budgets],
        dtype=np.float64,
    )
    hard_recalls = np.asarray(
        [float(np.mean(positive_probability >= threshold)) for threshold in thresholds],
        dtype=np.float64,
    )
    threshold_logits = np.log(
        np.clip(thresholds, EPS, 1.0 - EPS) / np.clip(1.0 - thresholds, EPS, 1.0 - EPS)
    )
    weights = np.asarray(base_weights, dtype=np.float64)
    weights = weights / max(float(weights.sum()), EPS)
    return {
        "budgets": [float(value) for value in budgets],
        "weights": weights.tolist(),
        "fpr_targets": [float(value * (1.0 + fpr_slack)) for value in budgets],
        "threshold_probabilities": thresholds.tolist(),
        "threshold_logits": threshold_logits.tolist(),
        "training_positive_hard_recalls": hard_recalls.tolist(),
        "negative_reference_rows": int(len(reference_indices)),
        "positive_rows": int(len(positive_indices)),
    }


class ProbabilityCalibrator:
    def __init__(self, method: str = "platt") -> None:
        self.method = method
        self.model: Any = None

    @staticmethod
    def _logit(probability: np.ndarray) -> np.ndarray:
        p = np.clip(probability, 1e-06, 1.0 - 1e-06)
        return np.log(p / (1.0 - p))

    def fit(self, probability: np.ndarray, y: np.ndarray) -> "ProbabilityCalibrator":
        if self.method == "none":
            return self
        if len(np.unique(y)) < 2:
            LOGGER.warning("Public benchmark configuration error")
            self.method = "none"
            return self
        if self.method == "platt":
            self.model = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
            self.model.fit(self._logit(probability).reshape(-1, 1), y)
        elif self.method == "isotonic":
            self.model = IsotonicRegression(out_of_bounds="clip")
            self.model.fit(probability, y)
        else:
            raise ValueError(f":{self.method}")
        return self

    def predict(self, probability: np.ndarray) -> np.ndarray:
        if self.method == "none" or self.model is None:
            return np.clip(probability, 0.0, 1.0)
        if self.method == "platt":
            return self.model.predict_proba(self._logit(probability).reshape(-1, 1))[:, 1]
        return np.asarray(self.model.predict(probability), dtype=np.float64)


def select_threshold(
    y: np.ndarray,
    probability: np.ndarray,
    strategy: str,
    beta: float,
    target_recall: float,
    alert_fraction: float,
) -> Tuple[float, Dict[str, Any]]:
    if strategy == "topk":
        if not 0.0 < alert_fraction <= 1.0:
            raise ValueError("--alert-fraction   (0, 1]")
        k = min(max(int(math.ceil(len(probability) * alert_fraction)), 1), len(probability))
        order = np.argsort(probability, kind="stable")[::-1]
        threshold = float(probability[order[k - 1]])
        prediction = probability >= threshold
        tp = int(y[prediction].sum())
        predicted = int(prediction.sum())
        return (
            threshold,
            {
                "strategy": "topk",
                "alert_fraction_requested": float(alert_fraction),
                "alert_fraction_actual": float(predicted / max(len(y), 1)),
                "validation_precision": float(tp / max(predicted, 1)),
                "validation_recall": float(tp / max(int(y.sum()), 1)),
                "k_requested": int(k),
                "predicted_positive_count": predicted,
            },
        )
    precision, recall, thresholds = precision_recall_curve(y, probability)
    if len(thresholds) == 0:
        return (0.5, {"validation_precision": 0.0, "validation_recall": 0.0})
    p, r = (precision[:-1], recall[:-1])
    if strategy == "recall":
        valid = np.flatnonzero(r >= target_recall)
        if len(valid):
            best_precision = np.nanmax(p[valid])
            candidates = valid[np.isclose(p[valid], best_precision)]
            index = int(candidates[np.argmax(thresholds[candidates])])
        else:
            index = int(np.argmax(r))
    else:
        beta2 = beta * beta
        fbeta = (1.0 + beta2) * p * r / np.maximum(beta2 * p + r, EPS)
        index = int(np.nanargmax(fbeta))
    threshold = float(thresholds[index])
    info = {
        "validation_precision": float(p[index]),
        "validation_recall": float(r[index]),
        "strategy": strategy,
        "beta": float(beta),
        "target_recall": float(target_recall),
        "alert_fraction": float(alert_fraction),
    }
    return (threshold, info)


def binary_metrics(y: np.ndarray, probability: np.ndarray, threshold: float) -> Dict[str, Any]:
    prediction = (probability >= threshold).astype(np.int64)
    cm = confusion_matrix(y, prediction, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    prevalence = float(np.mean(y))
    average_precision = float(average_precision_score(y, probability))

    def metrics_from_counts(
        local_tn: int, local_fp: int, local_fn: int, local_tp: int
    ) -> Dict[str, float]:
        precision = local_tp / max(local_tp + local_fp, 1)
        recall = local_tp / max(local_tp + local_fn, 1)
        specificity = local_tn / max(local_tn + local_fp, 1)
        npv = local_tn / max(local_tn + local_fn, 1)
        f1 = 2.0 * precision * recall / max(precision + recall, EPS)
        total = local_tn + local_fp + local_fn + local_tp
        mcc_denominator = math.sqrt(
            max(
                (local_tp + local_fp)
                * (local_tp + local_fn)
                * (local_tn + local_fp)
                * (local_tn + local_fn),
                0,
            )
        )
        return {
            "precision": float(precision),
            "recall": float(recall),
            "sensitivity": float(recall),
            "specificity": float(specificity),
            "npv": float(npv),
            "f1": float(f1),
            "accuracy": float((local_tp + local_tn) / max(total, 1)),
            "balanced_accuracy": float(0.5 * (recall + specificity)),
            "false_positive_rate": float(local_fp / max(local_fp + local_tn, 1)),
            "false_negative_rate": float(local_fn / max(local_fn + local_tp, 1)),
            "false_discovery_rate": float(local_fp / max(local_fp + local_tp, 1)),
            "mcc": float((local_tp * local_tn - local_fp * local_fn) / max(mcc_denominator, EPS)),
        }

    def quantiles(values: np.ndarray) -> Dict[str, float]:
        if len(values) == 0:
            return {}
        points = np.asarray([0.0, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0])
        result = np.quantile(values, points)
        return {
            f"q{int(round(point * 100)):02d}": float(value) for point, value in zip(points, result)
        }

    top_fraction: Dict[str, Dict[str, float | int]] = {}
    order = np.argsort(probability, kind="stable")[::-1]
    total_positive = int(y.sum())
    for fraction in (0.001, 0.005, 0.01, 0.02, 0.05, 0.1):
        k = min(max(int(math.ceil(len(y) * fraction)), 1), len(y))
        positives_at_k = int(y[order[:k]].sum())
        top_tp = positives_at_k
        top_fp = int(k - top_tp)
        top_fn = int(total_positive - top_tp)
        top_tn = int(len(y) - total_positive - top_fp)
        top_metrics = metrics_from_counts(top_tn, top_fp, top_fn, top_tp)
        key = f"{100.0 * fraction:g}%"
        top_fraction[key] = {
            "k": k,
            "positives": positives_at_k,
            "alert_fraction": float(fraction),
            "tn": top_tn,
            "fp": top_fp,
            "fn": top_fn,
            "tp": top_tp,
            **top_metrics,
            "precision_lift": top_metrics["precision"] / max(prevalence, EPS),
        }
    threshold_metrics = metrics_from_counts(int(tn), int(fp), int(fn), int(tp))
    result: Dict[str, Any] = {
        "threshold": float(threshold),
        "prevalence": prevalence,
        "average_precision": average_precision,
        "average_precision_lift": average_precision / max(prevalence, EPS),
        "roc_auc": float(roc_auc_score(y, probability)) if len(np.unique(y)) == 2 else float("nan"),
        **threshold_metrics,
        "brier": float(brier_score_loss(y, probability)),
        "predicted_positive_count": int(prediction.sum()),
        "predicted_positive_rate": float(prediction.mean()),
        "probability_quantiles": {
            "all": quantiles(probability),
            "positive": quantiles(probability[y == 1]),
            "negative": quantiles(probability[y == 0]),
        },
        "top_fraction": top_fraction,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    return result


def save_complete_metric_tables(
    output_dir: Path, split_metrics: Mapping[str, Optional[Mapping[str, Any]]]
) -> None:
    """Write paper-ready operating-point and confusion-matrix tables.

    The calibrated threshold and every fixed alert budget are deliberately
    separate rows.  This prevents a Top-1% confusion matrix from being
    mistaken for a probability-threshold confusion matrix.
    """
    rows: List[Dict[str, Any]] = []
    for split_name, metrics in split_metrics.items():
        if metrics is None:
            continue
        common = {
            "split": split_name,
            "rows": int(metrics["tn"] + metrics["fp"] + metrics["fn"] + metrics["tp"]),
            "positives": int(metrics["tp"] + metrics["fn"]),
            "prevalence": float(metrics["prevalence"]),
            "auprc": float(metrics["average_precision"]),
            "auprc_lift": float(metrics["average_precision_lift"]),
            "roc_auc": float(metrics["roc_auc"]),
            "brier": float(metrics["brier"]),
        }
        threshold_row = {
            **common,
            "operating_point": "calibrated_threshold",
            "alert_fraction": float(metrics["predicted_positive_rate"]),
            "threshold": float(metrics["threshold"]),
        }
        for key in (
            "tn",
            "fp",
            "fn",
            "tp",
            "precision",
            "recall",
            "sensitivity",
            "specificity",
            "npv",
            "f1",
            "accuracy",
            "balanced_accuracy",
            "false_positive_rate",
            "false_negative_rate",
            "false_discovery_rate",
            "mcc",
        ):
            threshold_row[key] = metrics[key]
        rows.append(threshold_row)
        for fraction_name, values in metrics["top_fraction"].items():
            row = {
                **common,
                "operating_point": f"top_{fraction_name}",
                "alert_fraction": float(values["alert_fraction"]),
                "threshold": float("nan"),
            }
            for key in (
                "tn",
                "fp",
                "fn",
                "tp",
                "precision",
                "recall",
                "sensitivity",
                "specificity",
                "npv",
                "f1",
                "accuracy",
                "balanced_accuracy",
                "false_positive_rate",
                "false_negative_rate",
                "false_discovery_rate",
                "mcc",
            ):
                row[key] = values[key]
            rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(
        output_dir / "complete_metrics_and_confusion_matrices.csv",
        index=False,
        encoding="utf-8-sig",
    )
    if not table.empty:
        table[["split", "operating_point", "rows", "positives", "tn", "fp", "fn", "tp"]].to_csv(
            output_dir / "confusion_matrices.csv", index=False, encoding="utf-8-sig"
        )


def log_classification_summary(name: str, metrics: Mapping[str, Any]) -> None:
    """Print operating-point metrics and an unambiguous 2x2 confusion matrix."""
    LOGGER.info(
        "%s | Precision %.6f | Accuracy %.6f | Recall %.6f",
        name,
        metrics["precision"],
        metrics["accuracy"],
        metrics["recall"],
    )
    LOGGER.info(
        "%s  | [[TN=%d, FP=%d], [FN=%d, TP=%d]]",
        name,
        metrics["tn"],
        metrics["fp"],
        metrics["fn"],
        metrics["tp"],
    )
    top_parts = []
    for key in ("0.1%", "0.5%", "1%", "2%", "5%", "10%"):
        item = metrics["top_fraction"][key]
        top_parts.append(f"Top{key}={item['positives']}/{item['k']},Recall={item['recall']:.4f}")
    LOGGER.info("%s Top-K  | %s", name, " | ".join(top_parts))


def base_feature_groups(preprocessor: PublicPreprocessor) -> List[Tuple[str, int, int]]:
    groups: List[Tuple[str, int, int]] = []
    cursor = 0
    for col in preprocessor.numeric_columns:
        groups.append((col, cursor, cursor + 1))
        cursor += 1
    for col, cardinality in zip(
        preprocessor.categorical_columns, preprocessor.categorical_cardinalities
    ):
        dim = embedding_dimension(cardinality)
        groups.append((f"CAT__{col}", cursor, cursor + dim))
        cursor += dim
    return groups


def aggregate_mask_importance(
    mask_mean: np.ndarray, preprocessor: PublicPreprocessor
) -> pd.DataFrame:
    importance = mask_mean.mean(axis=0)
    rows = []
    for name, start, end in base_feature_groups(preprocessor):
        rows.append({"feature": name, "mask_importance": float(importance[start:end].sum())})
    result = pd.DataFrame(rows).sort_values("mask_importance", ascending=False)
    total = result["mask_importance"].sum()
    if total > 0:
        result["mask_importance_normalized"] = result["mask_importance"] / total
    return result


def resolve_member_negative_ratio(args: argparse.Namespace) -> float:
    """Resolve the stable per-member ratio, with a narrow legacy fallback."""
    legacy = getattr(args, "negative_ratios", None)
    if legacy:
        values = [item.strip().lower() for item in legacy.split(",") if item.strip()]
        if len(values) != 1 or values[0] == "all":
            raise ValueError("all  --member-negative-ratio 3 3 5 10")
        LOGGER.warning("--negative-ratios   %s", values[0])
        ratio = float(values[0])
    else:
        ratio = float(args.member_negative_ratio)
    if ratio <= 0:
        raise ValueError("--member-negative-ratio  0")
    return ratio


def choose_device(text: str) -> torch.device:
    if text == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(text)
    if device.type == "cuda" and (not torch.cuda.is_available()):
        raise RuntimeError("CUDA")
    return device


def build_temporal_decay_weights(
    timestamps: Sequence[Any],
    y: np.ndarray,
    half_life_days: float,
    floor: float,
    class_normalize: bool,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Give recent training rows more influence without changing class balance.

    Raw weights follow ``floor + (1-floor) * 2**(-age / half_life)``.
    With class normalization (the default), positive and negative weights are
    normalized separately to mean one. Therefore the configured 1:r member
    sampling balance is preserved while recent rows within each class receive
    more influence than stale rows.
    """
    y = np.asarray(y, dtype=np.int64)
    if half_life_days <= 0:
        weights = np.ones(len(y), dtype=np.float32)
        return (
            weights,
            {
                "enabled": False,
                "half_life_days": float(half_life_days),
                "floor": float(floor),
                "class_normalize": bool(class_normalize),
            },
        )
    if not 0.0 < floor <= 1.0:
        raise ValueError("--time-decay-floor  (0, 1]")
    ts = pd.to_datetime(pd.Series(timestamps), errors="coerce")
    if len(ts) != len(y) or ts.isna().any():
        raise ValueError("Public benchmark configuration error")
    reference = ts.max()
    age_days = (reference - ts).dt.total_seconds().to_numpy(dtype=np.float64) / 86400.0
    age_days = np.maximum(age_days, 0.0)
    raw = floor + (1.0 - floor) * np.exp2(-age_days / half_life_days)
    weights = raw.copy()
    if class_normalize:
        for label in (0, 1):
            mask = y == label
            if mask.any():
                weights[mask] /= max(float(weights[mask].mean()), EPS)
    else:
        weights /= max(float(weights.mean()), EPS)
    weights = weights.astype(np.float32)

    def class_mean(label: int) -> float:
        mask = y == label
        return float(weights[mask].mean()) if mask.any() else float("nan")

    effective_size = float(np.square(weights.sum()) / max(float(np.square(weights).sum()), EPS))
    stats = {
        "enabled": True,
        "half_life_days": float(half_life_days),
        "floor": float(floor),
        "class_normalize": bool(class_normalize),
        "reference_timestamp": str(reference),
        "minimum": float(weights.min()),
        "median": float(np.median(weights)),
        "maximum": float(weights.max()),
        "positive_mean": class_mean(1),
        "negative_mean": class_mean(0),
        "effective_sample_size": effective_size,
    }
    return (weights, stats)


def safe_torch_load(path: Path, device: torch.device) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def embedding_dimension(cardinality: int) -> int:
    return int(min(32, max(4, round(4.0 * cardinality**0.25))))


def train_pipeline(args: argparse.Namespace) -> Dict[str, Any]:
    nptrbr_budgets = parse_float_list(args.nptrbr_budgets, "--nptrbr-budgets")
    nptrbr_weights = parse_float_list(args.nptrbr_weights, "--nptrbr-weights")
    nptrbr_weights = [v / float(sum(nptrbr_weights)) for v in nptrbr_weights]
    topk_budgets = [0.01, 0.02, 0.05]
    topk_base_weights = [0.5, 0.3, 0.2]
    seed_everything(args.seed)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    LOGGER.info("Device: %s", device)
    evaluate_test = args.run_mode == "final"
    LOGGER.info(
        "Run mode=%s | feature configuration=%s | protocol=%s",
        args.run_mode,
        "quick_check" if hasattr(args,"_transfer_protocol") else "public_ULB",
        getattr(args,"_transfer_protocol","fixed_row_order"),
    )
    df, y = load_training_frame(args)
    split_groups: Optional[np.ndarray] = None
    train_idx, val_idx, test_idx, split_info = fixed_split_for_training(args, df, y)
    early_val_idx = np.asarray(split_info.pop("_early_indices"), dtype=np.int64)
    calibration_idx = np.asarray(split_info.pop("_calibration_indices"), dtype=np.int64)
    calibration_split_info = split_info.pop("_calibration_info")
    split_info["validation_subsplit"] = calibration_split_info
    split_indices = {
        "training": train_idx,
        "early_validation": early_val_idx,
        "calibration": calibration_idx,
    }
    if evaluate_test:
        split_indices["test"] = test_idx
    validate_split_labels(y, split_indices)
    split_info["counts"] = {
        name: {
            "rows": int(len(indices)),
            "positives": int(y[indices].sum()),
            "negatives": int(len(indices) - y[indices].sum()),
            "prevalence": float(np.mean(y[indices])),
        }
        for name, indices in split_indices.items()
    }
    if not evaluate_test:
        split_info["test_lock"] = {
            "status": "locked_not_evaluated",
            "rows": int(len(test_idx)),
            "labels_and_metrics_hidden": True,
        }
        LOGGER.info(
            "Test locked: %s rows; no test predictions or metrics are computed.",
            f"{len(test_idx):,}",
        )
    manifest_split = np.full(len(df), "unused", dtype=object)
    manifest_split[train_idx] = "train"
    manifest_split[early_val_idx] = "early_validation"
    manifest_split[calibration_idx] = "calibration"
    manifest_split[test_idx] = "locked_test" if not evaluate_test else "test"
    pd.DataFrame({"source_index": df["_source_index"].to_numpy(), "split": manifest_split}).to_csv(
        output_dir / "split_manifest.csv", index=False, encoding="utf-8-sig"
    )
    train_df = df.iloc[train_idx].copy()
    early_val_df = df.iloc[early_val_idx].copy()
    calibration_df = df.iloc[calibration_idx].copy()
    y_train = y[train_idx]
    y_early_val = y[early_val_idx]
    y_calibration = y[calibration_idx]
    test_df = df.iloc[test_idx].copy() if evaluate_test else None
    y_test = y[test_idx] if evaluate_test else None
    del df, y
    train_df, early_val_df, calibration_df, test_df, history_config, history_protocol = (
        add_history_to_training_splits(
            args,
            train_df,
            y_train,
            early_val_df,
            y_early_val,
            calibration_df,
            y_calibration,
            test_df,
            y_test,
        )
    )
    preprocessor = PublicPreprocessor(
        timestamp_col=args.timestamp_col,
        category_min_frequency=args.category_min_frequency,
        max_categories=args.max_categories,
        prototype_tau=args.prototype_tau,
        use_altitude=False,
        use_history_risk=history_config.enabled,
        use_hstp=False,
        use_climate_features=False,
        time_fourier_harmonics=args.time_fourier_harmonics,
        use_absolute_time_trend=False,
        ple_bins=args.ple_bins,
        feature_stage="quick_check",
    )
    LOGGER.info("Fitting training-only public preprocessing | variant=%s", args.model_variant)
    train_arrays = preprocessor.fit_transform(train_df)
    early_val_arrays = preprocessor.transform(early_val_df)
    calibration_arrays = preprocessor.transform(calibration_df)
    test_arrays = preprocessor.transform(test_df) if test_df is not None else None
    train_sample_weight, time_decay_stats = build_temporal_decay_weights(
        train_df[args.timestamp_col],
        y_train,
        0.0,
        args.time_decay_floor,
        args.time_decay_class_normalize,
    )
    if time_decay_stats["enabled"]:
        LOGGER.info(
            "| half-life=%.1f  | floor=%.3f | class-normalize=%s | weight[min/median/max]=%.3f/%.3f/%.3f | ESS=%.0f",
            0.0,
            args.time_decay_floor,
            args.time_decay_class_normalize,
            time_decay_stats["minimum"],
            time_decay_stats["median"],
            time_decay_stats["maximum"],
            time_decay_stats["effective_sample_size"],
        )
    del train_df, early_val_df, calibration_df, test_df
    cardinalities = preprocessor.categorical_cardinalities
    embedding_dims = [embedding_dimension(cardinality) for cardinality in cardinalities]
    modulation_feature_names = set(QUICK_CHECK_NUMERIC + FULL_LAB_NUMERIC)
    modulation_feature_names.discard(CLIMATE_AVAILABILITY_COLUMN)
    numeric_modulation_mask = [
        column in modulation_feature_names for column in preprocessor.numeric_columns
    ]
    selected_modulation_features = [
        column
        for column, enabled in zip(preprocessor.numeric_columns, numeric_modulation_mask)
        if enabled
    ]
    if args.model_variant in {"star_tabm", "star_tabm_no_hstp", "mecg_star_tabm"}:
        if not selected_modulation_features:
            raise ValueError("feature-stage STaR / /")
        LOGGER.info(
            "Compatibility input fields=%d | %s",
            len(selected_modulation_features),
            ",".join(selected_modulation_features),
        )
        if (
            args.feature_modulation_scale == 0
            and args.feature_modulation_lambda_range == 0
            and (args.modulation_member_rank == 0)
        ):
            LOGGER.info(
                "Input FAM disabled: zero scales and member rank; parameter layout retained."
            )
    mecg_evidence: Optional[Dict[str, Any]] = None
    if args.model_variant == "mecg_star_tabm":
        LOGGER.info(
            "MECG  | evidence-prior=%s | group-gates=%s | absolute-time=%s",
            args.mecg_evidence_prior,
            args.mecg_group_gates,
            False,
        )
        mecg_evidence = compute_mecg_train_evidence(
            arrays=train_arrays,
            y=y_train,
            numeric_names=preprocessor.numeric_columns,
            categorical_names=preprocessor.categorical_columns,
            seed=args.seed + 71000,
            max_rows=args.mecg_evidence_max_rows,
            prior_min=args.mecg_prior_min,
            prior_max=args.mecg_prior_max,
            redundancy_threshold=args.mecg_redundancy_threshold,
            use_evidence_prior=args.mecg_evidence_prior,
        )
        with (output_dir / "mecg_train_evidence.json").open("w", encoding="utf-8") as handle:
            json.dump(to_jsonable(mecg_evidence), handle, ensure_ascii=False, indent=2)
        top_evidence = sorted(
            mecg_evidence["records"], key=lambda record: record["consensus_score"], reverse=True
        )[:10]
        LOGGER.info(
            "MECG : %s",
            ", ".join(
                (f"{record['feature']}={record['consensus_score']:.3f}" for record in top_evidence)
            ),
        )
    model_config = ModelConfig(
        n_numeric=train_arrays.x_numeric.shape[1],
        categorical_cardinalities=cardinalities,
        categorical_embedding_dims=embedding_dims,
        n_context=train_arrays.x_context.shape[1],
        hidden_dim=args.hidden_dim,
        ensemble_init=getattr(args, "ensemble_init", "legacy"),
        members=args.members,
        n_blocks=args.blocks,
        selector_steps=args.selector_steps,
        selector_gamma=args.selector_gamma,
        context_dim=args.context_dim,
        adapter_rank=args.adapter_rank,
        member_adapter_rank=args.member_adapter_rank,
        member_adapter_scale=args.member_adapter_scale,
        numeric_embedding_dim=args.numeric_embedding_dim,
        numeric_bin_edges=preprocessor.numeric_bin_edges
        if args.numeric_embedding_dim > 0
        else None,
        numeric_modulation_mask=numeric_modulation_mask,
        feature_modulation_scale=args.feature_modulation_scale,
        feature_modulation_lambda_range=args.feature_modulation_lambda_range,
        modulation_member_rank=args.modulation_member_rank,
        modulation_member_scale=args.modulation_member_scale,
        dropout=args.dropout,
        n_history=train_arrays.x_history.shape[1],
        history_hidden_dim=args.history_hidden_dim,
        branch_mode="static",
        model_variant=args.model_variant,
        mecg_enabled=args.mecg_enabled and args.model_variant == "mecg_star_tabm",
        mecg_numeric_names=list(preprocessor.numeric_columns)
        if mecg_evidence is not None
        else None,
        mecg_categorical_names=list(preprocessor.categorical_columns)
        if mecg_evidence is not None
        else None,
        mecg_group_names=list(mecg_evidence["group_names"]) if mecg_evidence is not None else None,
        mecg_group_ids=list(mecg_evidence["group_ids"]) if mecg_evidence is not None else None,
        mecg_feature_prior=list(mecg_evidence["feature_prior"])
        if mecg_evidence is not None
        else None,
        mecg_group_prior=list(mecg_evidence["group_prior"]) if mecg_evidence is not None else None,
        mecg_numeric_redundancy=list(mecg_evidence["numeric_redundancy"])
        if mecg_evidence is not None
        else None,
        mecg_temperature=args.mecg_temperature,
        mecg_stretch_low=args.mecg_stretch_low,
        mecg_stretch_high=args.mecg_stretch_high,
        mecg_group_gates=args.mecg_group_gates,
        mecg_stochastic_gates=args.mecg_stochastic_gates,
        mecg_target_active_ratio=args.mecg_target_active_ratio,
    )
    model = build_model(model_config).to(device)
    parameter_count = sum((parameter.numel() for parameter in model.parameters()))
    LOGGER.info(
        "Model=%s | parameters=%s | numeric=%d categorical=%d context=%d history=%d | branch=%s | members=%d",
        args.model_variant,
        f"{parameter_count:,}",
        model_config.n_numeric,
        len(cardinalities),
        model_config.n_context,
        model_config.n_history,
        model_config.branch_mode,
        model_config.members,
    )
    dataset = ArrayDataset(train_arrays, y_train, train_sample_weight)
    sampler_engine = DynamicNegativeSampler(
        y_train, train_arrays.matched_key, train_arrays.deviation_bin, seed=args.seed + 9000
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1), eta_min=args.learning_rate * 0.03
    )
    member_negative_ratio = resolve_member_negative_ratio(args)
    steady_mix = (args.mix_global + args.mix_prototype, args.mix_region, 0.0, args.mix_hard)
    warmup_mix = (0.7, 0.3, 0.0, 0.0)
    amp_enabled = bool(args.amp and device.type == "cuda")
    grad_scaler = make_grad_scaler(amp_enabled)
    positive_weight = torch.tensor(float(args.positive_loss_weight), device=device)
    if not math.isclose(float(args.positive_loss_weight), 1.0):
        LOGGER.warning("1:%.3g  --positive-loss-weight 1.0", member_negative_ratio)
    best_smoothed_ap = -np.inf
    best_epoch = -1
    top_checkpoints: List[Tuple[float, float, int, Dict[str, torch.Tensor]]] = []
    validation_ap_window: List[float] = []
    patience_counter = 0
    hard_bank: Optional[np.ndarray] = None
    topk_state: Optional[Dict[str, Any]] = None
    topk_threshold_logits: Optional[torch.Tensor] = None
    topk_adaptive_weights: Optional[torch.Tensor] = None
    topk_state_history: List[Dict[str, Any]] = []
    nptrbr_teacher: Optional[nn.Module] = None
    nptrbr_state: Optional[Dict[str, Any]] = None
    nptrbr_threshold_logits: Optional[torch.Tensor] = None
    nptrbr_weight_tensor = torch.as_tensor(nptrbr_weights, dtype=torch.float32, device=device)
    nptrbr_fpr_targets = torch.as_tensor(
        [value * (1.0 + args.nptrbr_fpr_slack) for value in nptrbr_budgets],
        dtype=torch.float32,
        device=device,
    )
    nptrbr_fpr_duals = torch.zeros(len(nptrbr_budgets), dtype=torch.float32, device=device)
    nptrbr_kl_dual = torch.zeros((), dtype=torch.float32, device=device)
    nptrbr_state_history: List[Dict[str, Any]] = []
    nptrbr_dual_history: List[Dict[str, Any]] = []
    history: List[Dict[str, Any]] = []
    start_time = time.time()
    start_epoch = 0
    if getattr(args, "_resume_training", False):
        request = json.loads((output_dir/"run_request.json").read_text(encoding="utf-8"))
        recovery = load_epoch_checkpoint(output_dir/"training_state.pt", request["request_sha256"])
        model.load_state_dict(recovery["model"])
        optimizer.load_state_dict(recovery["optimizer"])
        scheduler.load_state_dict(recovery["scheduler"])
        grad_scaler.load_state_dict(recovery["grad_scaler"])
        sampler_engine.restore_recovery_state(recovery["sampler"])
        restored = recovery["training"]
        best_smoothed_ap, best_epoch = restored["best_smoothed_ap"], restored["best_epoch"]
        top_checkpoints = restored["top_checkpoints"]
        validation_ap_window = restored["validation_ap_window"]
        patience_counter, hard_bank = restored["patience_counter"], restored["hard_bank"]
        topk_state, topk_state_history = restored["topk_state"], restored["topk_state_history"]
        topk_threshold_logits = restored["topk_threshold_logits"]
        topk_adaptive_weights = restored["topk_adaptive_weights"]
        if topk_threshold_logits is not None:
            topk_threshold_logits = topk_threshold_logits.to(device)
            topk_adaptive_weights = topk_adaptive_weights.to(device)
        nptrbr_state = restored["nptrbr_state"]
        nptrbr_threshold_logits = restored["nptrbr_threshold_logits"]
        if nptrbr_threshold_logits is not None:
            nptrbr_threshold_logits = nptrbr_threshold_logits.to(device)
        nptrbr_fpr_duals = restored["nptrbr_fpr_duals"].to(device)
        nptrbr_kl_dual = restored["nptrbr_kl_dual"].to(device)
        nptrbr_state_history, nptrbr_dual_history = restored["nptrbr_state_history"], restored["nptrbr_dual_history"]
        history = restored["history"]
        if recovery["teacher"] is not None:
            nptrbr_teacher = copy.deepcopy(model).to(device).eval()
            nptrbr_teacher.load_state_dict(recovery["teacher"])
            for parameter in nptrbr_teacher.parameters():
                parameter.requires_grad_(False)
        start_epoch = int(recovery["epoch"])
        start_time -= float(recovery["elapsed_seconds"])
        restore_rng(recovery["rng"])
        LOGGER.info("Recovered full training state after epoch %d; next epoch=%d", start_epoch, start_epoch+1)
    LOGGER.info("Numerical policy: forward AMP=%s; NP-TRBR probability/KL uses FP32; finite checks enabled", amp_enabled)
    for epoch in range(start_epoch, args.epochs):
        if patience_counter >= args.patience:
            break
        epoch_number = epoch + 1
        current_mecg_scale = (
            topk_recall_lambda_for_epoch(
                epoch_number, 1.0, args.mecg_warmup_epochs, args.mecg_ramp_epochs
            )
            if args.model_variant == "mecg_star_tabm" and args.mecg_enabled
            else 0.0
        )
        current_topk_lambda = topk_recall_lambda_for_epoch(
            epoch_number, 0.0, args.topk_recall_warmup, args.topk_recall_ramp_epochs
        )
        current_nptrbr_lambda = topk_recall_lambda_for_epoch(
            epoch_number, args.nptrbr_lambda, args.nptrbr_warmup, args.nptrbr_ramp_epochs
        )
        if current_nptrbr_lambda > 0.0 and nptrbr_teacher is None:
            nptrbr_teacher = copy.deepcopy(model).to(device)
            nptrbr_teacher.eval()
            for parameter in nptrbr_teacher.parameters():
                parameter.requires_grad_(False)
            LOGGER.info(
                "Frozen NP-TRBR teacher after BCE warmup=%d | first active epoch=%d",
                args.nptrbr_warmup,
                epoch_number,
            )
        nptrbr_refresh_due = current_nptrbr_lambda > 0.0 and (
            nptrbr_state is None
            or (epoch_number - args.nptrbr_warmup - 1) % args.nptrbr_refresh == 0
        )
        if nptrbr_refresh_due:
            nptrbr_state = estimate_nptrbr_budget_state(
                model=model,
                arrays=train_arrays,
                y_train=y_train,
                budgets=nptrbr_budgets,
                base_weights=nptrbr_weights,
                fpr_slack=args.nptrbr_fpr_slack,
                device=device,
                prediction_batch_size=args.prediction_batch_size,
                pool_size=args.nptrbr_pool_size,
                seed=args.seed + 40000 + epoch,
            )
            nptrbr_state["epoch"] = epoch_number
            nptrbr_state["recall_lambda"] = current_nptrbr_lambda
            nptrbr_state_history.append(dict(nptrbr_state))
            nptrbr_threshold_logits = torch.as_tensor(
                nptrbr_state["threshold_logits"], dtype=torch.float32, device=device
            )
            LOGGER.info(
                "NP-TRBR  | budgets=%s | hard-recall=%s | FPR-target=%s",
                [round(value, 4) for value in nptrbr_budgets],
                [round(value, 4) for value in nptrbr_state["training_positive_hard_recalls"]],
                [round(value, 4) for value in nptrbr_state["fpr_targets"]],
            )
        topk_refresh_due = current_topk_lambda > 0.0 and (
            topk_state is None
            or (epoch_number - args.topk_recall_warmup - 1) % args.topk_recall_refresh == 0
        )
        if topk_refresh_due:
            topk_state = estimate_topk_budget_state(
                model=model,
                arrays=train_arrays,
                y_train=y_train,
                budgets=topk_budgets,
                base_weights=topk_base_weights,
                device=device,
                prediction_batch_size=args.prediction_batch_size,
                pool_size=args.topk_recall_pool_size,
                seed=args.seed + 30000 + epoch,
            )
            topk_state["epoch"] = epoch_number
            topk_state["loss_lambda"] = current_topk_lambda
            topk_state_history.append(dict(topk_state))
            topk_threshold_logits = torch.as_tensor(
                topk_state["threshold_logits"], dtype=torch.float32, device=device
            )
            topk_adaptive_weights = torch.as_tensor(
                topk_state["adaptive_weights"], dtype=torch.float32, device=device
            )
            LOGGER.info(
                "Top-K  | budgets=%s | recall=%s | weights=%s",
                [round(value, 4) for value in topk_budgets],
                [round(value, 4) for value in topk_state["training_positive_recalls"]],
                [round(value, 4) for value in topk_state["adaptive_weights"]],
            )
        current_mix = warmup_mix if epoch_number < args.hard_start_epoch else steady_mix
        bag = sampler_engine.sample_member_bags(
            ratio=member_negative_ratio,
            members=args.members,
            seed=args.seed + 10000 + epoch,
            hard_bank=hard_bank,
            mix=current_mix,
        )
        epoch_dataset = MemberBaggingDataset(dataset, bag.indices, bag.member_mask)
        generator = torch.Generator()
        generator.manual_seed(args.seed + epoch)
        loader = DataLoader(
            epoch_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            generator=generator,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
            drop_last=False,
        )
        model.train()
        loss_total = 0.0
        auxiliary_loss_total = 0.0
        robust_loss_total = 0.0
        rank_loss_total = 0.0
        topk_recall_loss_total = 0.0
        nptrbr_loss_total = 0.0
        nptrbr_recall_total = np.zeros(len(nptrbr_budgets), dtype=np.float64)
        nptrbr_fpr_total = np.zeros(len(nptrbr_budgets), dtype=np.float64)
        nptrbr_fpr_violation_total = np.zeros(len(nptrbr_budgets), dtype=np.float64)
        nptrbr_negative_kl_total = 0.0
        nptrbr_kl_violation_total = 0.0
        decorrelation_loss_total = 0.0
        alignment_loss_total = 0.0
        mecg_l0_total = 0.0
        mecg_consensus_total = 0.0
        mecg_redundancy_total = 0.0
        mecg_budget_total = 0.0
        mecg_active_total = 0.0
        batches = 0
        for batch_number, (
            x_num,
            x_cat,
            x_ctx,
            x_history,
            time_group,
            target,
            sample_weight,
            member_mask,
        ) in enumerate(loader, start=1):
            x_num = x_num.to(device, non_blocking=True)
            x_cat = x_cat.to(device, non_blocking=True)
            x_ctx = x_ctx.to(device, non_blocking=True)
            x_history = x_history.to(device, non_blocking=True)
            time_group = time_group.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            sample_weight = sample_weight.to(device, non_blocking=True)
            member_mask = member_mask.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, amp_enabled):
                logits, sparse_entropy, _, components = model(
                    x_num, x_cat, x_ctx, x_history, return_components=True
                )
                ensure_finite("student logits", logits, epoch=epoch_number, batch=batch_number)
                expanded_target = target[:, None].expand_as(logits)
                element_loss = F.binary_cross_entropy_with_logits(
                    logits, expanded_target, reduction="none", pos_weight=positive_weight
                )
                weighted_member_mask = member_mask * sample_weight[:, None]
                member_denominator = weighted_member_mask.sum(dim=0).clamp_min(EPS)
                per_member_loss = (element_loss * weighted_member_mask).sum(
                    dim=0
                ) / member_denominator
                base_loss = per_member_loss.mean()
                row_denominator = member_mask.sum(dim=1).clamp_min(1.0)
                row_loss = (element_loss * member_mask).sum(dim=1) / row_denominator
                robust_loss = temporal_robustness(
                    row_loss,
                    time_group,
                    args.robust_min_group_count,
                    row_weight=sample_weight,
                    mode=args.temporal_robust_mode,
                    temperature=args.group_dro_temperature,
                )
                rank_loss = logits.new_tensor(0.0)
                topk_recall_loss_value = (
                    topk_budget_recall_loss(
                        logits,
                        target,
                        topk_threshold_logits,
                        topk_adaptive_weights,
                        args.topk_recall_temperature,
                    )
                    if current_topk_lambda > 0.0
                    else logits.new_tensor(0.0)
                )
                if (
                    current_nptrbr_lambda > 0.0
                    and nptrbr_teacher is not None
                    and (nptrbr_threshold_logits is not None)
                ):
                    with torch.no_grad():
                        teacher_logits, _, _, _ = nptrbr_teacher(
                            x_num, x_cat, x_ctx, x_history, return_components=True
                        )
                    nptrbr_terms = nptrbr_augmented_loss(
                        logits=logits,
                        teacher_logits=teacher_logits,
                        target=target,
                        threshold_logits=nptrbr_threshold_logits,
                        budget_weights=nptrbr_weight_tensor,
                        fpr_targets=nptrbr_fpr_targets,
                        fpr_duals=nptrbr_fpr_duals,
                        kl_dual=nptrbr_kl_dual,
                        recall_lambda=current_nptrbr_lambda,
                        temperature=args.nptrbr_temperature,
                        kl_delta=args.nptrbr_kl_delta,
                        fpr_rho=args.nptrbr_fpr_rho,
                        kl_rho=args.nptrbr_kl_rho,
                    )
                else:
                    zero_vector = torch.zeros(
                        len(nptrbr_budgets), dtype=torch.float32, device=device
                    )
                    nptrbr_terms = {
                        "loss": logits.new_tensor(0.0),
                        "recall_reward": logits.new_tensor(0.0),
                        "soft_recall": zero_vector,
                        "soft_fpr": zero_vector,
                        "fpr_violation": zero_vector,
                        "negative_kl": logits.new_tensor(0.0),
                        "kl_violation": logits.new_tensor(0.0),
                    }
                representation = components.get("representation")
                decorrelation_loss = logits.new_tensor(0.0)
                alignment_loss = logits.new_tensor(0.0)
                mecg_l0 = components.get("mecg_l0", logits.new_tensor(0.0))
                mecg_consensus = components.get("mecg_prior", logits.new_tensor(0.0))
                mecg_redundancy = components.get("mecg_redundancy", logits.new_tensor(0.0))
                mecg_budget = components.get("mecg_budget", logits.new_tensor(0.0))
                mecg_active = components.get("mecg_active_probability", logits.new_tensor(1.0))
                auxiliary_loss = logits.new_tensor(0.0)
                loss = (
                    base_loss
                    + 0.0 * robust_loss
                    + args.sparse_lambda * sparse_entropy
                    + 0.0 * auxiliary_loss
                    + 0.0 * rank_loss
                    + current_topk_lambda * topk_recall_loss_value
                    + nptrbr_terms["loss"]
                    + 0.0 * decorrelation_loss
                    + 0.0 * alignment_loss
                    + current_mecg_scale
                    * (
                        args.mecg_l0_lambda * mecg_l0
                        + args.mecg_consensus_lambda * mecg_consensus
                        + args.mecg_redundancy_lambda * mecg_redundancy
                        + args.mecg_budget_lambda * mecg_budget
                    )
                )
            ensure_finite("total loss", loss, epoch=epoch_number, batch=batch_number)
            ensure_finite("NP-TRBR KL", nptrbr_terms["negative_kl"], epoch=epoch_number, batch=batch_number)
            grad_scaler.scale(loss).backward()
            grad_scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            if not amp_enabled:
                ensure_finite("gradient norm", grad_norm, epoch=epoch_number, batch=batch_number)
            grad_scaler.step(optimizer)
            grad_scaler.update()
            loss_total += float(loss.detach().cpu())
            auxiliary_loss_total += float(auxiliary_loss.detach().cpu())
            robust_loss_total += float(robust_loss.detach().cpu())
            rank_loss_total += float(rank_loss.detach().cpu())
            topk_recall_loss_total += float(topk_recall_loss_value.detach().cpu())
            nptrbr_loss_total += float(nptrbr_terms["loss"].detach().cpu())
            nptrbr_recall_total += (
                nptrbr_terms["soft_recall"].detach().cpu().numpy().astype(np.float64, copy=False)
            )
            nptrbr_fpr_total += (
                nptrbr_terms["soft_fpr"].detach().cpu().numpy().astype(np.float64, copy=False)
            )
            nptrbr_fpr_violation_total += (
                nptrbr_terms["fpr_violation"].detach().cpu().numpy().astype(np.float64, copy=False)
            )
            nptrbr_negative_kl_total += float(nptrbr_terms["negative_kl"].detach().cpu())
            nptrbr_kl_violation_total += float(nptrbr_terms["kl_violation"].detach().cpu())
            decorrelation_loss_total += float(decorrelation_loss.detach().cpu())
            alignment_loss_total += float(alignment_loss.detach().cpu())
            mecg_l0_total += float(mecg_l0.detach().cpu())
            mecg_consensus_total += float(mecg_consensus.detach().cpu())
            mecg_redundancy_total += float(mecg_redundancy.detach().cpu())
            mecg_budget_total += float(mecg_budget.detach().cpu())
            mecg_active_total += float(mecg_active.detach().cpu())
            batches += 1
        if current_nptrbr_lambda > 0.0 and batches > 0:
            mean_fpr_violation = torch.as_tensor(
                nptrbr_fpr_violation_total / float(batches), dtype=torch.float32, device=device
            )
            mean_kl_violation = torch.as_tensor(
                nptrbr_kl_violation_total / float(batches), dtype=torch.float32, device=device
            )
            nptrbr_fpr_duals = torch.clamp(
                nptrbr_fpr_duals + args.nptrbr_dual_lr * mean_fpr_violation,
                min=0.0,
                max=args.nptrbr_dual_max,
            ).detach()
            nptrbr_kl_dual = torch.clamp(
                nptrbr_kl_dual + args.nptrbr_dual_lr * mean_kl_violation,
                min=0.0,
                max=args.nptrbr_dual_max,
            ).detach()
            nptrbr_dual_history.append(
                {
                    "epoch": epoch_number,
                    "fpr_duals": nptrbr_fpr_duals.detach().cpu().tolist(),
                    "kl_dual": float(nptrbr_kl_dual.detach().cpu()),
                    "mean_fpr_violation": mean_fpr_violation.detach().cpu().tolist(),
                    "mean_kl_violation": float(mean_kl_violation.detach().cpu()),
                }
            )
        scheduler.step()
        val_raw, _, _ = predict_probabilities(
            model, early_val_arrays, device, args.prediction_batch_size
        )
        val_ap = float(average_precision_score(y_early_val, val_raw))
        validation_ap_window.append(val_ap)
        smoothed_val_ap = float(np.mean(validation_ap_window[-args.early_stop_smoothing :]))
        epoch_record = {
            "epoch": epoch_number,
            "member_negative_ratio": member_negative_ratio,
            "sample": bag.stats,
            "train_loss": loss_total / max(batches, 1),
            "branch_auxiliary_loss": auxiliary_loss_total / max(batches, 1),
            "temporal_robust_loss": robust_loss_total / max(batches, 1),
            "rank_loss": rank_loss_total / max(batches, 1),
            "topk_recall_loss": topk_recall_loss_total / max(batches, 1),
            "topk_recall_lambda": current_topk_lambda,
            "topk_budget_state": dict(topk_state) if topk_state is not None else None,
            "nptrbr_loss": nptrbr_loss_total / max(batches, 1),
            "nptrbr_lambda": current_nptrbr_lambda,
            "nptrbr_soft_recall": (nptrbr_recall_total / max(batches, 1)).tolist(),
            "nptrbr_soft_fpr": (nptrbr_fpr_total / max(batches, 1)).tolist(),
            "nptrbr_negative_kl": nptrbr_negative_kl_total / max(batches, 1),
            "nptrbr_fpr_duals": nptrbr_fpr_duals.detach().cpu().tolist(),
            "nptrbr_kl_dual": float(nptrbr_kl_dual.detach().cpu()),
            "nptrbr_budget_state": dict(nptrbr_state) if nptrbr_state is not None else None,
            "decorrelation_loss": decorrelation_loss_total / max(batches, 1),
            "temporal_alignment_loss": alignment_loss_total / max(batches, 1),
            "mecg_scale": current_mecg_scale,
            "mecg_l0": mecg_l0_total / max(batches, 1),
            "mecg_consensus": mecg_consensus_total / max(batches, 1),
            "mecg_redundancy": mecg_redundancy_total / max(batches, 1),
            "mecg_budget": mecg_budget_total / max(batches, 1),
            "mecg_active_probability": mecg_active_total / max(batches, 1),
            "validation_average_precision_raw": val_ap,
            "validation_average_precision_smoothed": smoothed_val_ap,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(epoch_record)
        LOGGER.info(
            "Epoch %03d | member ratio=1:%.3g | rows=%s | unique neg=%s | coverage=%.2f%% | overlap=%.2f%% | loss=%.6f | topk=%.5f(lambda=%.4f) | NPTRBR=%.5f(lambda=%.4f,KL=%.6g) | MECG active=%.3f(scale=%.2f) | val AUPRC=%.6f | smooth=%.6f",
            epoch_number,
            member_negative_ratio,
            f"{len(bag.indices):,}",
            f"{bag.stats['unique_negative_rows']:,}",
            100.0 * bag.stats["cumulative_negative_coverage"],
            100.0 * bag.stats["member_overlap_rate"],
            epoch_record["train_loss"],
            epoch_record["topk_recall_loss"],
            current_topk_lambda,
            epoch_record["nptrbr_loss"],
            current_nptrbr_lambda,
            epoch_record["nptrbr_negative_kl"],
            epoch_record["mecg_active_probability"],
            current_mecg_scale,
            val_ap,
            smoothed_val_ap,
        )
        if epoch_number >= args.early_stop_warmup:
            top_checkpoints.append((val_ap, smoothed_val_ap, epoch_number, cpu_state_dict(model)))
            top_checkpoints.sort(key=lambda item: (item[0], item[2]), reverse=True)
            del top_checkpoints[args.checkpoint_ensemble_top_k :]
            if smoothed_val_ap > best_smoothed_ap + args.min_delta:
                best_smoothed_ap = smoothed_val_ap
                best_epoch = epoch_number
                patience_counter = 0
            else:
                patience_counter += 1
        else:
            patience_counter = 0
        should_stop = patience_counter >= args.patience
        first_hard_refresh = max(1, args.hard_start_epoch - 1)
        refresh_due = (
            args.mix_hard > 0
            and args.hard_refresh > 0
            and (epoch_number >= first_hard_refresh)
            and ((epoch_number - first_hard_refresh) % args.hard_refresh == 0)
        )
        if refresh_due and not should_stop:
            hard_bank = refresh_hard_bank(
                model=model,
                arrays=train_arrays,
                negative_indices=sampler_engine.negative,
                device=device,
                prediction_batch_size=args.prediction_batch_size,
                pool_size=args.hard_pool_size,
                bank_size=args.hard_bank_size,
                seed=args.seed + 20000 + epoch,
            )
            LOGGER.info("Refreshed previous-snapshot hard-negative bank: %s", f"{len(hard_bank):,}")
        save_epoch_checkpoint(output_dir/"training_state.pt", locals())
        LOGGER.info("Recovery state saved after epoch %d", epoch_number)
        if should_stop:
            LOGGER.info("Early stopping: no smoothed validation AP improvement for %d epochs.", args.patience)
            break
    if not top_checkpoints:
        raise RuntimeError("checkpoint  --early-stop-warmup")
    top_checkpoints.sort(key=lambda item: (item[0], item[2]), reverse=True)
    selected_states = [item[3] for item in top_checkpoints]
    ensemble_epochs = [int(item[2]) for item in top_checkpoints]
    model.to(device)
    mecg_selection_summary = export_mecg_checkpoint_selection(
        model=model,
        state_dicts=selected_states,
        config=model_config,
        output_path=output_dir / "mecg_feature_selection.csv",
        selection_threshold=args.mecg_selection_threshold,
    )
    if mecg_selection_summary is not None:
        LOGGER.info(
            "MECG =%d/%d | mean-open=%.3f | checkpoint gate SD=%.4f",
            mecg_selection_summary["selected_fields"],
            mecg_selection_summary["total_fields"],
            mecg_selection_summary["active_probability_mean"],
            mecg_selection_summary["checkpoint_stability_mean_std"],
        )
    early_raw, early_member_std, _ = predict_checkpoint_ensemble(
        model, selected_states, early_val_arrays, device, args.prediction_batch_size
    )
    best_ap = float(average_precision_score(y_early_val, early_raw))
    LOGGER.info(
        "Checkpoint probability ensemble | epochs=%s | best smoothed epoch=%d | smoothed AP=%.6f | ensemble raw AP=%.6f",
        ensemble_epochs,
        best_epoch,
        best_smoothed_ap,
        best_ap,
    )
    calibration_raw, calibration_member_std, _ = predict_checkpoint_ensemble(
        model, selected_states, calibration_arrays, device, args.prediction_batch_size
    )
    calibrator = ProbabilityCalibrator(args.calibration).fit(calibration_raw, y_calibration)
    early_probability = calibrator.predict(early_raw)
    calibration_probability = calibrator.predict(calibration_raw)
    threshold, threshold_info = select_threshold(
        y_calibration,
        calibration_probability,
        args.threshold_strategy,
        args.fbeta,
        args.target_recall,
        args.alert_fraction,
    )
    calibration_metrics = binary_metrics(y_calibration, calibration_probability, threshold)
    early_metrics = binary_metrics(y_early_val, early_probability, threshold)
    test_raw: Optional[np.ndarray] = None
    test_member_std: Optional[np.ndarray] = None
    test_probability: Optional[np.ndarray] = None
    test_metrics: Optional[Dict[str, Any]] = None
    if evaluate_test:
        if test_arrays is None or y_test is None:
            raise RuntimeError("Public benchmark configuration error")
        test_raw, test_member_std, _ = predict_checkpoint_ensemble(
            model, selected_states, test_arrays, device, args.prediction_batch_size
        )
        test_probability = calibrator.predict(test_raw)
        test_metrics = binary_metrics(y_test, test_probability, threshold)
    save_complete_metric_tables(
        output_dir,
        {
            "early_stopping_validation": early_metrics,
            "calibration_validation": calibration_metrics,
            "locked_test": test_metrics,
        },
    )

    def uncertainty_summary(values: np.ndarray) -> Dict[str, float]:
        return {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "q90": float(np.quantile(values, 0.9)),
            "q99": float(np.quantile(values, 0.99)),
            "max": float(np.max(values)),
        }

    importance_count = min(args.importance_max_rows, len(calibration_arrays))
    rng = np.random.default_rng(args.seed)
    importance_idx = rng.choice(len(calibration_arrays), size=importance_count, replace=False)
    _, _, mask_mean = predict_checkpoint_ensemble(
        model,
        selected_states,
        calibration_arrays,
        device,
        args.prediction_batch_size,
        indices=importance_idx,
        return_masks=True,
    )
    if mask_mean is not None:
        aggregate_mask_importance(mask_mean, preprocessor).to_csv(
            output_dir / "feature_importance.csv", index=False, encoding="utf-8-sig"
        )
    modulation_gate_key = "feature_modulator.gate_logit"
    context_input_key = "context_encoder.0.weight"
    if selected_modulation_features and all(
        (modulation_gate_key in state for state in selected_states)
    ):
        gate_values = np.mean(
            [torch.sigmoid(state[modulation_gate_key]).numpy() for state in selected_states], axis=0
        )
        pd.DataFrame(
            {"feature": selected_modulation_features, "modulation_gate": gate_values}
        ).sort_values("modulation_gate", ascending=False).to_csv(
            output_dir / "feature_modulation_gates.csv", index=False, encoding="utf-8-sig"
        )
    if all((context_input_key in state for state in selected_states)):
        context_importance = np.mean(
            [state[context_input_key].abs().mean(dim=0).numpy() for state in selected_states],
            axis=0,
        )
        if len(context_importance) == len(preprocessor.context_columns):
            pd.DataFrame(
                {
                    "context_feature": preprocessor.context_columns,
                    "encoder_weight_importance": context_importance,
                }
            ).sort_values("encoder_weight_importance", ascending=False).to_csv(
                output_dir / "context_feature_importance.csv", index=False, encoding="utf-8-sig"
            )
    history_importance: Optional[np.ndarray] = None
    fusion_weight_summary: Optional[Dict[str, float]] = None
    history_weight_key = "history_encoder.0.weight"
    if preprocessor.history_columns and all(
        (history_weight_key in state for state in selected_states)
    ):
        history_importance = np.mean(
            [state[history_weight_key].abs().mean(dim=0).numpy() for state in selected_states],
            axis=0,
        )
        importance_total = float(history_importance.sum())
        normalized_history_importance = (
            history_importance / importance_total if importance_total > 0 else history_importance
        )
        pd.DataFrame(
            {
                "feature": preprocessor.history_columns,
                "history_branch_weight_importance": history_importance,
                "history_branch_weight_importance_normalized": normalized_history_importance,
            }
        ).sort_values("history_branch_weight_importance_normalized", ascending=False).to_csv(
            output_dir / "history_feature_importance.csv", index=False, encoding="utf-8-sig"
        )
    variant_name = {
        "vanilla_tabm": "Vanilla-TabM-ControlledBaseline",
        "mecg_star_tabm": "Full-MECG-STaR-TabM",
    }[args.model_variant]
    checkpoint = {
        "format_version": 11,
        "model_name": variant_name,
        "model_variant": args.model_variant,
        "experiment_label": args.experiment_label,
        "training_strategy": "member_wise_dynamic_negative_bagging",
        "run_mode": args.run_mode,
        "feature_stage": "quick_check",
        "branch_mode": "static",
        "split_strategy": getattr(args,"_transfer_protocol","chronological"),
        "model_config": asdict(model_config),
        "model_state_dict": selected_states[0],
        "checkpoint_state_dicts": selected_states,
        "checkpoint_ensemble_method": "prediction_probability_mean",
        "threshold": threshold,
        "target_col": args.target_col,
        "timestamp_col": args.timestamp_col,
        "target_threshold": args.target_threshold,
        "positive_values": args.positive_values,
        "history_risk_config": to_jsonable(asdict(history_config)),
        "history_evaluation_protocol": history_protocol,
        "temporal_decay": to_jsonable(time_decay_stats),
        "dynamic_topk_recall": {
            "budgets": topk_budgets,
            "base_weights": topk_base_weights,
            "maximum_lambda": 0.0,
            "warmup_epochs": args.topk_recall_warmup,
            "ramp_epochs": args.topk_recall_ramp_epochs,
            "temperature": args.topk_recall_temperature,
            "refresh_epochs": args.topk_recall_refresh,
            "state_history": topk_state_history,
        },
        "nptrbr": {
            "enabled": bool(args.nptrbr_lambda > 0),
            "definition": "Neyman-Pearson Trust-Region Budgeted Recall",
            "budgets": nptrbr_budgets,
            "weights": nptrbr_weights,
            "maximum_recall_lambda": args.nptrbr_lambda,
            "warmup_epochs": args.nptrbr_warmup,
            "ramp_epochs": args.nptrbr_ramp_epochs,
            "temperature": args.nptrbr_temperature,
            "refresh_epochs": args.nptrbr_refresh,
            "negative_reference_pool_size": args.nptrbr_pool_size,
            "fpr_slack": args.nptrbr_fpr_slack,
            "kl_delta": args.nptrbr_kl_delta,
            "fpr_rho": args.nptrbr_fpr_rho,
            "kl_rho": args.nptrbr_kl_rho,
            "dual_learning_rate": args.nptrbr_dual_lr,
            "dual_maximum": args.nptrbr_dual_max,
            "teacher": "frozen_same-initialization_model_after_BCE_warmup",
            "threshold_source": "training_negatives_only",
            "state_history": nptrbr_state_history,
            "dual_history": nptrbr_dual_history,
        },
        "mecg": {
            "enabled": bool(args.model_variant == "mecg_star_tabm" and args.mecg_enabled),
            "definition": "Multi-Evidence Consensus Gating",
            "evidence_uses_only_training_partition": True,
            "evidence_prior_enabled": args.mecg_evidence_prior,
            "group_gates": args.mecg_group_gates,
            "hstp_enabled": False,
            "regularization": {
                "l0_lambda": args.mecg_l0_lambda,
                "consensus_lambda": args.mecg_consensus_lambda,
                "redundancy_lambda": args.mecg_redundancy_lambda,
                "budget_lambda": args.mecg_budget_lambda,
                "warmup_epochs": args.mecg_warmup_epochs,
                "ramp_epochs": args.mecg_ramp_epochs,
            },
            "selection": to_jsonable(mecg_selection_summary),
        },
        "mecg_selection_summary": to_jsonable(mecg_selection_summary),
        "best_epoch": best_epoch,
        "checkpoint_ensemble_epochs": ensemble_epochs,
        "checkpoint_ensemble_scores": [
            {
                "epoch": int(epoch),
                "raw_auprc": float(raw_score),
                "smoothed_auprc": float(smoothed_score),
            }
            for raw_score, smoothed_score, epoch, _ in top_checkpoints
        ],
        "best_smoothed_validation_average_precision": best_smoothed_ap,
    }
    torch.save(checkpoint, output_dir / "model.pt")
    with (output_dir / "preprocessor.pkl").open("wb") as handle:
        pickle.dump(preprocessor, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with (output_dir / "calibrator.pkl").open("wb") as handle:
        pickle.dump(calibrator, handle, protocol=pickle.HIGHEST_PROTOCOL)
    early_probability = calibrator.predict(early_raw)
    pd.DataFrame(
        {
            "source_index": early_val_arrays.source_index,
            "timestamp": early_val_arrays.timestamp,
            "target": y_early_val,
            "raw_probability": early_raw,
            "calibrated_probability": early_probability,
            "ensemble_std": early_member_std,
            "prediction": (early_probability >= threshold).astype(np.int8),
        }
    ).to_csv(
        output_dir / "early_stopping_validation_predictions.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(
        {
            "source_index": calibration_arrays.source_index,
            "timestamp": calibration_arrays.timestamp,
            "target": y_calibration,
            "raw_probability": calibration_raw,
            "calibrated_probability": calibration_probability,
            "ensemble_std": calibration_member_std,
            "prediction": (calibration_probability >= threshold).astype(np.int8),
        }
    ).to_csv(output_dir / "validation_predictions.csv", index=False, encoding="utf-8-sig")
    if evaluate_test:
        assert test_arrays is not None
        assert y_test is not None
        assert test_raw is not None
        assert test_probability is not None
        assert test_member_std is not None
        pd.DataFrame(
            {
                "source_index": test_arrays.source_index,
                "timestamp": test_arrays.timestamp,
                "target": y_test,
                "raw_probability": test_raw,
                "calibrated_probability": test_probability,
                "ensemble_std": test_member_std,
                "prediction": (test_probability >= threshold).astype(np.int8),
            }
        ).to_csv(output_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")
    results = {
        "model": variant_name,
        "model_variant": args.model_variant,
        "experiment_label": args.experiment_label,
        "run_mode": args.run_mode,
        "feature_stage": "quick_check",
        "branch_mode": "static",
        "fusion_weight_summary": fusion_weight_summary,
        "history_risk": {
            "config": to_jsonable(asdict(history_config)),
            "evaluation_protocol": history_protocol,
            "label_availability_assumption": f"target available after {history_config.label_delay_days:g} days"
            if history_config.enabled
            else "disabled",
        },
        "temporal_decay": to_jsonable(time_decay_stats),
        "dynamic_topk_recall": {
            "budgets": topk_budgets,
            "base_weights": topk_base_weights,
            "maximum_lambda": 0.0,
            "warmup_epochs": args.topk_recall_warmup,
            "ramp_epochs": args.topk_recall_ramp_epochs,
            "temperature": args.topk_recall_temperature,
            "refresh_epochs": args.topk_recall_refresh,
            "state_history": topk_state_history,
        },
        "nptrbr": to_jsonable(checkpoint["nptrbr"]),
        "mecg": to_jsonable(checkpoint["mecg"]),
        "parameter_count": parameter_count,
        "best_epoch": best_epoch,
        "checkpoint_ensemble_method": "prediction_probability_mean",
        "checkpoint_ensemble_epochs": ensemble_epochs,
        "checkpoint_ensemble_scores": [
            {
                "epoch": int(epoch),
                "raw_auprc": float(raw_score),
                "smoothed_auprc": float(smoothed_score),
            }
            for raw_score, smoothed_score, epoch, _ in top_checkpoints
        ],
        "best_smoothed_validation_average_precision": best_smoothed_ap,
        "best_validation_raw_average_precision": best_ap,
        "split": split_info,
        "training_sampling": {
            "strategy": "member_wise_dynamic_negative_bagging",
            "member_negative_ratio": member_negative_ratio,
            "members": args.members,
            "warmup_mix": warmup_mix,
            "steady_mix": steady_mix,
            "final_cumulative_negative_coverage": history[-1]["sample"][
                "cumulative_negative_coverage"
            ]
            if history
            else 0.0,
        },
        "threshold_selection": threshold_info,
        "calibration": calibrator.method,
        "early_stopping_validation": {
            "average_precision_raw": float(average_precision_score(y_early_val, early_raw)),
            "prevalence": float(np.mean(y_early_val)),
            "average_precision_lift_raw": float(
                average_precision_score(y_early_val, early_raw)
                / max(float(np.mean(y_early_val)), EPS)
            ),
            "complete_metrics_descriptive": early_metrics,
        },
        "calibration_validation_metrics": calibration_metrics,
        "validation_metrics": calibration_metrics,
        "test_metrics": test_metrics,
        "ensemble_disagreement": {
            "early_stopping_validation": uncertainty_summary(early_member_std),
            "calibration_validation": uncertainty_summary(calibration_member_std),
            "test": uncertainty_summary(test_member_std) if test_member_std is not None else None,
        },
        "feature_schema": {
            "numeric": preprocessor.numeric_columns,
            "modulated_numeric": [
                column
                for column, enabled in zip(preprocessor.numeric_columns, numeric_modulation_mask)
                if enabled
            ],
            "categorical": preprocessor.categorical_columns,
            "context": preprocessor.context_columns,
            "prototype_numeric": preprocessor.prototype.feature_cols
            if getattr(preprocessor, "use_hstp", True)
            else [],
            "long_tail_bounds": preprocessor.long_tail_bounds,
            "missing_indicators": preprocessor.missing_indicator_columns,
            "history": preprocessor.history_columns,
            "climate": getattr(preprocessor, "climate_columns", []),
            "numeric_encoding": "piecewise_linear_embedding"
            if model_config.numeric_embedding_dim > 0
            else "raw_scaled",
            "ple_bins": args.ple_bins,
            "numeric_embedding_dim": args.numeric_embedding_dim,
            "time_fourier_harmonics": args.time_fourier_harmonics,
            "absolute_time_trend": False,
        },
        "model_config": asdict(model_config),
        "training_seconds": time.time() - start_time,
        "arguments": vars(args),
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(results), handle, ensure_ascii=False, indent=2)
    with (output_dir / "training_history.json").open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(history), handle, ensure_ascii=False, indent=2)
    if test_metrics is not None:
        LOGGER.info(
            "| AUPRC %.6f (lift %.2fx) | ROC-AUC %.6f | Precision %.4f | Recall %.4f | F1 %.4f | Recall@Top1%% %.4f",
            test_metrics["average_precision"],
            test_metrics["average_precision_lift"],
            test_metrics["roc_auc"],
            test_metrics["precision"],
            test_metrics["recall"],
            test_metrics["f1"],
            test_metrics["top_fraction"]["1%"]["recall"],
        )
        log_classification_summary("test", test_metrics)
    else:
        LOGGER.info(
            "|   AUPRC %.6f (lift %.2fx) | Recall@Top1%% %.4f |",
            calibration_metrics["average_precision"],
            calibration_metrics["average_precision_lift"],
            calibration_metrics["top_fraction"]["1%"]["recall"],
        )
        log_classification_summary("calibration_validation", calibration_metrics)
    LOGGER.info("Results saved to: %s", output_dir)
    return results
