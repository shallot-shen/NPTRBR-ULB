from __future__ import annotations
import argparse
import logging
import math
from typing import Any, Dict, List, Sequence, Tuple
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

LOGGER = logging.getLogger("public_baselines")


def make_one_hot_encoder(min_frequency: int) -> OneHotEncoder:
    kwargs = {"handle_unknown": "ignore", "min_frequency": min_frequency, "dtype": np.float32}
    try:
        return OneHotEncoder(sparse_output=True, **kwargs)
    except TypeError:
        return OneHotEncoder(sparse=True, **kwargs)


def fit_logistic(
    train_x: pd.DataFrame,
    train_y: np.ndarray,
    train_weight: np.ndarray,
    numeric_cols: Sequence[str],
    categorical_cols: Sequence[str],
    args: argparse.Namespace,
) -> Pipeline:
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler(with_mean=False)),
        ]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot", make_one_hot_encoder(args.category_min_frequency)),
        ]
    )
    preprocess = ColumnTransformer(
        [
            ("numeric", numeric, list(numeric_cols)),
            ("categorical", categorical, list(categorical_cols)),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )
    classifier = LogisticRegression(
        C=args.logistic_c,
        solver="saga",
        l1_ratio=0.0,
        class_weight="balanced",
        max_iter=args.logistic_max_iter,
        tol=args.logistic_tol,
        random_state=args.seed,
        verbose=0,
    )
    pipeline = Pipeline([("preprocess", preprocess), ("classifier", classifier)])
    pipeline.fit(train_x, train_y, classifier__sample_weight=train_weight)
    iterations = int(np.max(pipeline.named_steps["classifier"].n_iter_))
    LOGGER.info(
        "Logistic =%d/%d | converged=%s",
        iterations,
        args.logistic_max_iter,
        iterations < args.logistic_max_iter,
    )
    return pipeline


def fit_catboost(
    train_x: pd.DataFrame,
    train_y: np.ndarray,
    train_weight: np.ndarray,
    early_x: pd.DataFrame,
    early_y: np.ndarray,
    categorical_cols: Sequence[str],
    args: argparse.Namespace,
) -> Tuple[Any, int, List[Dict[str, float]]]:
    try:
        from catboost import CatBoostClassifier
    except ImportError as exc:
        raise RuntimeError("catboost  pip install catboost") from exc
    catboost_kwargs: Dict[str, Any] = dict(
        iterations=args.catboost_iterations,
        depth=args.catboost_depth,
        learning_rate=args.catboost_learning_rate,
        l2_leaf_reg=args.catboost_l2_leaf_reg,
        loss_function="Logloss",
        eval_metric="Logloss",
        auto_class_weights="Balanced",
        random_seed=args.seed,
        random_strength=1.0,
        task_type=args.task_type,
        thread_count=args.workers,
        allow_writing_files=False,
        verbose=args.catboost_verbose,
    )
    if args.task_type == "GPU":
        catboost_kwargs["devices"] = args.devices
    model = CatBoostClassifier(**catboost_kwargs)
    model.fit(
        train_x,
        train_y,
        sample_weight=train_weight,
        cat_features=list(categorical_cols),
        use_best_model=False,
    )
    best_ap = -math.inf
    best_tree_count = 1
    stale_rounds = 0
    selection_trace: List[Dict[str, float]] = []
    LOGGER.info(
        "CatBoost :  sklearn average_precision_score  | patience=%d",
        args.catboost_early_stopping_rounds,
    )
    staged = model.staged_predict_proba(
        early_x,
        ntree_start=0,
        ntree_end=model.tree_count_,
        eval_period=1,
        thread_count=args.workers,
    )
    for tree_count, staged_probability in enumerate(staged, start=1):
        staged_probability = np.asarray(staged_probability[:, 1], dtype=np.float64)
        ap = float(average_precision_score(early_y, staged_probability))
        selection_trace.append({"tree_count": float(tree_count), "auprc": ap})
        if ap > best_ap + 1e-12:
            best_ap = ap
            best_tree_count = tree_count
            stale_rounds = 0
        else:
            stale_rounds += 1
        if stale_rounds >= args.catboost_early_stopping_rounds:
            break
    model.shrink(ntree_end=best_tree_count)
    LOGGER.info(
        "CatBoost  |  %d  |  =%d | sklearn AUPRC=%.6f",
        len(selection_trace),
        best_tree_count,
        best_ap,
    )
    return (model, best_tree_count - 1, selection_trace)
