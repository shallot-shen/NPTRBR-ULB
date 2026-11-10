"""Training-fitted ULB preprocessing; no private schema or contextual dataset."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

FEATURES = [f"V{i}" for i in range(1, 29)] + ["Amount"]
EPS = 1e-6


@dataclass
class PreparedArrays:
    x_numeric: np.ndarray
    x_categorical: np.ndarray
    x_context: np.ndarray
    x_history: np.ndarray
    time_group: np.ndarray
    matched_key: np.ndarray
    deviation_bin: np.ndarray
    source_index: np.ndarray
    timestamp: np.ndarray

    def __len__(self):
        return len(self.x_numeric)


class PublicPreprocessor:
    """Preserve the evaluated float32 scaling, PLE knots and constant context.

    Twelve constant compatibility columns are intentionally retained. Removing
    them changes layer dimensions and the evaluated parameter initialization.
    They are NOT transaction times or row-position features. Inference must
    use the same constant timestamp as training.
    """

    def __init__(
        self,
        timestamp_col="__timestamp__",
        category_min_frequency=5,
        max_categories=50_000,
        prototype_tau=50.0,
        use_altitude=False,
        use_history_risk=False,
        use_hstp=False,
        use_climate_features=False,
        time_fourier_harmonics=4,
        use_absolute_time_trend=False,
        ple_bins=16,
        feature_stage="quick_check",
    ):
        if any(
            (
                use_altitude,
                use_history_risk,
                use_hstp,
                use_climate_features,
                use_absolute_time_trend,
            )
        ):
            raise ValueError("Only the public ULB configuration is supported")
        if time_fourier_harmonics != 4 or ple_bins < 4:
            raise ValueError("Frozen ULB preprocessing needs four harmonics and PLE >=4 bins")
        self.timestamp_col = timestamp_col
        self.category_min_frequency = category_min_frequency
        self.max_categories = max_categories
        self.use_altitude = False
        self.use_history_risk = False
        self.use_hstp = False
        self.use_climate_features = False
        self.use_absolute_time_trend = False
        self.time_fourier_harmonics = time_fourier_harmonics
        self.ple_bins = ple_bins
        self.feature_stage = feature_stage
        self.raw_numeric_candidates = list(FEATURES)
        self.time_numeric_columns = [
            "_month_sin",
            "_month_cos",
            "_doy_sin",
            "_doy_cos",
            "_hour_sin",
            "_hour_cos",
        ] + [f"_annual_h{h}_{c}" for h in range(2, 5) for c in ("sin", "cos")]
        self.time_origin: Optional[pd.Timestamp] = None
        self.numeric_columns = []
        self.context_columns = []
        self.categorical_columns = []
        self.category_encoders = {}
        self.history_columns = []
        self.climate_columns = []
        self.missing_source_columns = []
        self.missing_indicator_columns = []
        self.long_tail_bounds = {}
        self.numeric_bin_edges = []
        self.numeric_median = self.numeric_mean = self.numeric_std = None
        self.context_median = self.context_mean = self.context_std = None
        self.deviation_bin_edges = None

    @staticmethod
    def _matrix(df, columns):
        def numeric_column(col):
            source = df[col] if col in df.columns else pd.Series(np.nan, index=df.index)
            return (
                pd.to_numeric(source, errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .to_numpy(dtype=np.float32)
            )

        return np.column_stack([numeric_column(col) for col in columns]).astype(
            np.float32, copy=False
        )

    @staticmethod
    def _fit_scale(matrix):
        median = np.nanmedian(matrix, axis=0).astype(np.float32)
        median = np.where(np.isfinite(median), median, 0.0).astype(np.float32)
        filled = np.where(np.isfinite(matrix), matrix, median)
        mean = filled.mean(axis=0, dtype=np.float64).astype(np.float32)
        std = filled.std(axis=0, dtype=np.float64).astype(np.float32)
        std = np.where(std > EPS, std, 1.0).astype(np.float32)
        return median, mean, std

    @staticmethod
    def _apply_scale(matrix, median, mean, std):
        filled = np.where(np.isfinite(matrix), matrix, median)
        return ((filled - mean) / std).astype(np.float32)

    def _engineer_time(self, df, fitting):
        out = df.copy()
        ts = pd.to_datetime(out[self.timestamp_col], errors="raise")
        if ts.nunique() != 1:
            raise ValueError("ULB reproduction uses one constant timestamp, not event chronology")
        if fitting:
            self.time_origin = ts.min()
        if self.time_origin is None or ts.iloc[0] != self.time_origin:
            raise ValueError("Timestamp differs from the fitted public compatibility context")
        day = ts.dt.dayofyear.to_numpy(dtype=np.float32)
        hour = (ts.dt.hour + ts.dt.minute / 60.0 + ts.dt.second / 3600.0).to_numpy(dtype=np.float32)
        month = ts.dt.month.to_numpy(dtype=np.float32)
        days = ((ts - self.time_origin) / pd.Timedelta(days=1)).to_numpy(dtype=np.float32)
        out["_month_sin"] = np.sin(2.0 * np.pi * (month - 1.0) / 12.0)
        out["_month_cos"] = np.cos(2.0 * np.pi * (month - 1.0) / 12.0)
        out["_doy_sin"] = np.sin(2.0 * np.pi * (day - 1.0) / 365.2425)
        out["_doy_cos"] = np.cos(2.0 * np.pi * (day - 1.0) / 365.2425)
        out["_hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
        out["_hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
        annual_phase = days / 365.2425
        for harmonic in range(2, self.time_fourier_harmonics + 1):
            angle = 2.0 * np.pi * float(harmonic) * annual_phase
            out[f"_annual_h{harmonic}_sin"] = np.sin(angle)
            out[f"_annual_h{harmonic}_cos"] = np.cos(angle)
        return out

    def _apply_feature_transforms(self, df):
        out = df.copy()
        for col in self.missing_source_columns:
            values = pd.to_numeric(out[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
            out[f"MISS__{col}"] = values.isna().to_numpy(dtype=np.float32)
        return out

    def fit_transform(self, df):
        augmented = self._engineer_time(df, fitting=True)
        self.missing_source_columns = []
        for col in self.raw_numeric_candidates:
            values = pd.to_numeric(augmented[col], errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            )
            if values.notna().any() and values.isna().any():
                self.missing_source_columns.append(col)
        self.missing_indicator_columns = [f"MISS__{c}" for c in self.missing_source_columns]
        augmented = self._apply_feature_transforms(augmented)
        candidates = (
            self.raw_numeric_candidates + self.missing_indicator_columns + self.time_numeric_columns
        )
        self.numeric_columns = [
            c for c in candidates if pd.to_numeric(augmented[c], errors="coerce").notna().any()
        ]
        self.context_columns = list(self.time_numeric_columns)
        x_num = self._matrix(augmented, self.numeric_columns)
        self.numeric_median, self.numeric_mean, self.numeric_std = self._fit_scale(x_num)
        scaled = self._apply_scale(x_num, self.numeric_median, self.numeric_mean, self.numeric_std)
        quantile_grid = np.linspace(0.0, 1.0, self.ple_bins + 1)
        self.numeric_bin_edges = []
        for index in range(scaled.shape[1]):
            finite = scaled[:, index][np.isfinite(scaled[:, index])]
            if finite.size < 2:
                edges = np.array([-1.0, 1.0], dtype=np.float32)
            else:
                edges = np.unique(np.quantile(finite, quantile_grid).astype(np.float32))
                if edges.size < 2:
                    center = float(edges[0]) if edges.size else 0.0
                    edges = np.array([center - 1.0, center + 1.0], dtype=np.float32)
            self.numeric_bin_edges.append([float(v) for v in edges])
        ctx = self._matrix(augmented, self.context_columns)
        self.context_median, self.context_mean, self.context_std = self._fit_scale(ctx)
        self.deviation_bin_edges = np.array([0.0], dtype=np.float32)
        return self._arrays_from_augmented(augmented)

    def transform(self, df):
        return self._arrays_from_augmented(
            self._apply_feature_transforms(self._engineer_time(df, False))
        )

    def _arrays_from_augmented(self, augmented):
        if self.numeric_median is None:
            raise RuntimeError("Fit preprocessing on the training partition first")
        n = len(augmented)
        ts = pd.to_datetime(augmented[self.timestamp_col])
        return PreparedArrays(
            x_numeric=self._apply_scale(
                self._matrix(augmented, self.numeric_columns),
                self.numeric_median,
                self.numeric_mean,
                self.numeric_std,
            ),
            x_categorical=np.empty((n, 0), dtype=np.int64),
            x_context=self._apply_scale(
                self._matrix(augmented, self.context_columns),
                self.context_median,
                self.context_mean,
                self.context_std,
            ),
            x_history=np.empty((n, 0), dtype=np.float32),
            time_group=(
                ts.dt.year.to_numpy(dtype=np.int64) * 10 + ts.dt.quarter.to_numpy(dtype=np.int64)
            ),
            matched_key=np.array([f"__MISSING__|{m}" for m in ts.dt.month], dtype=object),
            deviation_bin=np.ones(n, dtype=np.int16),
            source_index=augmented.get(
                "_source_index", pd.Series(augmented.index, index=augmented.index)
            ).to_numpy(),
            timestamp=ts.astype(str).to_numpy(),
        )

    @property
    def categorical_cardinalities(self):
        return []
