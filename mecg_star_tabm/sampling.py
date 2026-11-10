from __future__ import annotations
import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset

LOGGER = logging.getLogger("mecg_star_tabm")
EPS = 1e-6
MISSING_TOKEN = "__MISSING__"

from .preprocessing import PreparedArrays


class ArrayDataset(Dataset):
    def __init__(
        self, arrays: PreparedArrays, y: np.ndarray, sample_weight: Optional[np.ndarray] = None
    ) -> None:
        self.x_numeric = torch.from_numpy(arrays.x_numeric)
        self.x_categorical = torch.from_numpy(arrays.x_categorical)
        self.x_context = torch.from_numpy(arrays.x_context)
        self.x_history = torch.from_numpy(arrays.x_history)
        self.time_group = torch.from_numpy(arrays.time_group)
        self.y = torch.from_numpy(y.astype(np.float32, copy=False))
        if sample_weight is None:
            sample_weight = np.ones(len(y), dtype=np.float32)
        sample_weight = np.asarray(sample_weight, dtype=np.float32)
        if sample_weight.shape != (len(y),):
            raise ValueError("sample_weight")
        if not np.isfinite(sample_weight).all() or np.any(sample_weight <= 0):
            raise ValueError("sample_weight")
        self.sample_weight = torch.from_numpy(sample_weight)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, ...]:
        return (
            self.x_numeric[index],
            self.x_categorical[index],
            self.x_context[index],
            self.x_history[index],
            self.time_group[index],
            self.y[index],
            self.sample_weight[index],
        )


class MemberBaggingDataset(Dataset):
    """Epoch view containing the union of member-specific training bags."""

    def __init__(self, base: ArrayDataset, indices: np.ndarray, member_mask: np.ndarray) -> None:
        if len(indices) != len(member_mask):
            raise ValueError("indices member_mask")
        self.base = base
        self.indices = torch.from_numpy(indices.astype(np.int64, copy=False))
        self.member_mask = torch.from_numpy(member_mask.astype(np.float32, copy=False))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int) -> Tuple[torch.Tensor, ...]:
        index = int(self.indices[position])
        return (*self.base[index], self.member_mask[position])


@dataclass
class MemberBagSample:
    indices: np.ndarray
    member_mask: np.ndarray
    stats: Dict[str, Any]


class DynamicNegativeSampler:
    """Member-wise dynamic negative bagging inside a fixed training pool.

    Positives are shared by all members. Negatives are assigned to a single
    member whenever the pool is large enough, so each member keeps a stable
    class ratio while the ensemble covers roughly ``members * ratio`` times as
    many unique negatives per epoch. The coverage reservoir persists across
    epochs and is independent of validation/test data.
    """

    def __init__(
        self, y: np.ndarray, matched_key: np.ndarray, deviation_bin: np.ndarray, seed: int
    ) -> None:
        self.y = y.astype(np.int8)
        self.positive = np.flatnonzero(self.y == 1)
        self.negative = np.flatnonzero(self.y == 0)
        if len(self.positive) == 0 or len(self.negative) == 0:
            raise ValueError("Public benchmark configuration error")
        self.matched_key = matched_key
        self.deviation_bin = deviation_bin
        self.neg_by_match = self._group_indices(self.negative, matched_key)
        self.neg_by_bin = self._group_indices(self.negative, deviation_bin)
        exact_key = np.array(
            [f"{matched_key[i]}|D{int(deviation_bin[i])}" for i in range(len(y))], dtype=object
        )
        self.exact_key = exact_key
        self.neg_by_exact = self._group_indices(self.negative, exact_key)
        self.coverage_rng = np.random.default_rng(seed)
        self.coverage_order = self.negative.copy()
        self.coverage_rng.shuffle(self.coverage_order)
        self.coverage_cursor = 0
        self.coverage_cycles_completed = 0
        self.times_selected = np.zeros(len(self.y), dtype=np.uint32)

    def recovery_state(self):
        return {"coverage_order": self.coverage_order.copy(),
                "coverage_cursor": self.coverage_cursor,
                "coverage_cycles_completed": self.coverage_cycles_completed,
                "times_selected": self.times_selected.copy(),
                "coverage_rng": self.coverage_rng.bit_generator.state}

    def restore_recovery_state(self, state):
        if state["times_selected"].shape != self.times_selected.shape:
            raise ValueError("Sampler recovery pool changed")
        if not np.array_equal(np.sort(state["coverage_order"]), self.negative):
            raise ValueError("Sampler recovery negatives changed")
        self.coverage_order = state["coverage_order"].copy()
        self.coverage_cursor = int(state["coverage_cursor"])
        self.coverage_cycles_completed = int(state["coverage_cycles_completed"])
        self.times_selected = state["times_selected"].copy()
        self.coverage_rng.bit_generator.state = state["coverage_rng"]

    @staticmethod
    def _group_indices(indices: np.ndarray, keys: np.ndarray) -> Dict[Any, np.ndarray]:
        groups: Dict[Any, List[int]] = defaultdict(list)
        for idx in indices:
            groups[keys[idx]].append(int(idx))
        return {key: np.asarray(value, dtype=np.int64) for key, value in groups.items()}

    @staticmethod
    def _add_candidates(
        candidates: Iterable[int], selected: np.ndarray, output: List[int], target: int
    ) -> int:
        added = 0
        for value in candidates:
            idx = int(value)
            if not selected[idx]:
                selected[idx] = True
                output.append(idx)
                added += 1
                if added >= target:
                    break
        return added

    def _sample_from_positive_keys(
        self, n: int, rng: np.random.Generator, selected: np.ndarray, exact: bool
    ) -> List[int]:
        if n <= 0:
            return []
        groups = self.neg_by_exact if exact else self.neg_by_match
        keys = self.exact_key if exact else self.matched_key
        out: List[int] = []
        attempts = 0
        max_attempts = max(n * 20, 200)
        while len(out) < n and attempts < max_attempts:
            pos_idx = int(self.positive[rng.integers(0, len(self.positive))])
            pool = groups.get(keys[pos_idx])
            if pool is None or len(pool) == 0:
                if exact:
                    pool = self.neg_by_bin.get(self.deviation_bin[pos_idx])
                if pool is None or len(pool) == 0:
                    attempts += 1
                    continue
            idx = int(pool[rng.integers(0, len(pool))])
            if not selected[idx]:
                selected[idx] = True
                out.append(idx)
            attempts += 1
        return out

    def _sample_coverage(self, n: int, selected: np.ndarray) -> List[int]:
        if n <= 0:
            return []
        available = int(len(self.negative) - selected[self.negative].sum())
        target = min(n, available)
        out: List[int] = []
        while len(out) < target:
            if self.coverage_cursor >= len(self.coverage_order):
                self.coverage_rng.shuffle(self.coverage_order)
                self.coverage_cursor = 0
                self.coverage_cycles_completed += 1
            idx = int(self.coverage_order[self.coverage_cursor])
            self.coverage_cursor += 1
            if selected[idx]:
                continue
            selected[idx] = True
            out.append(idx)
        return out

    def _sample_random_available(
        self, n: int, rng: np.random.Generator, selected: np.ndarray
    ) -> List[int]:
        if n <= 0:
            return []
        available = self.negative[~selected[self.negative]]
        if len(available) == 0:
            return []
        draw = rng.choice(available, size=min(n, len(available)), replace=False)
        out: List[int] = []
        self._add_candidates(draw, selected, out, n)
        return out

    def sample_member_bags(
        self,
        ratio: float,
        members: int,
        seed: int,
        hard_bank: Optional[np.ndarray],
        mix: Tuple[float, float, float, float],
    ) -> MemberBagSample:
        if ratio <= 0:
            raise ValueError(": 0")
        if members < 1:
            raise ValueError("TabM 1")
        rng = np.random.default_rng(seed)
        target_per_member = max(1, int(round(len(self.positive) * ratio)))
        fractions = np.asarray(mix, dtype=np.float64)
        if np.any(fractions < 0) or fractions.sum() <= 0:
            raise ValueError("mix 0")
        fractions = fractions / fractions.sum()
        quotas = np.floor(fractions * target_per_member).astype(int)
        quotas[0] += target_per_member - int(quotas.sum())
        selected = np.zeros(len(self.y), dtype=bool)
        memberships: Dict[int, List[int]] = {}
        source_counts: Dict[str, int] = defaultdict(int)
        overlap_assignments = 0
        for member in range(members):
            member_values: List[int] = []
            coverage = self._sample_coverage(int(quotas[0]), selected)
            member_values.extend(coverage)
            source_counts["coverage"] += len(coverage)
            region = self._sample_from_positive_keys(int(quotas[1]), rng, selected, exact=False)
            member_values.extend(region)
            source_counts["region_month"] += len(region)
            prototype = self._sample_from_positive_keys(int(quotas[2]), rng, selected, exact=True)
            member_values.extend(prototype)
            source_counts["prototype_matched"] += len(prototype)
            hard: List[int] = []
            if int(quotas[3]) > 0 and hard_bank is not None and len(hard_bank):
                hard_order = hard_bank[rng.permutation(len(hard_bank))]
                self._add_candidates(hard_order, selected, hard, int(quotas[3]))
            member_values.extend(hard)
            source_counts["hard"] += len(hard)
            remaining = target_per_member - len(member_values)
            if remaining > 0:
                fill = self._sample_coverage(remaining, selected)
                member_values.extend(fill)
                source_counts["coverage_fallback"] += len(fill)
            remaining = target_per_member - len(member_values)
            if remaining > 0:
                fill = self._sample_random_available(remaining, rng, selected)
                member_values.extend(fill)
                source_counts["random_fallback"] += len(fill)
            remaining = target_per_member - len(member_values)
            if remaining > 0:
                already_member = np.zeros(len(self.y), dtype=bool)
                already_member[np.asarray(member_values, dtype=np.int64)] = True
                available = self.negative[~already_member[self.negative]]
                if len(available):
                    draw = rng.choice(
                        available, size=min(remaining, len(available)), replace=False
                    ).astype(np.int64)
                    member_values.extend(draw.tolist())
                    overlap_assignments += len(draw)
                    source_counts["overlap_fallback"] += len(draw)
            if len(member_values) != target_per_member:
                raise RuntimeError(
                    f"Public benchmark configuration error{member}Public benchmark configuration error{len(member_values)}Public benchmark configuration error{target_per_member}。"
                )
            for idx in member_values:
                memberships.setdefault(int(idx), []).append(member)
        negative_indices = np.fromiter(memberships.keys(), dtype=np.int64)
        indices = np.concatenate([self.positive, negative_indices])
        member_mask = np.zeros((len(indices), members), dtype=np.float32)
        member_mask[: len(self.positive), :] = 1.0
        for row, idx in enumerate(negative_indices, start=len(self.positive)):
            member_mask[row, memberships[int(idx)]] = 1.0
        self.times_selected[negative_indices] += 1
        member_negative_counts = member_mask[len(self.positive) :].sum(axis=0).astype(int)
        assignment_count = int(member_negative_counts.sum())
        unique_negative_count = len(negative_indices)
        cumulative_unique = int(np.count_nonzero(self.times_selected[self.negative]))
        stats: Dict[str, Any] = {
            "positive_rows": len(self.positive),
            "negative_pool_rows": len(self.negative),
            "unique_negative_rows": unique_negative_count,
            "negative_assignments": assignment_count,
            "target_negative_per_member": target_per_member,
            "member_negative_counts": member_negative_counts.tolist(),
            "member_negative_ratio": float(ratio),
            "effective_union_negative_ratio": unique_negative_count / len(self.positive),
            "member_overlap_assignments": overlap_assignments,
            "member_overlap_rate": max(assignment_count - unique_negative_count, 0)
            / max(assignment_count, 1),
            "cumulative_unique_negatives": cumulative_unique,
            "cumulative_negative_coverage": cumulative_unique / len(self.negative),
            "never_selected_negatives": len(self.negative) - cumulative_unique,
            "coverage_cycles_completed": self.coverage_cycles_completed,
            "source_counts": dict(source_counts),
            "mix": {
                "coverage": float(fractions[0]),
                "region_month": float(fractions[1]),
                "prototype_matched": float(fractions[2]),
                "hard": float(fractions[3]),
            },
        }
        return MemberBagSample(indices=indices, member_mask=member_mask, stats=stats)
