from __future__ import annotations
import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

LOGGER = logging.getLogger("mecg_star_tabm")
EPS = 1e-6
MISSING_TOKEN = "__MISSING__"


class NumericFeatureEncoder(nn.Module):
    """Raw numeric passthrough or train-fitted piecewise-linear embedding.

    The encoding uses cumulative linear spline bases over train-only quantile
    knots.  A linear layer can therefore learn a separate slope in every
    interval while remaining continuous at the knots.  This is the practical
    ingredient behind the strongest TabM-with-numerical-embeddings variants.
    """

    def __init__(
        self, n_features: int, bin_edges: Optional[Sequence[Sequence[float]]], embedding_dim: int
    ) -> None:
        super().__init__()
        self.n_features = int(n_features)
        self.embedding_dim = int(embedding_dim)
        self.use_ple = self.embedding_dim > 0
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        if not self.use_ple:
            self.output_dim = self.n_features
            return
        if bin_edges is None or len(bin_edges) != self.n_features:
            raise ValueError("PLE")
        for index, values in enumerate(bin_edges):
            edges = torch.as_tensor(values, dtype=torch.float32)
            if edges.ndim != 1 or edges.numel() < 2:
                raise ValueError(f"Public benchmark configuration error{index}PLE")
            edges = torch.unique_consecutive(edges)
            if edges.numel() < 2:
                edges = torch.tensor([-1.0, 1.0], dtype=torch.float32)
            self.register_buffer(f"edges_{index}", edges)
            intervals = int(edges.numel() - 1)
            weight = nn.Parameter(torch.empty(intervals, self.embedding_dim))
            nn.init.normal_(weight, mean=0.0, std=0.02)
            self.weights.append(weight)
            self.biases.append(nn.Parameter(torch.zeros(self.embedding_dim)))
        self.output_dim = self.n_features * self.embedding_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_ple:
            return x
        if x.size(-1) != self.n_features:
            raise ValueError("PLE")
        encoded: List[torch.Tensor] = []
        for index in range(self.n_features):
            edges = getattr(self, f"edges_{index}").to(dtype=x.dtype)
            left = edges[:-1]
            width = (edges[1:] - left).clamp_min(1e-06)
            basis = ((x[..., index : index + 1] - left) / width).clamp(0.0, 1.0)
            value = basis @ self.weights[index].to(dtype=x.dtype)
            encoded.append(value + self.biases[index].to(dtype=x.dtype))
        return torch.cat(encoded, dim=-1)


class FeatureAwareSpatioTemporalModulator(nn.Module):
    """Inactive compatibility input modulator retained from the evaluated run.

    The public recipe fixes shared_scale=lambda_range=member_rank=0. Keeping
    these layers preserves parameter initialization order and checkpoint keys.
    There is no feature-aware contextual modulation experiment in this release.
    """

    def __init__(
        self,
        n_features: int,
        context_dim: int,
        members: int,
        modulation_mask: Optional[Sequence[bool]],
        shared_scale: float,
        lambda_range: float,
        member_rank: int,
        member_scale: float,
    ) -> None:
        super().__init__()
        if modulation_mask is None:
            modulation_mask = [True] * int(n_features)
        if len(modulation_mask) != int(n_features):
            raise ValueError("Public benchmark configuration error")
        selected = [index for index, enabled in enumerate(modulation_mask) if enabled]
        if not selected:
            raise ValueError("STaR-TabM")
        if member_rank < 0:
            raise ValueError("rank")
        self.n_features = int(n_features)
        self.n_selected = len(selected)
        self.members = int(members)
        self.member_rank = int(member_rank)
        self.shared_scale = float(shared_scale)
        self.lambda_range = float(lambda_range)
        self.member_scale = float(member_scale)
        self.register_buffer("selected_indices", torch.as_tensor(selected, dtype=torch.long))
        self.context_to_shared = nn.Linear(context_dim, 3 * self.n_selected)
        nn.init.zeros_(self.context_to_shared.weight)
        nn.init.zeros_(self.context_to_shared.bias)
        self.gate_logit = nn.Parameter(torch.full((self.n_selected,), -2.0))
        if self.member_rank > 0:
            self.context_to_member = nn.Linear(
                context_dim, self.members * self.member_rank, bias=False
            )
            nn.init.zeros_(self.context_to_member.weight)
            self.member_gamma_basis = nn.Parameter(
                torch.randn(self.members, self.member_rank, self.n_selected) * 0.02
            )
            self.member_beta_basis = nn.Parameter(
                torch.randn(self.members, self.member_rank, self.n_selected) * 0.02
            )
        else:
            self.context_to_member = None
            self.register_parameter("member_gamma_basis", None)
            self.register_parameter("member_beta_basis", None)

    @staticmethod
    def _yeo_johnson(x: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
        """Differentiable Yeo-Johnson transform with stable lambda limits."""
        positive_x = x.clamp_min(0.0)
        negative_x = x.clamp_max(0.0)
        near_zero = lam.abs() < 0.001
        safe_lam = torch.where(near_zero, torch.ones_like(lam), lam)
        positive_general = torch.expm1(lam * torch.log1p(positive_x)) / safe_lam
        positive = torch.where(near_zero, torch.log1p(positive_x), positive_general)
        two_minus = 2.0 - lam
        near_two = two_minus.abs() < 0.001
        safe_two_minus = torch.where(near_two, torch.ones_like(two_minus), two_minus)
        negative_general = -torch.expm1(two_minus * torch.log1p(-negative_x)) / safe_two_minus
        negative = torch.where(near_two, -torch.log1p(-negative_x), negative_general)
        return torch.where(x >= 0.0, positive, negative)

    def forward(
        self, x: torch.Tensor, context: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if x.ndim != 2 or x.size(-1) != self.n_features:
            raise ValueError("[B, F]")
        batch = x.size(0)
        selected_x = x.index_select(-1, self.selected_indices)
        shared = self.context_to_shared(context)
        shared_gamma, shared_beta, shared_lambda = shared.chunk(3, dim=-1)
        gate = torch.sigmoid(self.gate_logit).unsqueeze(0)
        gamma = 1.0 + gate * self.shared_scale * torch.tanh(shared_gamma)
        beta = gate * self.shared_scale * torch.tanh(shared_beta)
        lam = 1.0 + gate * self.lambda_range * torch.tanh(shared_lambda)
        transformed = self._yeo_johnson(selected_x, lam)
        shared_view = gamma * transformed + beta
        member_view = shared_view.unsqueeze(1).expand(-1, self.members, -1)
        member_delta_norm = selected_x.new_tensor(0.0)
        if self.context_to_member is not None:
            assert self.member_gamma_basis is not None
            assert self.member_beta_basis is not None
            coefficient = self.context_to_member(context).view(
                batch, self.members, self.member_rank
            )
            delta_gamma = torch.einsum("bkr,krf->bkf", coefficient, self.member_gamma_basis)
            delta_beta = torch.einsum("bkr,krf->bkf", coefficient, self.member_beta_basis)
            residual_scale = self.member_scale / math.sqrt(float(self.member_rank))
            delta_gamma = residual_scale * torch.tanh(delta_gamma)
            delta_beta = residual_scale * torch.tanh(delta_beta)
            member_view = member_view * (1.0 + delta_gamma) + delta_beta
            member_delta_norm = 0.5 * (delta_gamma.square().mean() + delta_beta.square().mean())
        output = x.unsqueeze(1).expand(-1, self.members, -1).clone()
        output[..., self.selected_indices] = member_view
        diagnostics = {
            "feature_modulation_gate": gate.mean(),
            "feature_modulation_lambda_mean": lam.mean(),
            "feature_modulation_member_delta": member_delta_norm,
        }
        return (output, diagnostics)


class MultiEvidenceConsensusGate(nn.Module):
    """Hierarchical Hard-Concrete gate initialized by train-only evidence.

    MECG is shared by all TabM members.  This is intentional: member-specific
    negative bags diversify decision boundaries, while a shared field mask
    represents the cross-member consensus about which measurements are safe to
    use.  Group and field gates are multiplied, allowing an entire semantic
    family to close without forcing every member to rediscover that decision.
    """

    def __init__(
        self,
        n_numeric: int,
        n_categorical: int,
        group_ids: Sequence[int],
        feature_prior: Sequence[float],
        group_prior: Sequence[float],
        numeric_redundancy: Optional[Sequence[Sequence[float]]],
        temperature: float,
        stretch_low: float,
        stretch_high: float,
        use_group_gates: bool,
        stochastic: bool,
        identity: bool,
        target_active_ratio: float,
    ) -> None:
        super().__init__()
        self.n_numeric = int(n_numeric)
        self.n_categorical = int(n_categorical)
        self.n_fields = self.n_numeric + self.n_categorical
        self.temperature = float(temperature)
        self.stretch_low = float(stretch_low)
        self.stretch_high = float(stretch_high)
        self.use_group_gates = bool(use_group_gates)
        self.stochastic = bool(stochastic)
        self.identity = bool(identity)
        self.target_active_ratio = float(target_active_ratio)
        if self.n_fields < 1:
            raise ValueError("MECG")
        if len(group_ids) != self.n_fields or len(feature_prior) != self.n_fields:
            raise ValueError("MECG")
        if self.temperature <= 0:
            raise ValueError("MECG Hard-Concrete")
        if not self.stretch_low < 0.0 < 1.0 < self.stretch_high:
            raise ValueError("MECG stretch low<0<1<high")
        group_ids_tensor = torch.as_tensor(group_ids, dtype=torch.long)
        n_groups = int(group_ids_tensor.max().item()) + 1
        if len(group_prior) != n_groups:
            raise ValueError("MECG group_ids")
        feature_prior_tensor = torch.as_tensor(feature_prior, dtype=torch.float32).clamp(0.01, 0.99)
        group_prior_tensor = torch.as_tensor(group_prior, dtype=torch.float32).clamp(0.01, 0.99)
        self.register_buffer("group_ids", group_ids_tensor)
        self.register_buffer("feature_prior", feature_prior_tensor)
        self.register_buffer("group_prior", group_prior_tensor)
        self.feature_log_alpha = nn.Parameter(torch.logit(feature_prior_tensor))
        self.group_log_alpha = nn.Parameter(torch.logit(group_prior_tensor))
        if numeric_redundancy is None:
            redundancy = torch.zeros(self.n_numeric, self.n_numeric, dtype=torch.float32)
        else:
            redundancy = torch.as_tensor(numeric_redundancy, dtype=torch.float32)
        if redundancy.shape != (self.n_numeric, self.n_numeric):
            raise ValueError("MECG")
        self.register_buffer("numeric_redundancy", redundancy)

    @staticmethod
    def _bernoulli_kl(probability: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
        probability = probability.clamp(1e-05, 1.0 - 1e-05)
        prior = prior.clamp(1e-05, 1.0 - 1e-05)
        return (
            probability * torch.log(probability / prior)
            + (1.0 - probability) * torch.log((1.0 - probability) / (1.0 - prior))
        ).mean()

    def _open_probability(self, log_alpha: torch.Tensor) -> torch.Tensor:
        offset = self.temperature * math.log(-self.stretch_low / self.stretch_high)
        return torch.sigmoid(log_alpha - offset)

    def _deterministic_gate(self, log_alpha: torch.Tensor) -> torch.Tensor:
        stretched = (
            torch.sigmoid(log_alpha) * (self.stretch_high - self.stretch_low) + self.stretch_low
        )
        return stretched.clamp(0.0, 1.0)

    def _sample_gate(self, log_alpha: torch.Tensor) -> torch.Tensor:
        if not (self.training and self.stochastic):
            return self._deterministic_gate(log_alpha)
        uniform = torch.rand_like(log_alpha).clamp(1e-06, 1.0 - 1e-06)
        logistic_noise = torch.log(uniform) - torch.log1p(-uniform)
        relaxed = torch.sigmoid((log_alpha + logistic_noise) / self.temperature)
        stretched = relaxed * (self.stretch_high - self.stretch_low) + self.stretch_low
        return stretched.clamp(0.0, 1.0)

    def probabilities(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.identity:
            feature = torch.ones_like(self.feature_log_alpha)
            group = torch.ones_like(self.group_log_alpha)
            return (feature, group, feature)
        feature = self._open_probability(self.feature_log_alpha)
        group = self._open_probability(self.group_log_alpha)
        combined = feature * (
            group.index_select(0, self.group_ids)
            if self.use_group_gates
            else torch.ones_like(feature)
        )
        return (feature, group, combined)

    def forward(self) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if self.identity:
            combined_gate = torch.ones_like(self.feature_log_alpha)
            zero = combined_gate.new_tensor(0.0)
            return (
                combined_gate,
                {
                    "mecg_l0": zero,
                    "mecg_prior": zero,
                    "mecg_redundancy": zero,
                    "mecg_budget": zero,
                    "mecg_active_probability": combined_gate.mean(),
                },
            )
        feature_gate = self._sample_gate(self.feature_log_alpha)
        group_gate = self._sample_gate(self.group_log_alpha)
        combined_gate = feature_gate * (
            group_gate.index_select(0, self.group_ids)
            if self.use_group_gates
            else torch.ones_like(feature_gate)
        )
        feature_probability, group_probability, combined_probability = self.probabilities()
        l0_penalty = combined_probability.mean()
        if self.use_group_gates:
            l0_penalty = 0.75 * l0_penalty + 0.25 * group_probability.mean()
        prior_penalty = self._bernoulli_kl(feature_probability, self.feature_prior)
        if self.use_group_gates:
            prior_penalty = prior_penalty + 0.5 * self._bernoulli_kl(
                group_probability, self.group_prior
            )
        numeric_probability = combined_probability[: self.n_numeric]
        redundancy_mass = self.numeric_redundancy.sum()
        redundancy_penalty = (
            torch.einsum(
                "i,ij,j->", numeric_probability, self.numeric_redundancy, numeric_probability
            )
            / redundancy_mass.clamp_min(1.0)
            if self.n_numeric > 1
            else combined_probability.new_tensor(0.0)
        )
        budget_penalty = (combined_probability.mean() - self.target_active_ratio).square()
        return (
            combined_gate,
            {
                "mecg_l0": l0_penalty,
                "mecg_prior": prior_penalty,
                "mecg_redundancy": redundancy_penalty,
                "mecg_budget": budget_penalty,
                "mecg_active_probability": combined_probability.mean(),
            },
        )


class BatchEnsembleLinear(nn.Module):
    """Standard context-free BatchEnsemble linear layer used by vanilla TabM."""

    def __init__(self, in_features: int, out_features: int, members: int) -> None:
        super().__init__()
        self.members = members
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.r = nn.Parameter(torch.ones(members, in_features))
        self.s = nn.Parameter(torch.ones(members, out_features))
        self.bias = nn.Parameter(torch.zeros(members, out_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.normal_(self.r, mean=1.0, std=0.05)
        nn.init.normal_(self.s, mean=1.0, std=0.05)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1).expand(-1, self.members, -1)
        output = F.linear(x * self.r.unsqueeze(0), self.weight)
        return output * self.s.unsqueeze(0) + self.bias.unsqueeze(0)


class LowRankBatchEnsembleLinear(nn.Module):
    """BatchEnsemble layer with member-specific low-rank residual adapters.

    The shared matrix learns the common tabular relationship.  Each implicit
    ensemble member receives a small LoRA-style residual, increasing useful
    member diversity without duplicating the complete backbone.
    """

    def __init__(
        self, in_features: int, out_features: int, members: int, rank: int, scale: float = 0.1
    ) -> None:
        super().__init__()
        if rank < 1:
            raise ValueError("rank 1")
        self.members = int(members)
        self.rank = int(rank)
        self.scale = float(scale)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.r = nn.Parameter(torch.ones(members, in_features))
        self.s = nn.Parameter(torch.ones(members, out_features))
        self.bias = nn.Parameter(torch.zeros(members, out_features))
        self.adapter_down = nn.Parameter(torch.empty(members, rank, in_features))
        self.adapter_up = nn.Parameter(torch.zeros(members, out_features, rank))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.normal_(self.r, mean=1.0, std=0.05)
        nn.init.normal_(self.s, mean=1.0, std=0.05)
        nn.init.kaiming_uniform_(self.adapter_down, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1).expand(-1, self.members, -1)
        modulated = x * self.r.unsqueeze(0)
        shared = F.linear(modulated, self.weight) * self.s.unsqueeze(0)
        bottleneck = torch.einsum("bki,kri->bkr", x, self.adapter_down)
        residual = torch.einsum("bkr,kor->bko", bottleneck, self.adapter_up)
        residual = residual * (self.scale / math.sqrt(float(self.rank)))
        return shared + residual + self.bias.unsqueeze(0)


class LowRankTabMBlock(nn.Module):
    def __init__(
        self, hidden_dim: int, members: int, rank: int, adapter_scale: float, dropout: float
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc1 = LowRankBatchEnsembleLinear(
            hidden_dim, hidden_dim * 2, members, rank, adapter_scale
        )
        self.fc2 = LowRankBatchEnsembleLinear(
            hidden_dim * 2, hidden_dim, members, rank, adapter_scale
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.silu(self.fc1(self.norm(x)))
        x = self.dropout(x)
        x = self.fc2(x)
        return residual + self.dropout(x)


class VanillaTabMBlock(nn.Module):
    def __init__(self, hidden_dim: int, members: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc1 = BatchEnsembleLinear(hidden_dim, hidden_dim * 2, members)
        self.fc2 = BatchEnsembleLinear(hidden_dim * 2, hidden_dim, members)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = F.silu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return residual + self.dropout(x)


@dataclass
class ModelConfig:
    n_numeric: int
    categorical_cardinalities: List[int]
    categorical_embedding_dims: List[int]
    n_context: int
    hidden_dim: int = 192
    members: int = 8
    n_blocks: int = 3
    selector_steps: int = 3
    selector_gamma: float = 1.5
    context_dim: int = 64
    adapter_rank: int = 4
    member_adapter_rank: int = 0
    member_adapter_scale: float = 0.1
    numeric_embedding_dim: int = 0
    numeric_bin_edges: Optional[List[List[float]]] = None
    numeric_modulation_mask: Optional[List[bool]] = None
    feature_modulation_scale: float = 0.1
    feature_modulation_lambda_range: float = 0.5
    modulation_member_rank: int = 4
    modulation_member_scale: float = 0.05
    dropout: float = 0.15
    n_history: int = 0
    history_hidden_dim: int = 32
    branch_mode: str = "static"
    model_variant: str = "mecg_star_tabm"
    mecg_enabled: bool = False
    mecg_numeric_names: Optional[List[str]] = None
    mecg_categorical_names: Optional[List[str]] = None
    mecg_group_names: Optional[List[str]] = None
    mecg_group_ids: Optional[List[int]] = None
    mecg_feature_prior: Optional[List[float]] = None
    mecg_group_prior: Optional[List[float]] = None
    mecg_numeric_redundancy: Optional[List[List[float]]] = None
    mecg_temperature: float = 0.67
    mecg_stretch_low: float = -0.1
    mecg_stretch_high: float = 1.1
    mecg_group_gates: bool = True
    mecg_stochastic_gates: bool = True
    mecg_target_active_ratio: float = 0.55
    ensemble_init: str = "legacy"


class VanillaTabM(nn.Module):
    """Context-free TabM architecture under the shared rare-event trainer.

    This baseline deliberately excludes HSTP prototype features, the stepwise
    sparse selector, context-conditioned adapters and the history branch. It
    retains the same categorical embeddings, member count, training split,
    temporal weights and member-wise class-balanced sampling for a controlled
    architecture comparison.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.branch_mode != "static" or config.n_history != 0:
            raise ValueError("Vanilla TabM static")
        self.config = config
        self.numeric_encoder = NumericFeatureEncoder(
            config.n_numeric, config.numeric_bin_edges, config.numeric_embedding_dim
        )
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(cardinality, dim, padding_idx=0)
                for cardinality, dim in zip(
                    config.categorical_cardinalities, config.categorical_embedding_dims
                )
            ]
        )
        base_dim = self.numeric_encoder.output_dim + sum(config.categorical_embedding_dims)
        self.base_norm = nn.LayerNorm(base_dim)
        self.input_layer = BatchEnsembleLinear(base_dim, config.hidden_dim, config.members)
        self.blocks = nn.ModuleList(
            [
                VanillaTabMBlock(config.hidden_dim, config.members, config.dropout)
                for _ in range(config.n_blocks)
            ]
        )
        self.output_norm = nn.LayerNorm(config.hidden_dim)
        self.head = BatchEnsembleLinear(config.hidden_dim, 1, config.members)

    def forward(
        self,
        x_numeric: torch.Tensor,
        x_categorical: torch.Tensor,
        x_context: torch.Tensor,
        x_history: Optional[torch.Tensor] = None,
        return_masks: bool = False,
        return_components: bool = False,
    ) -> Any:
        del x_context, x_history, return_masks
        embedded = [embedding(x_categorical[:, i]) for i, embedding in enumerate(self.embeddings)]
        encoded_numeric = self.numeric_encoder(x_numeric)
        base = torch.cat([encoded_numeric] + embedded, dim=-1) if embedded else encoded_numeric
        hidden = F.silu(self.input_layer(self.base_norm(base)))
        for block in self.blocks:
            hidden = block(hidden)
        logits = self.head(self.output_norm(hidden)).squeeze(-1)
        sparse_entropy = logits.new_tensor(0.0)
        standard_output = (logits, sparse_entropy, None)
        if not return_components:
            return standard_output
        return (
            *standard_output,
            {
                "static_logits": logits,
                "history_logits": None,
                "fusion_weight": None,
                "representation": self.output_norm(hidden).mean(dim=1),
            },
        )


class STaRTabM(nn.Module):
    """Local parameter-sharing residual ensemble used by the public experiment.

    Compatibility context is constant on ULB and input modulation is disabled.
    The frozen parameter layout is retained for numerical reproducibility.
    This is a TabM-inspired backbone, not the official TabM implementation.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.branch_mode != "static" or config.n_history != 0:
            raise ValueError("star_tabm static --no-history-risk")
        if config.member_adapter_rank < 1:
            raise ValueError("star_tabm --member-adapter-rank 1")
        if config.numeric_embedding_dim < 1:
            raise ValueError("star_tabm PLE")
        self.config = config
        self.context_encoder = nn.Sequential(
            nn.Linear(config.n_context, config.context_dim),
            nn.LayerNorm(config.context_dim),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.context_dim, config.context_dim),
            nn.SiLU(),
        )
        self.feature_modulator = FeatureAwareSpatioTemporalModulator(
            n_features=config.n_numeric,
            context_dim=config.context_dim,
            members=config.members,
            modulation_mask=config.numeric_modulation_mask,
            shared_scale=config.feature_modulation_scale,
            lambda_range=config.feature_modulation_lambda_range,
            member_rank=config.modulation_member_rank,
            member_scale=config.modulation_member_scale,
        )
        self.numeric_encoder = NumericFeatureEncoder(
            config.n_numeric, config.numeric_bin_edges, config.numeric_embedding_dim
        )
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(cardinality, dim, padding_idx=0)
                for cardinality, dim in zip(
                    config.categorical_cardinalities, config.categorical_embedding_dims
                )
            ]
        )
        base_dim = self.numeric_encoder.output_dim + sum(config.categorical_embedding_dims)
        self.base_norm = nn.LayerNorm(base_dim)
        self.input_layer = LowRankBatchEnsembleLinear(
            base_dim,
            config.hidden_dim,
            config.members,
            config.member_adapter_rank,
            config.member_adapter_scale,
        )
        self.blocks = nn.ModuleList(
            [
                LowRankTabMBlock(
                    config.hidden_dim,
                    config.members,
                    config.member_adapter_rank,
                    config.member_adapter_scale,
                    config.dropout,
                )
                for _ in range(config.n_blocks)
            ]
        )
        self.context_to_hidden_film = nn.Linear(config.context_dim, 2 * config.hidden_dim)
        nn.init.zeros_(self.context_to_hidden_film.weight)
        nn.init.zeros_(self.context_to_hidden_film.bias)
        self.hidden_gate_logit = nn.Parameter(torch.tensor(-2.0))
        self.output_norm = nn.LayerNorm(config.hidden_dim)
        self.head = LowRankBatchEnsembleLinear(
            config.hidden_dim,
            1,
            config.members,
            config.member_adapter_rank,
            config.member_adapter_scale,
        )

    def forward(
        self,
        x_numeric: torch.Tensor,
        x_categorical: torch.Tensor,
        x_context: torch.Tensor,
        x_history: Optional[torch.Tensor] = None,
        return_masks: bool = False,
        return_components: bool = False,
    ) -> Any:
        del x_history, return_masks
        context = self.context_encoder(x_context)
        numeric_members, modulation_diagnostics = self.feature_modulator(x_numeric, context)
        numeric = self.numeric_encoder(numeric_members)
        embedded = [
            embedding(x_categorical[:, index]).unsqueeze(1).expand(-1, self.config.members, -1)
            for index, embedding in enumerate(self.embeddings)
        ]
        base = torch.cat([numeric] + embedded, dim=-1) if embedded else numeric
        hidden = F.silu(self.input_layer(self.base_norm(base)))
        hidden_film = self.context_to_hidden_film(context)
        hidden_gamma, hidden_beta = hidden_film.chunk(2, dim=-1)
        hidden_gate = torch.sigmoid(self.hidden_gate_logit)
        for block in self.blocks:
            hidden = block(hidden)
            hidden = hidden * (
                1.0 + hidden_gate * 0.1 * torch.tanh(hidden_gamma).unsqueeze(1)
            ) + hidden_gate * 0.1 * torch.tanh(hidden_beta).unsqueeze(1)
        representation_members = self.output_norm(hidden)
        logits = self.head(representation_members).squeeze(-1)
        sparse_entropy = logits.new_tensor(0.0)
        standard_output = (logits, sparse_entropy, None)
        if not return_components:
            return standard_output
        return (
            *standard_output,
            {
                "static_logits": logits,
                "history_logits": None,
                "fusion_weight": None,
                "representation": representation_members.mean(dim=1),
                "hidden_context_gate": hidden_gate,
                **modulation_diagnostics,
            },
        )


class MECGSTaRTabM(STaRTabM):
    """STaR-TabM preceded by Multi-Evidence Consensus Gating (MECG)."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        numeric_names = config.mecg_numeric_names or []
        categorical_names = config.mecg_categorical_names or []
        if len(numeric_names) != config.n_numeric:
            raise ValueError("MECG")
        if len(categorical_names) != len(config.categorical_cardinalities):
            raise ValueError("MECG")
        field_count = len(numeric_names) + len(categorical_names)
        if config.mecg_group_ids is None or config.mecg_feature_prior is None:
            raise ValueError("MECG")
        self.mecg_gate = MultiEvidenceConsensusGate(
            n_numeric=config.n_numeric,
            n_categorical=len(config.categorical_cardinalities),
            group_ids=config.mecg_group_ids,
            feature_prior=config.mecg_feature_prior,
            group_prior=config.mecg_group_prior or [],
            numeric_redundancy=config.mecg_numeric_redundancy,
            temperature=config.mecg_temperature,
            stretch_low=config.mecg_stretch_low,
            stretch_high=config.mecg_stretch_high,
            use_group_gates=config.mecg_group_gates,
            stochastic=config.mecg_stochastic_gates,
            identity=not config.mecg_enabled,
            target_active_ratio=config.mecg_target_active_ratio,
        )
        if field_count != self.mecg_gate.n_fields:
            raise RuntimeError("MECG")

    def forward(
        self,
        x_numeric: torch.Tensor,
        x_categorical: torch.Tensor,
        x_context: torch.Tensor,
        x_history: Optional[torch.Tensor] = None,
        return_masks: bool = False,
        return_components: bool = False,
    ) -> Any:
        del x_history, return_masks
        context = self.context_encoder(x_context)
        field_gate, mecg_diagnostics = self.mecg_gate()
        numeric_gate = field_gate[: self.config.n_numeric]
        categorical_gate = field_gate[self.config.n_numeric :]
        numeric_members, modulation_diagnostics = self.feature_modulator(x_numeric, context)
        numeric_members = numeric_members * numeric_gate.view(1, 1, -1)
        numeric = self.numeric_encoder(numeric_members)
        embedded = [
            (embedding(x_categorical[:, index]) * categorical_gate[index])
            .unsqueeze(1)
            .expand(-1, self.config.members, -1)
            for index, embedding in enumerate(self.embeddings)
        ]
        base = torch.cat([numeric] + embedded, dim=-1) if embedded else numeric
        hidden = F.silu(self.input_layer(self.base_norm(base)))
        hidden_film = self.context_to_hidden_film(context)
        hidden_gamma, hidden_beta = hidden_film.chunk(2, dim=-1)
        hidden_gate = torch.sigmoid(self.hidden_gate_logit)
        for block in self.blocks:
            hidden = block(hidden)
            hidden = hidden * (
                1.0 + hidden_gate * 0.1 * torch.tanh(hidden_gamma).unsqueeze(1)
            ) + hidden_gate * 0.1 * torch.tanh(hidden_beta).unsqueeze(1)
        representation_members = self.output_norm(hidden)
        logits = self.head(representation_members).squeeze(-1)
        sparse_entropy = logits.new_tensor(0.0)
        standard_output = (logits, sparse_entropy, None)
        if not return_components:
            return standard_output
        return (
            *standard_output,
            {
                "static_logits": logits,
                "history_logits": None,
                "fusion_weight": None,
                "representation": representation_members.mean(dim=1),
                "hidden_context_gate": hidden_gate,
                "mecg_field_gate_mean": field_gate.mean(),
                **mecg_diagnostics,
                **modulation_diagnostics,
            },
        )


def build_model(config: ModelConfig) -> nn.Module:
    if config.model_variant == "vanilla_tabm":
        model = VanillaTabM(config)
    elif config.model_variant == "mecg_star_tabm":
        model = MECGSTaRTabM(config)
    else:
        raise ValueError("Public release supports only MECG-STaR-TabM and its local TabM-style control")
    if config.ensemble_init not in {"legacy", "feature_normal", "feature_signs"}:
        raise ValueError("Unknown ensemble_init")
    if config.ensemble_init != "legacy":
        # Diversify original fields BEFORE the first mixing layer. All PLE
        # channels of one field share its initial member-specific multiplier.
        # This is an initialization option, not an additional learned gate.
        with torch.no_grad():
            r = model.input_layer.r
            raw = torch.randn(config.members, config.n_numeric, device=r.device)
            if config.ensemble_init == "feature_signs":
                raw = torch.where(raw >= 0, 1.0, -1.0)
            width = config.numeric_embedding_dim or 1
            count = config.n_numeric * width
            r[:, :count].copy_(raw.repeat_interleave(width, dim=1))
            if r.shape[1] > count:
                if config.ensemble_init == "feature_normal":
                    nn.init.normal_(r[:, count:])
                else:
                    r[:, count:].bernoulli_(0.5).mul_(2).add_(-1)
    return model
