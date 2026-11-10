from __future__ import annotations
import logging
from typing import Dict, List, Optional
import torch
import torch.nn.functional as F
from .training_state import ensure_finite

LOGGER = logging.getLogger("mecg_star_tabm")
EPS = 1e-6
MISSING_TOKEN = "__MISSING__"


def temporal_robustness(
    row_loss: torch.Tensor,
    time_group: torch.Tensor,
    min_count: int,
    row_weight: Optional[torch.Tensor] = None,
    mode: str = "variance",
    temperature: float = 8.0,
) -> torch.Tensor:
    means: List[torch.Tensor] = []
    for group in torch.unique(time_group):
        mask = time_group == group
        if int(mask.sum()) >= min_count:
            if row_weight is None:
                means.append(row_loss[mask].mean())
            else:
                weight = row_weight[mask]
                means.append((row_loss[mask] * weight).sum() / weight.sum().clamp_min(EPS))
    if len(means) < 2:
        return row_loss.new_tensor(0.0)
    stacked = torch.stack(means)
    if mode == "variance":
        return stacked.var(unbiased=False)
    if mode == "group_dro":
        if temperature <= 0:
            raise ValueError("GroupDRO temperature 0")
        smooth_worst = torch.logsumexp(stacked * temperature, dim=0) / temperature
        return (smooth_worst - stacked.mean()).clamp_min(0.0)
    raise ValueError(f":{mode}")


def hard_pairwise_ranking_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    member_mask: torch.Tensor,
    temperature: float,
    max_hard_negatives: int,
) -> torch.Tensor:
    """Rank positives above each member's highest-scoring sampled negatives.

    This differentiable objective targets the top of the ranked list, which is
    much closer to AUPRC/Recall@Top-K than ordinary pointwise BCE.  It is kept
    as an auxiliary loss because BCE remains important for probability shape.
    """
    if temperature <= 0:
        raise ValueError("--rank-temperature 0")
    losses: List[torch.Tensor] = []
    positive = target > 0.5
    negative = ~positive
    for member in range(logits.size(1)):
        active = member_mask[:, member] > 0
        positive_scores = logits[positive & active, member]
        negative_scores = logits[negative & active, member]
        if positive_scores.numel() == 0 or negative_scores.numel() == 0:
            continue
        if max_hard_negatives > 0 and negative_scores.numel() > max_hard_negatives:
            negative_scores = torch.topk(negative_scores, k=max_hard_negatives, largest=True).values
        margin = (negative_scores.unsqueeze(0) - positive_scores.unsqueeze(1)) / temperature
        losses.append(F.softplus(margin).mean())
    if not losses:
        return logits.new_tensor(0.0)
    return torch.stack(losses).mean()


def topk_budget_recall_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    threshold_logits: Optional[torch.Tensor],
    budget_weights: Optional[torch.Tensor],
    temperature: float,
) -> torch.Tensor:
    """Push positives above training-only score cutoffs for several budgets.

    Unlike a batch-local top-k loss, the cutoffs are periodically estimated on
    a uniform reference sample from the complete *training* pool.  The loss is
    therefore aligned with operational alert budgets and remains independent
    of validation/test prevalence.
    """
    if threshold_logits is None or budget_weights is None:
        return logits.new_tensor(0.0)
    positive = target > 0.5
    if not bool(positive.any()):
        return logits.new_tensor(0.0)
    ensemble_probability = torch.sigmoid(logits).mean(dim=1).clamp(min=EPS, max=1.0 - EPS)
    ensemble_score = torch.logit(ensemble_probability)
    positive_score = ensemble_score[positive].unsqueeze(1)
    margin = (threshold_logits.to(dtype=logits.dtype).unsqueeze(0) - positive_score) / temperature
    per_budget = temperature * F.softplus(margin).mean(dim=0)
    weights = budget_weights.to(dtype=logits.dtype)
    weights = weights / weights.sum().clamp_min(EPS)
    return torch.sum(weights * per_budget)


def _ensemble_probability(logits: torch.Tensor) -> torch.Tensor:
    """Inference-aligned probability used by NP-TRBR constraints."""
    return torch.sigmoid(logits.float()).mean(dim=1).clamp(min=EPS, max=1.0 - EPS)


def nptrbr_augmented_loss(
    logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    target: torch.Tensor,
    threshold_logits: torch.Tensor,
    budget_weights: torch.Tensor,
    fpr_targets: torch.Tensor,
    fpr_duals: torch.Tensor,
    kl_dual: torch.Tensor,
    recall_lambda: float,
    temperature: float,
    kl_delta: float,
    fpr_rho: float,
    kl_rho: float,
) -> Dict[str, torch.Tensor]:
    """Compute the existing objective in FP32, outside ambient autocast.

    A finite-input guard exposes the first failing batch rather than letting
    NaNs contaminate projected duals and only fail later inside sklearn.
    """
    ensure_finite("NP-TRBR student logits", logits)
    ensure_finite("NP-TRBR teacher logits", teacher_logits)
    with torch.autocast(device_type=logits.device.type, enabled=False):
        terms = _nptrbr_augmented_loss_fp32(
            logits.float(), teacher_logits.float(), target, threshold_logits.float(),
            budget_weights.float(), fpr_targets.float(), fpr_duals.float(), kl_dual.float(),
            recall_lambda, temperature, kl_delta, fpr_rho, kl_rho,
        )
    for name, value in terms.items():
        ensure_finite("NP-TRBR "+name, value)
    return terms


def _nptrbr_augmented_loss_fp32(
    logits, teacher_logits, target, threshold_logits, budget_weights, fpr_targets,
    fpr_duals, kl_dual, recall_lambda, temperature, kl_delta, fpr_rho, kl_rho,
):
    """Neyman-Pearson trust-region budgeted-recall objective.

    The score cutoffs are estimated exclusively on training negatives.  The
    objective rewards soft positive recall above those cutoffs while an
    augmented Lagrangian constrains soft negative FPR.  A frozen warm-start
    teacher supplies a Bernoulli-KL trust region on negatives, limiting the
    degradation of already-correct background predictions.
    """
    if temperature <= 0:
        raise ValueError("NP-TRBR temperature must be positive.")
    positive = target > 0.5
    negative = ~positive
    zero = logits.new_tensor(0.0, dtype=torch.float32)
    if not bool(positive.any()) or not bool(negative.any()):
        return {
            "loss": zero,
            "recall_reward": zero,
            "soft_recall": torch.zeros_like(threshold_logits, dtype=torch.float32),
            "soft_fpr": torch.zeros_like(threshold_logits, dtype=torch.float32),
            "fpr_violation": torch.zeros_like(threshold_logits, dtype=torch.float32),
            "negative_kl": zero,
            "kl_violation": zero,
        }
    student_probability = _ensemble_probability(logits)
    teacher_probability = _ensemble_probability(teacher_logits).detach()
    student_score = torch.logit(student_probability)
    cutoff = threshold_logits.float().unsqueeze(0)
    soft_selected = torch.sigmoid((student_score.unsqueeze(1) - cutoff) / float(temperature))
    soft_recall = soft_selected[positive].mean(dim=0)
    soft_fpr = soft_selected[negative].mean(dim=0)
    weights = budget_weights.float()
    weights = weights / weights.sum().clamp_min(EPS)
    recall_reward = torch.sum(weights * soft_recall)
    fpr_violation = soft_fpr - fpr_targets.float()
    p0 = teacher_probability[negative].clamp(EPS, 1.0 - EPS)
    p = student_probability[negative].clamp(EPS, 1.0 - EPS)
    negative_kl = (
        p0 * (torch.log(p0) - torch.log(p)) + (1.0 - p0) * (torch.log1p(-p0) - torch.log1p(-p))
    ).mean()
    kl_violation = negative_kl - float(kl_delta)
    fpr_penalty = torch.sum(
        fpr_duals.float() * fpr_violation + 0.5 * float(fpr_rho) * F.relu(fpr_violation).square()
    )
    kl_penalty = (
        kl_dual.float() * kl_violation + 0.5 * float(kl_rho) * F.relu(kl_violation).square()
    )
    loss = -float(recall_lambda) * recall_reward + fpr_penalty + kl_penalty
    return {
        "loss": loss,
        "recall_reward": recall_reward,
        "soft_recall": soft_recall,
        "soft_fpr": soft_fpr,
        "fpr_violation": fpr_violation,
        "negative_kl": negative_kl,
        "kl_violation": kl_violation,
    }


def topk_recall_lambda_for_epoch(
    epoch_number: int, maximum: float, warmup_epochs: int, ramp_epochs: int
) -> float:
    if maximum <= 0.0 or epoch_number <= warmup_epochs:
        return 0.0
    progress = min(1.0, float(epoch_number - warmup_epochs) / float(max(ramp_epochs, 1)))
    smooth_progress = progress * progress * (3.0 - 2.0 * progress)
    return float(maximum * smooth_progress)


def representation_decorrelation_loss(representation: torch.Tensor) -> torch.Tensor:
    """Reduce redundant shortcut dimensions without using future/test rows."""
    if representation.ndim != 2 or representation.size(0) < 2:
        return representation.new_tensor(0.0)
    representation_fp32 = representation.float()
    centered = representation_fp32 - representation_fp32.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / float(max(representation.size(0) - 1, 1))
    diagonal = torch.diagonal(covariance)
    off_diagonal = covariance - torch.diag_embed(diagonal)
    return off_diagonal.square().mean()


def temporal_representation_alignment_loss(
    representation: torch.Tensor, time_group: torch.Tensor, min_count: int
) -> torch.Tensor:
    """Align first and second moments across observed training time groups.

    This is deliberately computed only inside the sampled training batch.  It
    never reads validation/test rows and therefore cannot leak future-period
    statistics.  The global batch moments are detached targets: the network is
    asked to remove gratuitous quarter-specific scale/offset shortcuts without
    collapsing the representation itself.
    """
    if representation.ndim != 2 or representation.size(0) < 2:
        return representation.new_tensor(0.0)
    representation_fp32 = representation.float()
    global_mean = representation_fp32.mean(dim=0).detach()
    global_variance = representation_fp32.var(dim=0, unbiased=False).detach()
    penalties: List[torch.Tensor] = []
    for group in torch.unique(time_group):
        mask = time_group == group
        if int(mask.sum()) < min_count:
            continue
        group_representation = representation_fp32[mask]
        group_mean = group_representation.mean(dim=0)
        group_variance = group_representation.var(dim=0, unbiased=False)
        penalties.append(
            (group_mean - global_mean).square().mean()
            + (group_variance - global_variance).square().mean()
        )
    if len(penalties) < 2:
        return representation.new_tensor(0.0)
    return torch.stack(penalties).mean()
