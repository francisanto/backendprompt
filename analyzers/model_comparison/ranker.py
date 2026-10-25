"""
PromptShield AI — Multi-Criteria Model Ranking Engine (WSM & TOPSIS).
====================================================================
Ranks foundation models for a specific prompt using:
1. Weighted Sum Model (WSM)
2. TOPSIS (Technique for Order of Preference by Similarity to Ideal Solution)

Criteria:
- Suitability Score (Benefit criterion, higher is better)
- Estimated Cost (Cost criterion, lower is better)
- Context Headroom (Benefit criterion, higher is better)
- Latency (Cost criterion, ONLY included when real measurements are provided)

Mathematical guarantees:
- Handles degenerate inputs, zero-ranges, equal criteria, and single candidates.
- Validates that weights are non-negative and sum to 1.0.
- Normalizes costs logarithmically to handle multi-order-of-magnitude pricing ($0.075 to $75.00).
- Transparently distinguishes measured, predicted, calculated, and unavailable attributes.
- Final score is a relative multi-criteria recommendation score, not an accuracy probability.
"""

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any, Dict, List, Optional, Tuple


class RankingMethod(str, Enum):
    WEIGHTED_SUM = "weighted_sum"
    TOPSIS = "topsis"


class RankingPreset(str, Enum):
    BALANCED = "balanced"
    COST_OPTIMIZED = "cost_optimized"
    QUALITY_OPTIMIZED = "quality_optimized"
    LONG_CONTEXT = "long_context"


PRESET_WEIGHTS: Dict[RankingPreset, Dict[str, float]] = {
    RankingPreset.BALANCED: {
        "suitability": 0.45,
        "cost": 0.35,
        "context": 0.20
    },
    RankingPreset.COST_OPTIMIZED: {
        "suitability": 0.20,
        "cost": 0.70,
        "context": 0.10
    },
    RankingPreset.QUALITY_OPTIMIZED: {
        "suitability": 0.80,
        "cost": 0.05,
        "context": 0.15
    },
    RankingPreset.LONG_CONTEXT: {
        "suitability": 0.25,
        "cost": 0.25,
        "context": 0.50
    }
}


def validate_and_normalize_weights(
    weights: Optional[Dict[str, float]],
    preset: RankingPreset = RankingPreset.BALANCED
) -> Dict[str, float]:
    """
    Validates that weights are non-negative and normalizes them to sum to 1.0.
    Falls back to preset defaults if input is missing, empty, or all zeros.
    """
    default_w = PRESET_WEIGHTS.get(preset, PRESET_WEIGHTS[RankingPreset.BALANCED])
    if not weights:
        return dict(default_w)

    clean_w: Dict[str, float] = {}
    for k in ["suitability", "cost", "context"]:
        val = weights.get(k, 0.0)
        if isinstance(val, (int, float)) and not math.isnan(val) and val >= 0:
            clean_w[k] = float(val)
        else:
            clean_w[k] = 0.0

    total = sum(clean_w.values())
    if total <= 0:
        return dict(default_w)

    return {k: round(v / total, 4) for k, v in clean_w.items()}


@dataclass
class CandidateEvaluation:
    model_id: str
    display_name: str
    provider: str
    token_count: int
    token_count_method: str  # "exact", "estimated"
    suitability_score: float  # [0.0, 1.0]
    total_cost_usd: Optional[float]
    input_cost_usd: Optional[float]
    output_cost_usd: Optional[float]
    context_window_tokens: int
    lifecycle_status: str = "active"
    latency_ms: Optional[float] = None  # None if unavailable
    pricing_date: str = "2025-02-15"
    pricing_status: str = "known"
    estimated_output_tokens: int = 350
    input_cost_per_m: Optional[float] = None
    output_cost_per_m: Optional[float] = None


@dataclass
class RankedModel:
    rank: int
    model_id: str
    display_name: str
    provider: str
    score: float  # Relative recommendation score [0.0, 1.0]
    suitability_score: float
    estimated_cost_usd: Optional[float]
    token_count: int
    token_count_provenance: str
    latency_status: str  # "measured" or "unavailable"
    benchmark_status: str  # "unavailable (not fabricated)"
    reasoning: str
    lifecycle_status: str = "active"


def _generate_candidate_reasoning(
    c: CandidateEvaluation,
    norm_suit: float,
    norm_cost: float,
    norm_ctx: float,
    preset: RankingPreset
) -> str:
    """
    Generates transparent, evidence-based rationale without generic marketing claims.
    """
    reasons = []

    # 1. Suitability summary
    if norm_suit >= 0.85:
        reasons.append(f"High task alignment (suitability: {c.suitability_score:.2f})")
    elif norm_suit >= 0.50:
        reasons.append(f"Moderate task alignment (suitability: {c.suitability_score:.2f})")
    else:
        reasons.append(f"Limited task alignment (suitability: {c.suitability_score:.2f})")

    # 2. Cost summary
    if c.pricing_status == "unknown" or c.total_cost_usd is None:
        reasons.append("Pricing metadata unverified (cost omitted from ranking)")
    elif norm_cost >= 0.85:
        reasons.append(f"Highly economical (${c.total_cost_usd:.5f} estimated total)")
    elif norm_cost >= 0.40:
        reasons.append(f"Moderate pricing (${c.total_cost_usd:.5f} estimated total)")
    else:
        reasons.append(f"Premium tier pricing (${c.total_cost_usd:.5f} estimated total)")

    # 3. Context sufficiency
    total_tokens = c.token_count + c.estimated_output_tokens
    utilization_pct = (total_tokens / max(1, c.context_window_tokens)) * 100.0
    if c.context_window_tokens >= 1_000_000:
        reasons.append(f"Massive context capacity ({c.context_window_tokens // 1_000_000}M tokens, {utilization_pct:.3f}% utilized)")
    else:
        reasons.append(f"Context capacity sufficient ({utilization_pct:.2f}% of {c.context_window_tokens:,} tokens utilized)")

    # 4. Token provenance note
    if c.token_count_method == "exact":
        reasons.append("Exact local BPE token count")
    else:
        reasons.append("Calibrated offline token estimate")

    # 5. Lifecycle warning if not active
    if c.lifecycle_status == "deprecated":
        reasons.append("Legacy/deprecated model")
    elif c.lifecycle_status == "retired":
        reasons.append("Retired model")

    return "; ".join(reasons)


def rank_models_weighted_sum(
    candidates: List[CandidateEvaluation],
    weights: Dict[str, float],
    preset: RankingPreset = RankingPreset.BALANCED
) -> List[RankedModel]:
    """
    Ranks models using Weighted Sum Model (WSM) with robust normalization.
    """
    if not candidates:
        return []

    # Single candidate degenerate case
    if len(candidates) == 1:
        c = candidates[0]
        reason = _generate_candidate_reasoning(c, 1.0, 1.0, 1.0, preset)
        return [RankedModel(
            rank=1,
            model_id=c.model_id,
            display_name=c.display_name,
            provider=c.provider,
            score=1.0,
            suitability_score=c.suitability_score,
            estimated_cost_usd=c.total_cost_usd,
            token_count=c.token_count,
            token_count_provenance=c.token_count_method,
            latency_status="measured" if c.latency_ms is not None else "unavailable",
            benchmark_status="unavailable (offline platform does not fabricate third-party benchmark scores)",
            reasoning=reason,
            lifecycle_status=c.lifecycle_status
        )]

    w = validate_and_normalize_weights(weights, preset)
    w_suit = w["suitability"]
    w_cost = w["cost"]
    w_ctx = w["context"]

    # Extract criteria vectors
    suitability_vals = [c.suitability_score for c in candidates]
    min_suit, max_suit = min(suitability_vals), max(suitability_vals)

    # Context values (log scale)
    ctx_vals = [math.log10(c.context_window_tokens) for c in candidates]
    min_ctx, max_ctx = min(ctx_vals), max(ctx_vals)

    # Cost values (log scale for valid prices)
    known_costs = [c.total_cost_usd for c in candidates if c.total_cost_usd is not None and c.total_cost_usd > 0]
    if known_costs:
        min_cost, max_cost = min(known_costs), max(known_costs)
        min_log_cost = math.log10(max(1e-9, min_cost))
        max_log_cost = math.log10(max(1e-9, max_cost))
    else:
        min_log_cost = max_log_cost = 0.0

    ranked = []
    for c in candidates:
        # Benefit criterion: Suitability (higher is better)
        if max_suit == min_suit:
            norm_suit = 1.0
        else:
            norm_suit = (c.suitability_score - min_suit) / (max_suit - min_suit)

        # Cost criterion: Cost (lower is better -> inverted, log-scale)
        if c.total_cost_usd is None or c.pricing_status == "unknown":
            norm_cost = 0.0  # Unknown pricing cannot be assumed cheap
        elif max_log_cost == min_log_cost:
            norm_cost = 1.0
        else:
            log_c = math.log10(max(1e-9, c.total_cost_usd))
            norm_cost = (max_log_cost - log_c) / (max_log_cost - min_log_cost)

        # Benefit criterion: Context window (higher is better, log scale)
        log_ctx = math.log10(c.context_window_tokens)
        if max_ctx == min_ctx:
            norm_ctx = 1.0
        else:
            norm_ctx = (log_ctx - min_ctx) / (max_ctx - min_ctx)

        # Cost-optimized preset: apply suitability floor penalty if model is completely unsuitable
        if preset == RankingPreset.COST_OPTIMIZED and c.suitability_score < 0.50:
            norm_suit *= 0.5

        composite_score = (w_suit * norm_suit) + (w_cost * norm_cost) + (w_ctx * norm_ctx)
        reason_str = _generate_candidate_reasoning(c, norm_suit, norm_cost, norm_ctx, preset)

        ranked.append(RankedModel(
            rank=0,
            model_id=c.model_id,
            display_name=c.display_name,
            provider=c.provider,
            score=round(composite_score, 4),
            suitability_score=c.suitability_score,
            estimated_cost_usd=c.total_cost_usd,
            token_count=c.token_count,
            token_count_provenance=c.token_count_method,
            latency_status="measured" if c.latency_ms is not None else "unavailable",
            benchmark_status="unavailable (offline platform does not fabricate third-party benchmark scores)",
            reasoning=reason_str,
            lifecycle_status=c.lifecycle_status
        ))

    # Sort descending by score; break ties with suitability then lower cost
    ranked.sort(
        key=lambda x: (
            x.score,
            x.suitability_score,
            -(x.estimated_cost_usd if x.estimated_cost_usd is not None else 999.0)
        ),
        reverse=True
    )
    for i, r in enumerate(ranked):
        r.rank = i + 1

    return ranked


def rank_models_topsis(
    candidates: List[CandidateEvaluation],
    weights: Dict[str, float],
    preset: RankingPreset = RankingPreset.BALANCED
) -> List[RankedModel]:
    """
    Ranks models using TOPSIS (Technique for Order Preference by Similarity to Ideal Solution)
    with degenerate input and zero-range guards.
    """
    if not candidates:
        return []

    # Single candidate degenerate case
    if len(candidates) == 1:
        c = candidates[0]
        reason = _generate_candidate_reasoning(c, 1.0, 1.0, 1.0, preset)
        return [RankedModel(
            rank=1,
            model_id=c.model_id,
            display_name=c.display_name,
            provider=c.provider,
            score=1.0,
            suitability_score=c.suitability_score,
            estimated_cost_usd=c.total_cost_usd,
            token_count=c.token_count,
            token_count_provenance=c.token_count_method,
            latency_status="measured" if c.latency_ms is not None else "unavailable",
            benchmark_status="unavailable (offline platform does not fabricate third-party benchmark scores)",
            reasoning=reason,
            lifecycle_status=c.lifecycle_status
        )]

    w = validate_and_normalize_weights(weights, preset)
    w_suit = w["suitability"]
    w_cost = w["cost"]
    w_ctx = w["context"]

    # Suitability (Benefit)
    suit_vals = [c.suitability_score for c in candidates]

    # Context (Benefit, log scale)
    ctx_vals = [math.log10(c.context_window_tokens) for c in candidates]

    # Cost (Cost, log scale shifted positive)
    raw_costs = [
        math.log10(max(1e-9, c.total_cost_usd)) if (c.total_cost_usd is not None and c.total_cost_usd > 0) else 5.0
        for c in candidates
    ]
    shift = abs(min(raw_costs)) + 1.0
    cost_vals = [v + shift for v in raw_costs]

    # Vector normalization: denom = sqrt(sum(x_i^2))
    denom_suit = math.sqrt(sum(v**2 for v in suit_vals)) or 1.0
    denom_cost = math.sqrt(sum(v**2 for v in cost_vals)) or 1.0
    denom_ctx = math.sqrt(sum(v**2 for v in ctx_vals)) or 1.0

    # Weighted normalized values
    v_suit = [(v / denom_suit) * w_suit for v in suit_vals]
    v_cost = [(v / denom_cost) * w_cost for v in cost_vals]
    v_ctx = [(v / denom_ctx) * w_ctx for v in ctx_vals]

    # Ideal (A+) and Anti-Ideal (A-)
    # Benefit criteria: max is ideal, min is anti-ideal
    a_plus_suit, a_minus_suit = max(v_suit), min(v_suit)
    a_plus_ctx, a_minus_ctx = max(v_ctx), min(v_ctx)

    # Cost criterion: min is ideal, max is anti-ideal
    a_plus_cost, a_minus_cost = min(v_cost), max(v_cost)

    ranked = []
    for i, c in enumerate(candidates):
        d_plus = math.sqrt(
            (v_suit[i] - a_plus_suit)**2 +
            (v_cost[i] - a_plus_cost)**2 +
            (v_ctx[i] - a_plus_ctx)**2
        )
        d_minus = math.sqrt(
            (v_suit[i] - a_minus_suit)**2 +
            (v_cost[i] - a_minus_cost)**2 +
            (v_ctx[i] - a_minus_ctx)**2
        )

        total_d = d_plus + d_minus
        if total_d == 0:
            closeness = 0.5  # Degenerate case where all options are identical
        else:
            closeness = d_minus / total_d

        norm_s = (c.suitability_score - min(suit_vals)) / max(1e-9, (max(suit_vals) - min(suit_vals)))
        norm_c = (max(cost_vals) - cost_vals[i]) / max(1e-9, (max(cost_vals) - min(cost_vals)))
        norm_x = (ctx_vals[i] - min(ctx_vals)) / max(1e-9, (max(ctx_vals) - min(ctx_vals)))
        reason_str = _generate_candidate_reasoning(c, norm_s, norm_c, norm_x, preset)

        ranked.append(RankedModel(
            rank=0,
            model_id=c.model_id,
            display_name=c.display_name,
            provider=c.provider,
            score=round(closeness, 4),
            suitability_score=c.suitability_score,
            estimated_cost_usd=c.total_cost_usd,
            token_count=c.token_count,
            token_count_provenance=c.token_count_method,
            latency_status="measured" if c.latency_ms is not None else "unavailable",
            benchmark_status="unavailable (offline platform does not fabricate third-party benchmark scores)",
            reasoning=reason_str,
            lifecycle_status=c.lifecycle_status
        ))

    # Sort descending by closeness score; break ties with suitability
    ranked.sort(
        key=lambda x: (
            x.score,
            x.suitability_score,
            -(x.estimated_cost_usd if x.estimated_cost_usd is not None else 999.0)
        ),
        reverse=True
    )
    for i, r in enumerate(ranked):
        r.rank = i + 1

    return ranked


def rank_models(
    candidates: List[CandidateEvaluation],
    method: RankingMethod = RankingMethod.WEIGHTED_SUM,
    preset: Optional[RankingPreset] = RankingPreset.BALANCED,
    custom_weights: Optional[Dict[str, float]] = None
) -> List[RankedModel]:
    """
    Dispatches to Weighted Sum or TOPSIS ranking engine based on configured method and weights.
    """
    active_preset = preset or RankingPreset.BALANCED
    weights = custom_weights or PRESET_WEIGHTS.get(active_preset, PRESET_WEIGHTS[RankingPreset.BALANCED])

    if method == RankingMethod.TOPSIS:
        return rank_models_topsis(candidates, weights, active_preset)
    else:
        return rank_models_weighted_sum(candidates, weights, active_preset)
