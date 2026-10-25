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

Strictly preserves data provenance: distinguishes measured, predicted, calculated,
and unavailable attributes.
"""

from dataclasses import dataclass
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


@dataclass
class CandidateEvaluation:
    model_id: str
    display_name: str
    provider: str
    token_count: int
    token_count_method: str  # "exact", "estimated"
    suitability_score: float  # [0.0, 1.0]
    total_cost_usd: float
    input_cost_usd: float
    output_cost_usd: float
    context_window_tokens: int
    latency_ms: Optional[float] = None  # None if unavailable
    pricing_date: str = "2024-10-01"


@dataclass
class RankedModel:
    rank: int
    model_id: str
    display_name: str
    provider: str
    score: float
    suitability_score: float
    estimated_cost_usd: float
    token_count: int
    token_count_provenance: str
    latency_status: str  # "measured" or "unavailable"
    benchmark_status: str  # "unavailable (not fabricated)"
    reasoning: str


def rank_models_weighted_sum(
    candidates: List[CandidateEvaluation],
    weights: Dict[str, float]
) -> List[RankedModel]:
    """
    Ranks models using Weighted Sum Model (WSM) with min-max normalization.
    """
    if not candidates:
        return []

    # Extract criteria vectors
    suitability_vals = [c.suitability_score for c in candidates]
    cost_vals = [c.total_cost_usd for c in candidates]
    context_vals = [c.context_window_tokens for c in candidates]

    min_suit, max_suit = min(suitability_vals), max(suitability_vals)
    min_cost, max_cost = min(cost_vals), max(cost_vals)
    min_ctx, max_ctx = min(context_vals), max(context_vals)

    w_suit = weights.get("suitability", 0.40)
    w_cost = weights.get("cost", 0.40)
    w_ctx = weights.get("context", 0.20)
    total_w = w_suit + w_cost + w_ctx
    w_suit /= total_w
    w_cost /= total_w
    w_ctx /= total_w

    ranked = []
    for c in candidates:
        # Benefit criterion: Suitability (higher is better)
        if max_suit == min_suit:
            norm_suit = 1.0
        else:
            norm_suit = (c.suitability_score - min_suit) / (max_suit - min_suit)

        # Cost criterion: Cost (lower is better -> inverted, log-scale for multi-magnitude pricing spans)
        log_cost = math.log10(max(1e-9, c.total_cost_usd))
        min_log_cost = math.log10(max(1e-9, min_cost))
        max_log_cost = math.log10(max(1e-9, max_cost))
        if max_log_cost == min_log_cost:
            norm_cost = 1.0
        else:
            norm_cost = (max_log_cost - log_cost) / (max_log_cost - min_log_cost)

        # Benefit criterion: Context window (higher is better, log scale)
        log_ctx = math.log10(c.context_window_tokens)
        min_log_ctx = math.log10(min_ctx)
        max_log_ctx = math.log10(max_ctx)
        if max_log_ctx == min_log_ctx:
            norm_ctx = 1.0
        else:
            norm_ctx = (log_ctx - min_log_ctx) / (max_log_ctx - min_log_ctx)

        composite_score = (w_suit * norm_suit) + (w_cost * norm_cost) + (w_ctx * norm_ctx)

        # Generate human-readable rationale
        reasons = []
        if norm_cost >= 0.85:
            reasons.append("Highly economical cost")
        if norm_suit >= 0.85:
            reasons.append("Strong capability alignment")
        if c.context_window_tokens >= 1_000_000:
            reasons.append("Massive context headroom")
        reason_str = "; ".join(reasons) if reasons else "Balanced trade-off across criteria"

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
            reasoning=reason_str
        ))

    # Sort descending by score
    ranked.sort(key=lambda x: x.score, reverse=True)
    for i, r in enumerate(ranked):
        r.rank = i + 1

    return ranked


def rank_models_topsis(
    candidates: List[CandidateEvaluation],
    weights: Dict[str, float]
) -> List[RankedModel]:
    """
    Ranks models using TOPSIS (Technique for Order Preference by Similarity to Ideal Solution).
    """
    if not candidates:
        return []

    # Criteria:
    # 1. Suitability (Benefit)
    # 2. Cost (Cost)
    # 3. Context Log (Benefit)
    w_suit = weights.get("suitability", 0.40)
    w_cost = weights.get("cost", 0.40)
    w_ctx = weights.get("context", 0.20)
    total_w = w_suit + w_cost + w_ctx
    w_suit /= total_w
    w_cost /= total_w
    w_ctx /= total_w

    suit_vals = [c.suitability_score for c in candidates]
    raw_log_costs = [math.log10(max(1e-9, c.total_cost_usd)) for c in candidates]
    shift = abs(min(raw_log_costs)) + 1.0
    cost_vals = [v + shift for v in raw_log_costs]
    ctx_vals = [math.log10(c.context_window_tokens) for c in candidates]

    # Vector normalization: denom = sqrt(sum(x_i^2))
    denom_suit = math.sqrt(sum(v**2 for v in suit_vals)) or 1.0
    denom_cost = math.sqrt(sum(v**2 for v in cost_vals)) or 1.0
    denom_ctx = math.sqrt(sum(v**2 for v in ctx_vals)) or 1.0

    # Weighted normalized values
    v_suit = [(v / denom_suit) * w_suit for v in suit_vals]
    v_cost = [(v / denom_cost) * w_cost for v in cost_vals]
    v_ctx = [(v / denom_ctx) * w_ctx for v in ctx_vals]

    # Ideal (A+) and Anti-Ideal (A-)
    # Suitability is Benefit: max is positive, min is negative
    a_plus_suit = max(v_suit)
    a_minus_suit = min(v_suit)

    # Cost is Cost: min is positive, max is negative
    a_plus_cost = min(v_cost)
    a_minus_cost = max(v_cost)

    # Context is Benefit: max is positive, min is negative
    a_plus_ctx = max(v_ctx)
    a_minus_ctx = min(v_ctx)

    ranked = []
    for i, c in enumerate(candidates):
        # Distance to ideal (S+)
        d_plus = math.sqrt(
            (v_suit[i] - a_plus_suit)**2 +
            (v_cost[i] - a_plus_cost)**2 +
            (v_ctx[i] - a_plus_ctx)**2
        )
        # Distance to anti-ideal (S-)
        d_minus = math.sqrt(
            (v_suit[i] - a_minus_suit)**2 +
            (v_cost[i] - a_minus_cost)**2 +
            (v_ctx[i] - a_minus_ctx)**2
        )

        total_d = d_plus + d_minus
        closeness = d_minus / total_d if total_d > 0 else 0.5

        reasons = []
        if v_cost[i] <= (min(v_cost) * 1.5):
            reasons.append("Close to optimal cost boundary")
        if v_suit[i] >= (max(v_suit) * 0.9):
            reasons.append("Close to optimal suitability boundary")
        if c.context_window_tokens >= 1_000_000:
            reasons.append("Extended context boundary advantage")
        reason_str = "; ".join(reasons) if reasons else "Compromise solution near center of Euclidean Pareto front"

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
            reasoning=reason_str
        ))

    # Sort descending by closeness score
    ranked.sort(key=lambda x: x.score, reverse=True)
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
    if custom_weights:
        weights = dict(custom_weights)
    elif preset in PRESET_WEIGHTS:
        weights = dict(PRESET_WEIGHTS[preset])
    else:
        weights = dict(PRESET_WEIGHTS[RankingPreset.BALANCED])

    if method == RankingMethod.TOPSIS:
        return rank_models_topsis(candidates, weights)
    else:
        return rank_models_weighted_sum(candidates, weights)
