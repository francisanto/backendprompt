"""
PromptShield AI — Model Comparison Engine Orchestrator.
======================================================
Coordinates local feature extraction, task classification, token counting,
cost estimation, suitability scoring, and multi-criteria ranking across
GPT, Gemini, and Claude foundation models without external API calls.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from analyzers.model_comparison.registry import ModelMetadata, registry, LifecycleStatus
from analyzers.model_comparison.tokenizer_adapters import get_tokenizer_adapter
from analyzers.model_comparison.feature_extractor import PromptFeatures, feature_extractor
from analyzers.model_comparison.task_classifier import TaskClassificationResult, task_classifier
from analyzers.model_comparison.cost_calculator import CostBreakdown, calculate_model_cost
from analyzers.model_comparison.suitability import calculate_suitability
from analyzers.model_comparison.ranker import (
    CandidateEvaluation,
    RankedModel,
    RankingMethod,
    RankingPreset,
    rank_models,
    validate_and_normalize_weights
)


@dataclass
class ModelComparisonDetails:
    model_id: str
    display_name: str
    provider: str
    context_window_tokens: int
    token_count: int
    token_count_method: str  # "exact", "estimated", "unsupported"
    is_exact_token_count: bool
    estimated_output_tokens: int
    cost_breakdown: CostBreakdown
    suitability_score: float
    capabilities: List[str]
    strengths: List[str]
    provenance: Dict[str, str]
    lifecycle_status: str = "active"
    source_url: str = ""
    verification_date: str = "2025-02-15"


@dataclass
class ModelComparisonReport:
    prompt_text: str
    prompt_features: PromptFeatures
    task_classification: TaskClassificationResult
    ranking_method: str
    ranking_preset: str
    applied_weights: Dict[str, float]
    ranked_models: List[RankedModel]
    model_details: Dict[str, ModelComparisonDetails]
    provenance_declarations: Dict[str, str]


class ModelComparisonEngine:
    """Orchestrates multi-model analysis, costing, and ranking."""

    def __init__(self):
        self.registry = registry
        self.feature_extractor = feature_extractor
        self.task_classifier = task_classifier

    def compare_prompt(
        self,
        prompt: str,
        method: RankingMethod = RankingMethod.WEIGHTED_SUM,
        preset: RankingPreset = RankingPreset.BALANCED,
        custom_weights: Optional[Dict[str, float]] = None,
        model_ids: Optional[List[str]] = None,
        output_tokens_override: Optional[int] = None,
        include_deprecated: bool = False
    ) -> ModelComparisonReport:
        """
        Runs local prompt analysis, feature extraction, token counting, costing,
        and ranking across foundation models.
        """
        # 1. Feature Extraction
        features = self.feature_extractor.extract_features(prompt)
        output_tokens = output_tokens_override or features.estimated_output_tokens

        # 2. Local Task Classification
        task_res = self.task_classifier.classify_task(prompt)

        # 3. Model Selection
        # Single consistent registry snapshot for the entire request
        if model_ids:
            target_models = []
            for mid in model_ids:
                m = self.registry.get_model(mid)
                if m:
                    target_models.append(m)
            # If none matched, fallback to eligible models
            if not target_models:
                target_models = self.registry.list_models(include_deprecated=include_deprecated)
        else:
            # Default recommendation pool: excludes deprecated/retired models unless requested
            target_models = self.registry.list_models(include_deprecated=include_deprecated)

        # 4. Token Counting, Cost Calculation, and Suitability Scoring
        candidates: List[CandidateEvaluation] = []
        details_map: Dict[str, ModelComparisonDetails] = {}

        for model in target_models:
            adapter = get_tokenizer_adapter(model.model_id)
            tok_res = adapter.count_tokens(prompt)

            cost = calculate_model_cost(
                model=model,
                input_tokens=tok_res.token_count,
                estimated_output_tokens=output_tokens
            )

            suitability = calculate_suitability(
                model=model,
                features=features,
                predicted_task=task_res.predicted_category,
                task_confidence=task_res.confidence,
                task_result=task_res
            )

            candidates.append(CandidateEvaluation(
                model_id=model.model_id,
                display_name=model.display_name,
                provider=model.provider,
                token_count=tok_res.token_count,
                token_count_method=tok_res.method,
                suitability_score=suitability,
                total_cost_usd=cost.total_estimated_cost_usd,
                input_cost_usd=cost.input_cost_usd,
                output_cost_usd=cost.output_cost_usd,
                context_window_tokens=model.context_window_tokens,
                lifecycle_status=model.lifecycle_status.value if hasattr(model.lifecycle_status, "value") else str(model.lifecycle_status),
                latency_ms=None,  # Offline engine does not invent live latency
                pricing_date=cost.pricing_effective_date,
                pricing_status=cost.pricing_status,
                estimated_output_tokens=output_tokens,
                input_cost_per_m=model.pricing.input_cost_per_m,
                output_cost_per_m=model.pricing.output_cost_per_m
            ))

            details_map[model.model_id] = ModelComparisonDetails(
                model_id=model.model_id,
                display_name=model.display_name,
                provider=model.provider,
                context_window_tokens=model.context_window_tokens,
                token_count=tok_res.token_count,
                token_count_method=tok_res.method,
                is_exact_token_count=tok_res.is_exact,
                estimated_output_tokens=output_tokens,
                cost_breakdown=cost,
                suitability_score=suitability,
                capabilities=model.capabilities,
                strengths=model.strengths,
                provenance={
                    "token_count": "exact_local_bpe" if tok_res.is_exact else "calibrated_offline_estimate",
                    "input_cost": "calculated_from_token_count_and_published_pricing" if cost.pricing_status == "known" else "unknown_unverified",
                    "output_cost": "estimated_from_heuristic_response_length_and_published_pricing" if cost.pricing_status == "known" else "unknown_unverified",
                    "task_category": f"predicted_by_local_ml_model (confidence: {task_res.confidence})",
                    "suitability_score": "heuristically_predicted_from_structural_features_and_task_alignment",
                    "latency": "unavailable (not measured or fabricated offline)",
                    "external_benchmark_scores": "unavailable (not fabricated offline)",
                    "billing_note": tok_res.billing_disclaimer
                },
                lifecycle_status=model.lifecycle_status.value if hasattr(model.lifecycle_status, "value") else str(model.lifecycle_status),
                source_url=model.source_url,
                verification_date=model.verification_date
            )

        # 5. Multi-Criteria Ranking
        ranked = rank_models(
            candidates=candidates,
            method=method,
            preset=preset,
            custom_weights=custom_weights
        )

        effective_weights = validate_and_normalize_weights(custom_weights, preset)

        return ModelComparisonReport(
            prompt_text=prompt,
            prompt_features=features,
            task_classification=task_res,
            ranking_method=method.value if hasattr(method, "value") else str(method),
            ranking_preset=preset.value if hasattr(preset, "value") else str(preset),
            applied_weights=effective_weights,
            ranked_models=ranked,
            model_details=details_map,
            provenance_declarations={
                "offline_execution": "100% local execution; zero prompts transmitted to external model APIs.",
                "token_counting": "OpenAI token counts are exact via official tiktoken; Gemini and Claude token counts are calibrated offline estimates.",
                "pricing_data": "Public pricing snapshot verified against official documentation (Feb 2025; USD per 1M tokens).",
                "benchmarks": "No third-party quality benchmarks or live latencies are fabricated.",
                "scoring_interpretation": "Scores represent relative multi-criteria trade-off rankings under selected preset, not absolute correctness probabilities."
            }
        )


# Singleton engine instance
comparison_engine = ModelComparisonEngine()
