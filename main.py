from typing import List, Optional
from fastapi import FastAPI, HTTPException, Query

from models.schemas import (
    PromptRequest,
    ModelComparisonRequest,
    ModelComparisonResponse,
    ModelMetadataSchema
)
from analyzers.prompt_analyzer import analyze_prompt
from analyzers.model_comparison import (
    comparison_engine,
    registry,
    RankingMethod,
    RankingPreset
)
from analyzers.model_comparison.registry import LifecycleStatus

app = FastAPI(
    title="PromptShield AI",
    description="Multi-model AI security analysis and local model-comparison backend evaluating prompt injection, application risks, token costs, and model suitability.",
    version="1.1.0"
)


@app.get("/")
def home():
    return {"message": "PromptShield AI backend is running"}


@app.post("/analyze")
def analyze(data: PromptRequest):
    mode = data.analysis_mode.value if hasattr(data.analysis_mode, "value") else str(data.analysis_mode)
    return analyze_prompt(data.prompt, analysis_mode=mode)


# ---------------------------------------------------------------------------
# Model Comparison & Ranking Endpoints
# ---------------------------------------------------------------------------

@app.get("/models", response_model=List[ModelMetadataSchema])
def list_models(
    provider: Optional[str] = Query(None, description="Filter by provider: openai, google, anthropic"),
    include_deprecated: bool = Query(False, description="Whether to include deprecated or retired models (default: False)"),
    status: Optional[str] = Query(None, description="Filter by specific lifecycle status: active, deprecated, retired, preview, unverified")
):
    """
    Returns registered foundation models, context windows, tokenizer specs, lifecycle status,
    and dated pricing. By default, only active/eligible models are exposed.
    """
    lifecycle_enum = None
    if status:
        try:
            lifecycle_enum = LifecycleStatus(status.lower())
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid status '{status}'. Valid values: active, deprecated, retired, preview, unverified."
            )

    models = registry.list_models(
        provider=provider,
        include_deprecated=include_deprecated,
        lifecycle_status=lifecycle_enum
    )

    return [
        ModelMetadataSchema(
            model_id=m.model_id,
            display_name=m.display_name,
            provider=m.provider,
            family=m.family,
            context_window_tokens=m.context_window_tokens,
            max_output_tokens=m.max_output_tokens,
            input_cost_per_m=m.pricing.input_cost_per_m,
            output_cost_per_m=m.pricing.output_cost_per_m,
            pricing_effective_date=m.pricing.pricing_effective_date,
            currency=m.pricing.currency,
            tokenizer_name=m.tokenizer_name,
            capabilities=m.capabilities,
            strengths=m.strengths,
            description=m.description,
            lifecycle_status=m.lifecycle_status.value if hasattr(m.lifecycle_status, "value") else str(m.lifecycle_status),
            source_url=m.source_url,
            verification_date=m.verification_date,
            deprecation_date=m.deprecation_date
        )
        for m in models
    ]


@app.get("/models/{model_id}", response_model=ModelMetadataSchema)
def get_model(model_id: str):
    """
    Returns metadata, lifecycle status, and pricing for a specific registered model ID.
    Supports querying both active and historical/deprecated models by ID.
    """
    m = registry.get_model(model_id)
    if not m:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_id}' not found in registry. Check /models for registered model IDs."
        )
    return ModelMetadataSchema(
        model_id=m.model_id,
        display_name=m.display_name,
        provider=m.provider,
        family=m.family,
        context_window_tokens=m.context_window_tokens,
        max_output_tokens=m.max_output_tokens,
        input_cost_per_m=m.pricing.input_cost_per_m,
        output_cost_per_m=m.pricing.output_cost_per_m,
        pricing_effective_date=m.pricing.pricing_effective_date,
        currency=m.pricing.currency,
        tokenizer_name=m.tokenizer_name,
        capabilities=m.capabilities,
        strengths=m.strengths,
        description=m.description,
        lifecycle_status=m.lifecycle_status.value if hasattr(m.lifecycle_status, "value") else str(m.lifecycle_status),
        source_url=m.source_url,
        verification_date=m.verification_date,
        deprecation_date=m.deprecation_date
    )


@app.post("/compare-models", response_model=ModelComparisonResponse)
def compare_models_endpoint(req: ModelComparisonRequest):
    """
    Evaluates a prompt across registered foundation models (GPT, Gemini, Claude).
    Calculates exact/calibrated token counts, input/output costs, suitability, and
    multi-criteria WSM / TOPSIS ranking without making external API calls.
    Does not transmit user prompts to third-party model generation APIs.
    """
    method_val = req.method.value if hasattr(req.method, "value") else str(req.method)
    preset_val = req.preset.value if hasattr(req.preset, "value") else str(req.preset)

    method = RankingMethod(method_val)
    preset = RankingPreset(preset_val)

    report = comparison_engine.compare_prompt(
        prompt=req.prompt,
        method=method,
        preset=preset,
        custom_weights=req.custom_weights,
        model_ids=req.models,
        output_tokens_override=req.expected_output_tokens,
        include_deprecated=bool(req.include_deprecated)
    )

    # Convert dataclasses to dicts matching Pydantic response schema
    model_details_dict = {}
    for mid, d in report.model_details.items():
        model_details_dict[mid] = {
            "model_id": d.model_id,
            "display_name": d.display_name,
            "provider": d.provider,
            "context_window_tokens": d.context_window_tokens,
            "token_count": d.token_count,
            "token_count_method": d.token_count_method,
            "is_exact_token_count": d.is_exact_token_count,
            "estimated_output_tokens": d.estimated_output_tokens,
            "cost_breakdown": {
                "input_tokens": d.cost_breakdown.input_tokens,
                "estimated_output_tokens": d.cost_breakdown.estimated_output_tokens,
                "input_cost_usd": d.cost_breakdown.input_cost_usd,
                "output_cost_usd": d.cost_breakdown.output_cost_usd,
                "total_estimated_cost_usd": d.cost_breakdown.total_estimated_cost_usd,
                "input_cost_per_m": d.cost_breakdown.input_cost_per_m,
                "output_cost_per_m": d.cost_breakdown.output_cost_per_m,
                "pricing_effective_date": d.cost_breakdown.pricing_effective_date,
                "currency": d.cost_breakdown.currency,
                "pricing_status": d.cost_breakdown.pricing_status
            },
            "suitability_score": d.suitability_score,
            "capabilities": d.capabilities,
            "strengths": d.strengths,
            "provenance": d.provenance,
            "lifecycle_status": d.lifecycle_status,
            "source_url": d.source_url,
            "verification_date": d.verification_date
        }

    ranked_models_list = [
        {
            "rank": rm.rank,
            "model_id": rm.model_id,
            "display_name": rm.display_name,
            "provider": rm.provider,
            "score": rm.score,
            "suitability_score": rm.suitability_score,
            "estimated_cost_usd": rm.estimated_cost_usd,
            "token_count": rm.token_count,
            "token_count_provenance": rm.token_count_provenance,
            "latency_status": rm.latency_status,
            "benchmark_status": rm.benchmark_status,
            "reasoning": rm.reasoning,
            "lifecycle_status": rm.lifecycle_status
        }
        for rm in report.ranked_models
    ]

    return {
        "status": "success",
        "prompt_text": report.prompt_text,
        "prompt_features": {
            "char_length": report.prompt_features.char_length,
            "word_count": report.prompt_features.word_count,
            "line_count": report.prompt_features.line_count,
            "code_proportion": report.prompt_features.code_proportion,
            "has_code_block": report.prompt_features.has_code_block,
            "complexity_score": report.prompt_features.complexity_score,
            "technical_keyword_density": report.prompt_features.technical_keyword_density,
            "security_sensitivity": report.prompt_features.security_sensitivity,
            "estimated_output_tokens": report.prompt_features.estimated_output_tokens,
            "detected_indicators": report.prompt_features.detected_indicators
        },
        "task_classification": {
            "predicted_category": report.task_classification.predicted_category,
            "confidence": report.task_classification.confidence,
            "category_probabilities": report.task_classification.category_probabilities,
            "model_source": report.task_classification.model_source,
            "is_low_confidence": report.task_classification.is_low_confidence,
            "top_categories": report.task_classification.top_categories,
            "confidence_status": report.task_classification.confidence_status
        },
        "ranking_method": report.ranking_method,
        "ranking_preset": report.ranking_preset,
        "applied_weights": report.applied_weights,
        "ranked_models": ranked_models_list,
        "model_details": model_details_dict,
        "provenance_declarations": report.provenance_declarations
    }