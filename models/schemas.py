from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class AnalysisMode(str, Enum):
    PROMPT_SECURITY = "prompt_security"
    APPLICATION_RISK = "application_risk"
    BOTH = "both"


class PromptRequest(BaseModel):
    prompt: str = Field(
        ...,
        description="The prompt text to analyze"
    )
    analysis_mode: Optional[AnalysisMode] = Field(
        default=AnalysisMode.BOTH,
        description="Analysis mode: 'prompt_security', 'application_risk', or 'both'"
    )

    @field_validator("prompt")
    @classmethod
    def check_prompt(cls, v: str) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("Prompt cannot be empty or contain only whitespace.")
        return v


# ---------------------------------------------------------------------------
# Model Comparison Schemas
# ---------------------------------------------------------------------------

class RankingMethodEnum(str, Enum):
    WEIGHTED_SUM = "weighted_sum"
    TOPSIS = "topsis"


class RankingPresetEnum(str, Enum):
    BALANCED = "balanced"
    COST_OPTIMIZED = "cost_optimized"
    QUALITY_OPTIMIZED = "quality_optimized"
    LONG_CONTEXT = "long_context"


class ModelComparisonRequest(BaseModel):
    prompt: str = Field(
        ...,
        description="The prompt text to analyze and evaluate across models"
    )
    method: Optional[RankingMethodEnum] = Field(
        default=RankingMethodEnum.WEIGHTED_SUM,
        description="Ranking methodology: 'weighted_sum' or 'topsis'"
    )
    preset: Optional[RankingPresetEnum] = Field(
        default=RankingPresetEnum.BALANCED,
        description="Objective weighting preset: 'balanced', 'cost_optimized', 'quality_optimized', or 'long_context'"
    )
    custom_weights: Optional[Dict[str, float]] = Field(
        default=None,
        description="Optional custom weights for ranking criteria (suitability, cost, context)"
    )
    models: Optional[List[str]] = Field(
        default=None,
        description="Optional list of specific model IDs to compare (default: active registered models)"
    )
    expected_output_tokens: Optional[int] = Field(
        default=None,
        description="Optional manual override for expected response length in tokens"
    )
    include_deprecated: Optional[bool] = Field(
        default=False,
        description="Whether to include deprecated or retired models in comparison pool"
    )

    @field_validator("prompt")
    @classmethod
    def check_prompt(cls, v: str) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("Prompt cannot be empty or contain only whitespace.")
        return v


class CostBreakdownSchema(BaseModel):
    input_tokens: int
    estimated_output_tokens: int
    input_cost_usd: Optional[float] = None
    output_cost_usd: Optional[float] = None
    total_estimated_cost_usd: Optional[float] = None
    input_cost_per_m: Optional[float] = None
    output_cost_per_m: Optional[float] = None
    pricing_effective_date: str
    currency: str = "USD"
    pricing_status: str = "known"


class ModelComparisonItemSchema(BaseModel):
    model_id: str
    display_name: str
    provider: str
    context_window_tokens: int
    token_count: int
    token_count_method: str
    is_exact_token_count: bool
    estimated_output_tokens: int
    cost_breakdown: CostBreakdownSchema
    suitability_score: float
    capabilities: List[str]
    strengths: List[str]
    provenance: Dict[str, str]
    lifecycle_status: str = "active"
    source_url: Optional[str] = ""
    verification_date: Optional[str] = ""


class RankedModelSchema(BaseModel):
    rank: int
    model_id: str
    display_name: str
    provider: str
    score: float
    suitability_score: float
    estimated_cost_usd: Optional[float] = None
    token_count: int
    token_count_provenance: str
    latency_status: str
    benchmark_status: str
    reasoning: str
    lifecycle_status: str = "active"


class PromptFeaturesSchema(BaseModel):
    char_length: int
    word_count: int
    line_count: int
    code_proportion: float
    has_code_block: bool
    complexity_score: float
    technical_keyword_density: float
    security_sensitivity: float
    estimated_output_tokens: int
    detected_indicators: List[str]


class TaskClassificationSchema(BaseModel):
    predicted_category: str
    confidence: float
    category_probabilities: Dict[str, float]
    model_source: str
    is_low_confidence: Optional[bool] = False
    top_categories: Optional[List[str]] = None
    confidence_status: Optional[str] = "confident"


class ModelComparisonResponse(BaseModel):
    status: str = "success"
    prompt_text: str
    prompt_features: PromptFeaturesSchema
    task_classification: TaskClassificationSchema
    ranking_method: str
    ranking_preset: str
    applied_weights: Dict[str, float]
    ranked_models: List[RankedModelSchema]
    model_details: Dict[str, ModelComparisonItemSchema]
    provenance_declarations: Dict[str, str]


class ModelMetadataSchema(BaseModel):
    model_id: str
    display_name: str
    provider: str
    family: str
    context_window_tokens: int
    max_output_tokens: int
    input_cost_per_m: Optional[float] = None
    output_cost_per_m: Optional[float] = None
    pricing_effective_date: str
    currency: str
    tokenizer_name: str
    capabilities: List[str]
    strengths: List[str]
    description: str
    lifecycle_status: str = "active"
    source_url: Optional[str] = ""
    verification_date: Optional[str] = ""
    deprecation_date: Optional[str] = None