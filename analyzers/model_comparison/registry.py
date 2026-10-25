"""
PromptShield AI — Model Metadata & Pricing Registry.
===================================================
Provides a centralized, configurable registry of leading foundation models across:
- OpenAI (GPT-4o, GPT-4o-mini, GPT-4-turbo, GPT-3.5-turbo)
- Google (Gemini 1.5 Pro, Gemini 1.5 Flash, Gemini 2.0 Flash)
- Anthropic (Claude 3.5 Sonnet, Claude 3.5 Haiku, Claude 3 Opus)

Includes verified lifecycle status (active, deprecated, retired, preview, unverified),
context limits, dated public pricing metadata (USD per 1M tokens), official source URLs,
verification dates, tokenizer references, and capability profiles.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class LifecycleStatus(str, Enum):
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    RETIRED = "retired"
    PREVIEW = "preview"
    UNVERIFIED = "unverified"


@dataclass
class ModelPricing:
    """Pricing metadata in USD per million tokens."""
    input_cost_per_m: Optional[float]
    output_cost_per_m: Optional[float]
    pricing_effective_date: str = "2025-02-15"
    currency: str = "USD"
    is_known: bool = True
    source_url: str = ""
    notes: Optional[str] = None


@dataclass
class ModelMetadata:
    """Comprehensive specifications for a registered AI model."""
    model_id: str
    display_name: str
    provider: str  # "openai", "google", "anthropic"
    family: str
    context_window_tokens: int
    max_output_tokens: int
    pricing: ModelPricing
    tokenizer_name: str
    capabilities: List[str] = field(default_factory=list)
    strengths: List[str] = field(default_factory=list)
    description: str = ""
    lifecycle_status: LifecycleStatus = LifecycleStatus.ACTIVE
    source_url: str = ""
    verification_date: str = "2025-02-15"
    deprecation_date: Optional[str] = None


# Default Model Registry — Verified against official provider documentation (Feb 2025)
DEFAULT_MODELS: Dict[str, ModelMetadata] = {
    # -----------------------------------------------------------------------
    # OpenAI Models
    # Source: https://platform.openai.com/docs/models
    # Pricing: https://platform.openai.com/docs/pricing
    # -----------------------------------------------------------------------
    "gpt-4o": ModelMetadata(
        model_id="gpt-4o",
        display_name="OpenAI GPT-4o",
        provider="openai",
        family="gpt",
        context_window_tokens=128_000,
        max_output_tokens=16_384,
        pricing=ModelPricing(
            input_cost_per_m=2.50,
            output_cost_per_m=10.00,
            pricing_effective_date="2024-10-01",
            source_url="https://platform.openai.com/docs/pricing"
        ),
        tokenizer_name="o200k_base",
        capabilities=["code_generation", "security_analysis", "reasoning_and_math", "multimodal", "structured_output"],
        strengths=["High coding accuracy", "Fast multimodal processing", "Balanced cost-performance"],
        description="Flagship omni model from OpenAI with strong general intelligence and high token throughput.",
        lifecycle_status=LifecycleStatus.ACTIVE,
        source_url="https://platform.openai.com/docs/models#gpt-4o",
        verification_date="2025-02-15"
    ),
    "gpt-4o-mini": ModelMetadata(
        model_id="gpt-4o-mini",
        display_name="OpenAI GPT-4o-mini",
        provider="openai",
        family="gpt",
        context_window_tokens=128_000,
        max_output_tokens=16_384,
        pricing=ModelPricing(
            input_cost_per_m=0.15,
            output_cost_per_m=0.60,
            pricing_effective_date="2024-10-01",
            source_url="https://platform.openai.com/docs/pricing"
        ),
        tokenizer_name="o200k_base",
        capabilities=["general_instruction", "data_analysis", "code_generation", "fast_inference"],
        strengths=["Extremely low cost", "Sub-second response latency", "Cost-effective routine tasks"],
        description="Cost-optimized lightweight model from OpenAI designed for fast, affordable high-volume tasks.",
        lifecycle_status=LifecycleStatus.ACTIVE,
        source_url="https://platform.openai.com/docs/models#gpt-4o-mini",
        verification_date="2025-02-15"
    ),
    "gpt-4-turbo": ModelMetadata(
        model_id="gpt-4-turbo",
        display_name="OpenAI GPT-4 Turbo (Legacy)",
        provider="openai",
        family="gpt",
        context_window_tokens=128_000,
        max_output_tokens=4_096,
        pricing=ModelPricing(
            input_cost_per_m=10.00,
            output_cost_per_m=30.00,
            pricing_effective_date="2024-10-01",
            source_url="https://platform.openai.com/docs/pricing"
        ),
        tokenizer_name="cl100k_base",
        capabilities=["complex_reasoning", "code_generation", "security_analysis"],
        strengths=["Deep multi-step reasoning", "High architectural comprehension"],
        description="Previous-generation flagship from OpenAI; superseded by GPT-4o.",
        lifecycle_status=LifecycleStatus.DEPRECATED,
        source_url="https://platform.openai.com/docs/deprecations#gpt-4-turbo",
        verification_date="2025-02-15",
        deprecation_date="2024-10-01"
    ),
    "gpt-3.5-turbo": ModelMetadata(
        model_id="gpt-3.5-turbo",
        display_name="OpenAI GPT-3.5 Turbo (Retired)",
        provider="openai",
        family="gpt",
        context_window_tokens=16_385,
        max_output_tokens=4_096,
        pricing=ModelPricing(
            input_cost_per_m=0.50,
            output_cost_per_m=1.50,
            pricing_effective_date="2024-07-01",
            source_url="https://platform.openai.com/docs/deprecations"
        ),
        tokenizer_name="cl100k_base",
        capabilities=["general_instruction"],
        strengths=["Legacy compatibility"],
        description="Legacy 3.5 generation model; retired from active recommendation pool in favor of GPT-4o-mini.",
        lifecycle_status=LifecycleStatus.RETIRED,
        source_url="https://platform.openai.com/docs/deprecations",
        verification_date="2025-02-15",
        deprecation_date="2024-07-01"
    ),

    # -----------------------------------------------------------------------
    # Google Gemini Models
    # Source: https://ai.google.dev/gemini-api/docs/models
    # Deprecations: https://ai.google.dev/gemini-api/docs/deprecations
    # -----------------------------------------------------------------------
    "gemini-1.5-pro": ModelMetadata(
        model_id="gemini-1.5-pro",
        display_name="Google Gemini 1.5 Pro",
        provider="google",
        family="gemini",
        context_window_tokens=2_000_000,
        max_output_tokens=8_192,
        pricing=ModelPricing(
            input_cost_per_m=1.25,
            output_cost_per_m=5.00,
            pricing_effective_date="2024-10-01",
            source_url="https://ai.google.dev/gemini-api/docs/models/gemini#gemini-1.5-pro"
        ),
        tokenizer_name="gemini_256k_bpe_calibrated",
        capabilities=["long_context", "code_generation", "data_analysis", "reasoning_and_math"],
        strengths=["2M token context window", "Cross-repository code reasoning", "Massive document ingestion"],
        description="Google's advanced multimodal model featuring an industry-leading 2-million-token context window.",
        lifecycle_status=LifecycleStatus.ACTIVE,
        source_url="https://ai.google.dev/gemini-api/docs/models/gemini#gemini-1.5-pro",
        verification_date="2025-02-15"
    ),
    "gemini-1.5-flash": ModelMetadata(
        model_id="gemini-1.5-flash",
        display_name="Google Gemini 1.5 Flash",
        provider="google",
        family="gemini",
        context_window_tokens=1_000_000,
        max_output_tokens=8_192,
        pricing=ModelPricing(
            input_cost_per_m=0.075,
            output_cost_per_m=0.30,
            pricing_effective_date="2024-10-01",
            source_url="https://ai.google.dev/gemini-api/docs/models/gemini#gemini-1.5-flash"
        ),
        tokenizer_name="gemini_256k_bpe_calibrated",
        capabilities=["long_context", "high_speed", "general_instruction", "data_analysis"],
        strengths=["Lowest token pricing", "1M token context", "High throughput streaming"],
        description="Ultra-fast, lightweight model optimized for high-volume, long-context tasks with minimal latency.",
        lifecycle_status=LifecycleStatus.ACTIVE,
        source_url="https://ai.google.dev/gemini-api/docs/models/gemini#gemini-1.5-flash",
        verification_date="2025-02-15"
    ),
    "gemini-2.0-flash": ModelMetadata(
        model_id="gemini-2.0-flash",
        display_name="Google Gemini 2.0 Flash",
        provider="google",
        family="gemini",
        context_window_tokens=1_000_000,
        max_output_tokens=8_192,
        pricing=ModelPricing(
            input_cost_per_m=0.10,
            output_cost_per_m=0.40,
            pricing_effective_date="2024-12-01",
            source_url="https://ai.google.dev/gemini-api/docs/models/gemini#gemini-2.0-flash"
        ),
        tokenizer_name="gemini_256k_bpe_calibrated",
        capabilities=["agentic_coding", "high_speed", "multimodal", "code_generation"],
        strengths=["Next-gen agentic capabilities", "Real-time multimodal responsiveness"],
        description="Next-generation Gemini Flash model built for agentic execution and fast multi-step tasks.",
        lifecycle_status=LifecycleStatus.ACTIVE,
        source_url="https://ai.google.dev/gemini-api/docs/models/gemini#gemini-2.0-flash",
        verification_date="2025-02-15"
    ),

    # -----------------------------------------------------------------------
    # Anthropic Claude Models
    # Source: https://docs.anthropic.com/en/docs/about-claude/models
    # Pricing: https://www.anthropic.com/pricing
    # -----------------------------------------------------------------------
    "claude-3-5-sonnet": ModelMetadata(
        model_id="claude-3-5-sonnet",
        display_name="Anthropic Claude 3.5 Sonnet",
        provider="anthropic",
        family="claude",
        context_window_tokens=200_000,
        max_output_tokens=8_192,
        pricing=ModelPricing(
            input_cost_per_m=3.00,
            output_cost_per_m=15.00,
            pricing_effective_date="2024-10-01",
            source_url="https://www.anthropic.com/pricing"
        ),
        tokenizer_name="claude_calibrated_heuristic",
        capabilities=["code_generation", "security_analysis", "reasoning_and_math", "creative_and_writing"],
        strengths=["State-of-the-art coding benchmark performance", "Nuanced technical writing", "Precise tool use"],
        description="Anthropic's flagship model delivering software engineering and reasoning capabilities.",
        lifecycle_status=LifecycleStatus.ACTIVE,
        source_url="https://docs.anthropic.com/en/docs/about-claude/models#claude-3-5-sonnet",
        verification_date="2025-02-15"
    ),
    "claude-3-5-haiku": ModelMetadata(
        model_id="claude-3-5-haiku",
        display_name="Anthropic Claude 3.5 Haiku",
        provider="anthropic",
        family="claude",
        context_window_tokens=200_000,
        max_output_tokens=8_192,
        pricing=ModelPricing(
            input_cost_per_m=0.80,
            output_cost_per_m=4.00,
            pricing_effective_date="2024-11-01",
            source_url="https://www.anthropic.com/pricing"
        ),
        tokenizer_name="claude_calibrated_heuristic",
        capabilities=["high_speed", "code_generation", "general_instruction", "data_analysis"],
        strengths=["Fast code generation", "Affordable Anthropic intelligence", "Structured data parsing"],
        description="Anthropic's fastest, most responsive model providing high code performance at low cost.",
        lifecycle_status=LifecycleStatus.ACTIVE,
        source_url="https://docs.anthropic.com/en/docs/about-claude/models#claude-3-5-haiku",
        verification_date="2025-02-15"
    ),
    "claude-3-opus": ModelMetadata(
        model_id="claude-3-opus",
        display_name="Anthropic Claude 3 Opus",
        provider="anthropic",
        family="claude",
        context_window_tokens=200_000,
        max_output_tokens=4_096,
        pricing=ModelPricing(
            input_cost_per_m=15.00,
            output_cost_per_m=75.00,
            pricing_effective_date="2024-10-01",
            source_url="https://www.anthropic.com/pricing"
        ),
        tokenizer_name="claude_calibrated_heuristic",
        capabilities=["complex_reasoning", "creative_and_writing", "security_analysis", "code_generation"],
        strengths=["Deep contextual comprehension", "Sophisticated literary writing", "Complex mathematical proofs"],
        description="Anthropic's deeply analytical model designed for complex multi-faceted research and synthesis.",
        lifecycle_status=LifecycleStatus.ACTIVE,
        source_url="https://docs.anthropic.com/en/docs/about-claude/models#claude-3-opus",
        verification_date="2025-02-15"
    )
}


class ModelRegistry:
    """Manages model metadata and allows dynamic configuration and eligibility filtering."""

    def __init__(self):
        self._models: Dict[str, ModelMetadata] = dict(DEFAULT_MODELS)

    def get_model(self, model_id: str) -> Optional[ModelMetadata]:
        """Retrieves a specific model by ID regardless of lifecycle status."""
        return self._models.get(model_id)

    def list_models(
        self,
        provider: Optional[str] = None,
        include_deprecated: bool = False,
        lifecycle_status: Optional[LifecycleStatus] = None
    ) -> List[ModelMetadata]:
        """
        Lists models with optional provider and lifecycle eligibility filtering.
        By default (include_deprecated=False), excludes deprecated and retired models.
        """
        models = list(self._models.values())
        if provider:
            models = [m for m in models if m.provider.lower() == provider.lower()]

        if lifecycle_status is not None:
            models = [m for m in models if m.lifecycle_status == lifecycle_status]
        elif not include_deprecated:
            # Eligible default recommendation pool: only ACTIVE and PREVIEW models
            models = [
                m for m in models
                if m.lifecycle_status in {LifecycleStatus.ACTIVE, LifecycleStatus.PREVIEW}
            ]

        return models

    def is_model_eligible(self, model_id: str) -> bool:
        """Checks if a model is eligible for default recommendation pool."""
        m = self.get_model(model_id)
        if not m:
            return False
        return m.lifecycle_status in {LifecycleStatus.ACTIVE, LifecycleStatus.PREVIEW}

    def register_model(self, model: ModelMetadata) -> None:
        """Registers or overwrites a model in the active registry."""
        self._models[model.model_id] = model

    def update_pricing(
        self,
        model_id: str,
        input_cost: Optional[float],
        output_cost: Optional[float],
        date_str: str,
        is_known: bool = True
    ) -> bool:
        """Updates pricing metadata for a registered model."""
        if model_id in self._models:
            self._models[model_id].pricing = ModelPricing(
                input_cost_per_m=input_cost,
                output_cost_per_m=output_cost,
                pricing_effective_date=date_str,
                is_known=is_known,
                source_url=self._models[model_id].pricing.source_url
            )
            return True
        return False


# Singleton registry instance
registry = ModelRegistry()
