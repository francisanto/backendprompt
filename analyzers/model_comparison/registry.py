"""
PromptShield AI — Model Metadata & Pricing Registry.
===================================================
Provides a centralized, configurable registry of leading foundation models across:
- OpenAI (GPT-4o, GPT-4o-mini, GPT-4-turbo)
- Google (Gemini 1.5 Pro, Gemini 1.5 Flash, Gemini 2.0 Flash)
- Anthropic (Claude 3.5 Sonnet, Claude 3.5 Haiku, Claude 3 Opus)

Includes context limits, dated public pricing metadata (USD per 1M tokens),
tokenizer references, and capability profiles.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ModelPricing:
    """Pricing metadata in USD per million tokens."""
    input_cost_per_m: float
    output_cost_per_m: float
    pricing_effective_date: str = "2024-10-01"
    currency: str = "USD"


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


# Default Model Registry
DEFAULT_MODELS: Dict[str, ModelMetadata] = {
    # OpenAI Models
    "gpt-4o": ModelMetadata(
        model_id="gpt-4o",
        display_name="OpenAI GPT-4o",
        provider="openai",
        family="gpt",
        context_window_tokens=128_000,
        max_output_tokens=16_384,
        pricing=ModelPricing(input_cost_per_m=2.50, output_cost_per_m=10.00, pricing_effective_date="2024-10-01"),
        tokenizer_name="o200k_base",
        capabilities=["code_generation", "security_analysis", "reasoning_and_math", "multimodal", "structured_output"],
        strengths=["High coding accuracy", "Fast multimodal processing", "Balanced cost-performance"],
        description="Flagship omni model from OpenAI with strong general intelligence and high token throughput."
    ),
    "gpt-4o-mini": ModelMetadata(
        model_id="gpt-4o-mini",
        display_name="OpenAI GPT-4o-mini",
        provider="openai",
        family="gpt",
        context_window_tokens=128_000,
        max_output_tokens=16_384,
        pricing=ModelPricing(input_cost_per_m=0.15, output_cost_per_m=0.60, pricing_effective_date="2024-10-01"),
        tokenizer_name="o200k_base",
        capabilities=["general_instruction", "data_analysis", "code_generation", "fast_inference"],
        strengths=["Extremely low cost", "Sub-second response latency", "Cost-effective routine tasks"],
        description="Cost-optimized lightweight model from OpenAI designed for fast, affordable high-volume tasks."
    ),
    "gpt-4-turbo": ModelMetadata(
        model_id="gpt-4-turbo",
        display_name="OpenAI GPT-4 Turbo",
        provider="openai",
        family="gpt",
        context_window_tokens=128_000,
        max_output_tokens=4_096,
        pricing=ModelPricing(input_cost_per_m=10.00, output_cost_per_m=30.00, pricing_effective_date="2024-10-01"),
        tokenizer_name="cl100k_base",
        capabilities=["complex_reasoning", "code_generation", "security_analysis"],
        strengths=["Deep multi-step reasoning", "High architectural comprehension"],
        description="Previous-generation flagship from OpenAI with proven code generation and analytical capability."
    ),

    # Google Gemini Models
    "gemini-1.5-pro": ModelMetadata(
        model_id="gemini-1.5-pro",
        display_name="Google Gemini 1.5 Pro",
        provider="google",
        family="gemini",
        context_window_tokens=2_000_000,
        max_output_tokens=8_192,
        pricing=ModelPricing(input_cost_per_m=1.25, output_cost_per_m=5.00, pricing_effective_date="2024-10-01"),
        tokenizer_name="gemini_256k_bpe_calibrated",
        capabilities=["long_context", "code_generation", "data_analysis", "reasoning_and_math"],
        strengths=["2M token context window", "Cross-repository code reasoning", "Massive document ingestion"],
        description="Google's advanced multimodal model featuring an industry-leading 2-million-token context window."
    ),
    "gemini-1.5-flash": ModelMetadata(
        model_id="gemini-1.5-flash",
        display_name="Google Gemini 1.5 Flash",
        provider="google",
        family="gemini",
        context_window_tokens=1_000_000,
        max_output_tokens=8_192,
        pricing=ModelPricing(input_cost_per_m=0.075, output_cost_per_m=0.30, pricing_effective_date="2024-10-01"),
        tokenizer_name="gemini_256k_bpe_calibrated",
        capabilities=["long_context", "high_speed", "general_instruction", "data_analysis"],
        strengths=["Lowest token pricing", "1M token context", "High throughput streaming"],
        description="Ultra-fast, lightweight model optimized for high-volume, long-context tasks with minimal latency."
    ),
    "gemini-2.0-flash": ModelMetadata(
        model_id="gemini-2.0-flash",
        display_name="Google Gemini 2.0 Flash",
        provider="google",
        family="gemini",
        context_window_tokens=1_000_000,
        max_output_tokens=8_192,
        pricing=ModelPricing(input_cost_per_m=0.10, output_cost_per_m=0.40, pricing_effective_date="2024-12-01"),
        tokenizer_name="gemini_256k_bpe_calibrated",
        capabilities=["agentic_coding", "high_speed", "multimodal", "code_generation"],
        strengths=["Next-gen agentic capabilities", "Real-time multimodal responsiveness"],
        description="Next-generation Gemini Flash model built for agentic execution and fast multi-step tasks."
    ),

    # Anthropic Claude Models
    "claude-3-5-sonnet": ModelMetadata(
        model_id="claude-3-5-sonnet",
        display_name="Anthropic Claude 3.5 Sonnet",
        provider="anthropic",
        family="claude",
        context_window_tokens=200_000,
        max_output_tokens=8_192,
        pricing=ModelPricing(input_cost_per_m=3.00, output_cost_per_m=15.00, pricing_effective_date="2024-10-01"),
        tokenizer_name="claude_calibrated_heuristic",
        capabilities=["code_generation", "security_analysis", "reasoning_and_math", "creative_and_writing"],
        strengths=["State-of-the-art coding benchmark performance", "Nuanced technical writing", "Precise tool use"],
        description="Anthropic's flagship model delivering state-of-the-art software engineering and reasoning capabilities."
    ),
    "claude-3-5-haiku": ModelMetadata(
        model_id="claude-3-5-haiku",
        display_name="Anthropic Claude 3.5 Haiku",
        provider="anthropic",
        family="claude",
        context_window_tokens=200_000,
        max_output_tokens=8_192,
        pricing=ModelPricing(input_cost_per_m=0.80, output_cost_per_m=4.00, pricing_effective_date="2024-11-01"),
        tokenizer_name="claude_calibrated_heuristic",
        capabilities=["high_speed", "code_generation", "general_instruction", "data_analysis"],
        strengths=["Fast code generation", "Affordable Anthropic intelligence", "Structured data parsing"],
        description="Anthropic's fastest, most responsive model providing near-Sonnet code performance at a fraction of the cost."
    ),
    "claude-3-opus": ModelMetadata(
        model_id="claude-3-opus",
        display_name="Anthropic Claude 3 Opus",
        provider="anthropic",
        family="claude",
        context_window_tokens=200_000,
        max_output_tokens=4_096,
        pricing=ModelPricing(input_cost_per_m=15.00, output_cost_per_m=75.00, pricing_effective_date="2024-10-01"),
        tokenizer_name="claude_calibrated_heuristic",
        capabilities=["complex_reasoning", "creative_and_writing", "security_analysis"],
        strengths=["Deep contextual comprehension", "Sophisticated literary writing", "Complex mathematical proofs"],
        description="Anthropic's most deeply analytical model designed for complex multi-faceted research and synthesis."
    )
}


class ModelRegistry:
    """Manages model metadata and allows dynamic configuration."""

    def __init__(self):
        self._models: Dict[str, ModelMetadata] = dict(DEFAULT_MODELS)

    def get_model(self, model_id: str) -> Optional[ModelMetadata]:
        return self._models.get(model_id)

    def list_models(self, provider: Optional[str] = None) -> List[ModelMetadata]:
        models = list(self._models.values())
        if provider:
            models = [m for m in models if m.provider.lower() == provider.lower()]
        return models

    def register_model(self, model: ModelMetadata) -> None:
        self._models[model.model_id] = model

    def update_pricing(self, model_id: str, input_cost: float, output_cost: float, date_str: str) -> bool:
        if model_id in self._models:
            self._models[model_id].pricing = ModelPricing(
                input_cost_per_m=input_cost,
                output_cost_per_m=output_cost,
                pricing_effective_date=date_str
            )
            return True
        return False


# Singleton registry instance
registry = ModelRegistry()
