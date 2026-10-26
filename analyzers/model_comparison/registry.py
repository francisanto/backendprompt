"""
PromptShield AI — Model Metadata, Product Registry & Freshness Catalog.
========================================================================
Audit Date: 2026-10-25
Provides a centralized, configurable registry of leading AI systems across three
distinct product categories:

1. Category A — Foundation Models (OpenAI, Google, Anthropic, xAI)
2. Category B — IDEs & Coding Assistants (Cursor, Windsurf, GitHub Copilot, Google Antigravity)
3. Category C — AI App Builders (Vercel v0, Bolt.new, Lovable, Replit)

Enforces strict lifecycle and freshness verification:
- Default recommendation pools include ONLY verified, active, generally available systems.
- Retired, shut-down, deprecated, preview, and unverified systems are strictly excluded from default rankings.
- Stale or unknown pricing is flagged and never converted to free/zero-cost.
- Prevents cross-category ranking: foundation models, IDEs, and app builders are evaluated
  under separate category-specific criteria, scoring models, and explanations.
- Records exact source URLs, verification dates, audit timestamps, and field-level provenance.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union


class LifecycleStatus(str, Enum):
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    RETIRED = "retired"
    PREVIEW = "preview"
    UNVERIFIED = "unverified"


class ProductCategory(str, Enum):
    FOUNDATION_MODEL = "foundation_model"
    IDE_ASSISTANT = "ide_assistant"
    APP_BUILDER = "app_builder"


class ProductAvailability(str, Enum):
    GENERALLY_AVAILABLE = "generally_available"
    PUBLIC_BETA = "public_beta"
    WAITLIST = "waitlist"
    DISCONTINUED = "discontinued"
    UNVERIFIED = "unverified"


@dataclass
class ModelPricing:
    """Pricing metadata in USD per million tokens with freshness and provenance tracking."""
    input_cost_per_m: Optional[float]
    output_cost_per_m: Optional[float]
    pricing_effective_date: str = "2026-09-15"
    pricing_verification_date: str = "2026-10-25"
    currency: str = "USD"
    is_known: bool = True
    is_stale: bool = False
    source_url: str = ""
    notes: Optional[str] = None


@dataclass
class ModelMetadata:
    """Specifications for a registered foundation model with audit tracking."""
    model_id: str
    display_name: str
    provider: str  # "openai", "google", "anthropic", "xai"
    family: str
    context_window_tokens: int
    max_output_tokens: int
    pricing: ModelPricing
    tokenizer_name: str
    category: ProductCategory = ProductCategory.FOUNDATION_MODEL
    capabilities: List[str] = field(default_factory=list)
    strengths: List[str] = field(default_factory=list)
    description: str = ""
    lifecycle_status: LifecycleStatus = LifecycleStatus.ACTIVE
    source_url: str = ""
    verification_date: str = "2026-10-25"
    last_audit_date: str = "2026-10-25"
    deprecation_date: Optional[str] = None
    retirement_date: Optional[str] = None
    replacement_model_id: Optional[str] = None
    unverified_fields: List[str] = field(default_factory=list)


@dataclass
class IDEProductMetadata:
    """Specifications for a developer environment or coding assistant."""
    product_id: str
    display_name: str
    vendor: str
    category: ProductCategory = ProductCategory.IDE_ASSISTANT
    availability: ProductAvailability = ProductAvailability.GENERALLY_AVAILABLE
    supported_os: List[str] = field(default_factory=list)
    editor_integration: List[str] = field(default_factory=list)
    supported_models: List[str] = field(default_factory=list)
    key_capabilities: List[str] = field(default_factory=list)
    context_features: List[str] = field(default_factory=list)
    execution_capabilities: List[str] = field(default_factory=list)
    pricing_tier_summary: str = ""
    pricing_model: str = "freemium_subscription"
    free_tier: str = ""
    paid_tiers: List[str] = field(default_factory=list)
    privacy_statement: str = ""
    description: str = ""
    source_url: str = ""
    verification_date: str = "2026-09-10"
    last_audit_date: str = "2026-10-25"
    criteria_scores: Dict[str, float] = field(default_factory=dict)
    unverified_fields: List[str] = field(default_factory=list)


@dataclass
class AppBuilderMetadata:
    """Specifications for an AI application builder or prototyping platform."""
    product_id: str
    display_name: str
    vendor: str
    category: ProductCategory = ProductCategory.APP_BUILDER
    availability: ProductAvailability = ProductAvailability.GENERALLY_AVAILABLE
    primary_stacks: List[str] = field(default_factory=list)
    supported_backends: List[str] = field(default_factory=list)
    database_auth_support: List[str] = field(default_factory=list)
    deployment_options: List[str] = field(default_factory=list)
    github_integration: str = ""
    source_code_export: str = ""
    key_capabilities: List[str] = field(default_factory=list)
    supported_models: List[str] = field(default_factory=list)
    pricing_tier_summary: str = ""
    pricing_model: str = "freemium_credits"
    free_tier: str = ""
    paid_tiers: List[str] = field(default_factory=list)
    documented_limitations: List[str] = field(default_factory=list)
    description: str = ""
    source_url: str = ""
    verification_date: str = "2026-08-10"
    last_audit_date: str = "2026-10-25"
    criteria_scores: Dict[str, float] = field(default_factory=dict)
    unverified_fields: List[str] = field(default_factory=list)


# ===========================================================================
# Category A: Foundation Models Catalogue
# Audited: 2026-10-25 against official provider documentation
# ===========================================================================
DEFAULT_MODELS: Dict[str, ModelMetadata] = {
    # -----------------------------------------------------------------------
    # OpenAI Active GA Models (Current Generation)
    # Source: https://platform.openai.com/docs/models
    # Pricing: https://openai.com/api/pricing
    # -----------------------------------------------------------------------
    "gpt-4o": ModelMetadata(
        model_id="gpt-4o",
        display_name="OpenAI GPT-4o",
        provider="openai",
        family="gpt-4o",
        context_window_tokens=128_000,
        max_output_tokens=16_384,
        pricing=ModelPricing(
            input_cost_per_m=2.50,
            output_cost_per_m=10.00,
            pricing_effective_date="2024-10-01",
            pricing_verification_date="2026-10-26",
            source_url="https://openai.com/api/pricing"
        ),
        tokenizer_name="o200k_base",
        capabilities=["code_generation", "security_analysis", "multimodal", "structured_output", "general_instruction"],
        strengths=["High multimodal fidelity", "Broad language and code capabilities", "Fast token throughput"],
        description="OpenAI's flagship multimodal model for high-intelligence everyday coding, analysis, and generation.",
        lifecycle_status=LifecycleStatus.ACTIVE,
        source_url="https://platform.openai.com/docs/models#gpt-4o",
        verification_date="2026-10-26",
        last_audit_date="2026-10-26"
    ),
    "gpt-4o-mini": ModelMetadata(
        model_id="gpt-4o-mini",
        display_name="OpenAI GPT-4o-mini",
        provider="openai",
        family="gpt-4o",
        context_window_tokens=128_000,
        max_output_tokens=16_384,
        pricing=ModelPricing(
            input_cost_per_m=0.15,
            output_cost_per_m=0.60,
            pricing_effective_date="2024-10-01",
            pricing_verification_date="2026-10-26",
            source_url="https://openai.com/api/pricing"
        ),
        tokenizer_name="o200k_base",
        capabilities=["fast_inference", "code_generation", "general_instruction", "data_analysis"],
        strengths=["Ultra-low token pricing", "High speed", "Cost-effective routine processing"],
        description="Fast, highly cost-effective model designed for high-volume, low-latency processing.",
        lifecycle_status=LifecycleStatus.ACTIVE,
        source_url="https://platform.openai.com/docs/models#gpt-4o-mini",
        verification_date="2026-10-26",
        last_audit_date="2026-10-26"
    ),
    "o1": ModelMetadata(
        model_id="o1",
        display_name="OpenAI o1",
        provider="openai",
        family="o1",
        context_window_tokens=200_000,
        max_output_tokens=100_000,
        pricing=ModelPricing(
            input_cost_per_m=15.00,
            output_cost_per_m=60.00,
            pricing_effective_date="2024-12-05",
            pricing_verification_date="2026-10-26",
            source_url="https://openai.com/api/pricing"
        ),
        tokenizer_name="o200k_base",
        capabilities=["complex_reasoning", "stem_reasoning", "code_generation", "security_analysis"],
        strengths=["Deep internal chain-of-thought reasoning", "Competitive programming benchmark leader", "Advanced mathematics"],
        description="Reasoning model engineered to spend more time thinking before responding to complex problems.",
        lifecycle_status=LifecycleStatus.ACTIVE,
        source_url="https://platform.openai.com/docs/models#o1",
        verification_date="2026-10-26",
        last_audit_date="2026-10-26"
    ),
    "o1-mini": ModelMetadata(
        model_id="o1-mini",
        display_name="OpenAI o1-mini",
        provider="openai",
        family="o1",
        context_window_tokens=128_000,
        max_output_tokens=65_536,
        pricing=ModelPricing(
            input_cost_per_m=1.10,
            output_cost_per_m=4.40,
            pricing_effective_date="2024-09-12",
            pricing_verification_date="2026-10-26",
            source_url="https://openai.com/api/pricing"
        ),
        tokenizer_name="o200k_base",
        capabilities=["fast_reasoning", "code_generation", "stem_reasoning"],
        strengths=["Faster reasoning response", "Economical STEM performance", "Code generation optimization"],
        description="Faster, more cost-effective reasoning model optimized for coding, math, and scientific tasks.",
        lifecycle_status=LifecycleStatus.ACTIVE,
        source_url="https://platform.openai.com/docs/models#o1-mini",
        verification_date="2026-10-26",
        last_audit_date="2026-10-26"
    ),
    "o3-mini": ModelMetadata(
        model_id="o3-mini",
        display_name="OpenAI o3-mini",
        provider="openai",
        family="o3",
        context_window_tokens=200_000,
        max_output_tokens=100_000,
        pricing=ModelPricing(
            input_cost_per_m=1.10,
            output_cost_per_m=4.40,
            pricing_effective_date="2025-01-31",
            pricing_verification_date="2026-10-26",
            source_url="https://openai.com/api/pricing"
        ),
        tokenizer_name="o200k_base",
        capabilities=["complex_reasoning", "stem_reasoning", "code_generation", "structured_output"],
        strengths=["High-intelligence STEM and coding performance", "Configurable reasoning effort", "Low token cost"],
        description="OpenAI's latest cost-efficient reasoning model with adjustable reasoning effort and STEM focus.",
        lifecycle_status=LifecycleStatus.ACTIVE,
        source_url="https://platform.openai.com/docs/models#o3-mini",
        verification_date="2026-10-26",
        last_audit_date="2026-10-26"
    ),

    # -----------------------------------------------------------------------
    # Google Gemini Active GA Models (Current Generation)
    # Source: https://ai.google.dev/gemini-api/docs/models
    # Pricing: https://ai.google.dev/pricing
    # -----------------------------------------------------------------------
    "gemini-2.0-flash": ModelMetadata(
        model_id="gemini-2.0-flash",
        display_name="Google Gemini 2.0 Flash",
        provider="google",
        family="gemini-2",
        context_window_tokens=1_048_576,
        max_output_tokens=8_192,
        pricing=ModelPricing(
            input_cost_per_m=0.10,
            output_cost_per_m=0.40,
            pricing_effective_date="2025-02-05",
            pricing_verification_date="2026-10-26",
            source_url="https://ai.google.dev/pricing"
        ),
        tokenizer_name="gemini_256k_bpe_calibrated",
        capabilities=["high_speed", "price_performance", "long_context", "multimodal", "code_generation"],
        strengths=["Next-gen speed and efficiency", "1M token context window", "Native multimodal tool use"],
        description="Google's next-generation workhorse multimodal model offering high speed and 1M context.",
        lifecycle_status=LifecycleStatus.ACTIVE,
        source_url="https://ai.google.dev/gemini-api/docs/models#gemini-2.0-flash",
        verification_date="2026-10-26",
        last_audit_date="2026-10-26"
    ),
    "gemini-2.0-flash-lite": ModelMetadata(
        model_id="gemini-2.0-flash-lite",
        display_name="Google Gemini 2.0 Flash-Lite",
        provider="google",
        family="gemini-2",
        context_window_tokens=1_048_576,
        max_output_tokens=8_192,
        pricing=ModelPricing(
            input_cost_per_m=0.075,
            output_cost_per_m=0.30,
            pricing_effective_date="2025-02-05",
            pricing_verification_date="2026-10-26",
            source_url="https://ai.google.dev/pricing"
        ),
        tokenizer_name="gemini_256k_bpe_calibrated",
        capabilities=["fast_inference", "price_performance", "long_context", "multimodal"],
        strengths=["Lowest token pricing for Gemini 2.0 family", "High throughput", "1M token context capacity"],
        description="Cost-optimized edition of Gemini 2.0 built for high-throughput, latency-sensitive tasks.",
        lifecycle_status=LifecycleStatus.ACTIVE,
        source_url="https://ai.google.dev/gemini-api/docs/models#gemini-2.0-flash-lite",
        verification_date="2026-10-26",
        last_audit_date="2026-10-26"
    ),
    "gemini-1.5-pro": ModelMetadata(
        model_id="gemini-1.5-pro",
        display_name="Google Gemini 1.5 Pro",
        provider="google",
        family="gemini-1.5",
        context_window_tokens=2_097_152,
        max_output_tokens=8_192,
        pricing=ModelPricing(
            input_cost_per_m=1.25,
            output_cost_per_m=5.00,
            pricing_effective_date="2024-10-01",
            pricing_verification_date="2026-10-26",
            source_url="https://ai.google.dev/pricing"
        ),
        tokenizer_name="gemini_256k_bpe_calibrated",
        capabilities=["long_context", "complex_reasoning", "multimodal", "code_generation"],
        strengths=["Industry-leading 2M token context window", "Cross-document repository synthesis", "Multimodal audio/video reasoning"],
        description="Google's foundation model built for massive context retrieval and complex multimodal analysis.",
        lifecycle_status=LifecycleStatus.ACTIVE,
        source_url="https://ai.google.dev/gemini-api/docs/models#gemini-1.5-pro",
        verification_date="2026-10-26",
        last_audit_date="2026-10-26"
    ),
    "gemini-1.5-flash": ModelMetadata(
        model_id="gemini-1.5-flash",
        display_name="Google Gemini 1.5 Flash",
        provider="google",
        family="gemini-1.5",
        context_window_tokens=1_048_576,
        max_output_tokens=8_192,
        pricing=ModelPricing(
            input_cost_per_m=0.075,
            output_cost_per_m=0.30,
            pricing_effective_date="2024-10-01",
            pricing_verification_date="2026-10-26",
            source_url="https://ai.google.dev/pricing"
        ),
        tokenizer_name="gemini_256k_bpe_calibrated",
        capabilities=["high_speed", "long_context", "multimodal", "general_instruction"],
        strengths=["Affordable 1M context", "Fast response times", "Proven multimodal capability"],
        description="Lightweight, fast model designed for high-frequency multimodal and text processing.",
        lifecycle_status=LifecycleStatus.ACTIVE,
        source_url="https://ai.google.dev/gemini-api/docs/models#gemini-1.5-flash",
        verification_date="2026-10-26",
        last_audit_date="2026-10-26"
    ),

    # -----------------------------------------------------------------------
    # Anthropic Claude Active GA Models (Current Generation)
    # Source: https://docs.anthropic.com/en/docs/about-claude/models
    # Pricing: https://www.anthropic.com/pricing
    # -----------------------------------------------------------------------
    "claude-3-7-sonnet": ModelMetadata(
        model_id="claude-3-7-sonnet",
        display_name="Anthropic Claude 3.7 Sonnet",
        provider="anthropic",
        family="claude-3.7",
        context_window_tokens=200_000,
        max_output_tokens=64_000,
        pricing=ModelPricing(
            input_cost_per_m=3.00,
            output_cost_per_m=15.00,
            pricing_effective_date="2025-02-24",
            pricing_verification_date="2026-10-26",
            source_url="https://www.anthropic.com/pricing"
        ),
        tokenizer_name="claude_calibrated_heuristic",
        capabilities=["hybrid_reasoning", "code_generation", "security_analysis", "agentic_coding", "complex_reasoning"],
        strengths=["State-of-the-art hybrid reasoning", "Extended thinking mode with up to 64K output tokens", "Top-tier agentic software engineering"],
        description="Anthropic's premier hybrid reasoning model combining instant response with extended chain-of-thought.",
        lifecycle_status=LifecycleStatus.ACTIVE,
        source_url="https://docs.anthropic.com/en/docs/about-claude/models#claude-3-7-sonnet",
        verification_date="2026-10-26",
        last_audit_date="2026-10-26"
    ),
    "claude-3-5-sonnet": ModelMetadata(
        model_id="claude-3-5-sonnet",
        display_name="Anthropic Claude 3.5 Sonnet",
        provider="anthropic",
        family="claude-3.5",
        context_window_tokens=200_000,
        max_output_tokens=8_192,
        pricing=ModelPricing(
            input_cost_per_m=3.00,
            output_cost_per_m=15.00,
            pricing_effective_date="2024-10-22",
            pricing_verification_date="2026-10-26",
            source_url="https://www.anthropic.com/pricing"
        ),
        tokenizer_name="claude_calibrated_heuristic",
        capabilities=["code_generation", "security_analysis", "agentic_coding", "nuanced_writing", "structured_output"],
        strengths=["Industry benchmark leader in software engineering", "Precise tool use and instruction following", "Reliable multi-file coding"],
        description="Frontier coding and reasoning model delivering high precision for software development.",
        lifecycle_status=LifecycleStatus.ACTIVE,
        source_url="https://docs.anthropic.com/en/docs/about-claude/models#claude-3-5-sonnet",
        verification_date="2026-10-26",
        last_audit_date="2026-10-26"
    ),
    "claude-3-5-haiku": ModelMetadata(
        model_id="claude-3-5-haiku",
        display_name="Anthropic Claude 3.5 Haiku",
        provider="anthropic",
        family="claude-3.5",
        context_window_tokens=200_000,
        max_output_tokens=8_192,
        pricing=ModelPricing(
            input_cost_per_m=0.80,
            output_cost_per_m=4.00,
            pricing_effective_date="2024-11-04",
            pricing_verification_date="2026-10-26",
            source_url="https://www.anthropic.com/pricing"
        ),
        tokenizer_name="claude_calibrated_heuristic",
        capabilities=["high_speed", "code_generation", "general_instruction", "data_analysis"],
        strengths=["Sub-second latency", "Superior speed-to-intelligence balance", "Structured output extraction"],
        description="Anthropic's fast, cost-effective model providing high responsiveness and coding aptitude.",
        lifecycle_status=LifecycleStatus.ACTIVE,
        source_url="https://docs.anthropic.com/en/docs/about-claude/models#claude-3-5-haiku",
        verification_date="2026-10-26",
        last_audit_date="2026-10-26"
    ),
    "claude-3-opus": ModelMetadata(
        model_id="claude-3-opus",
        display_name="Anthropic Claude 3 Opus",
        provider="anthropic",
        family="claude-3",
        context_window_tokens=200_000,
        max_output_tokens=4_096,
        pricing=ModelPricing(
            input_cost_per_m=15.00,
            output_cost_per_m=75.00,
            pricing_effective_date="2024-03-04",
            pricing_verification_date="2026-10-26",
            source_url="https://www.anthropic.com/pricing"
        ),
        tokenizer_name="claude_calibrated_heuristic",
        capabilities=["complex_reasoning", "creative_and_writing", "security_analysis"],
        strengths=["Deep nuanced writing", "Extensive comprehension of complex domain material", "Mathematical reasoning"],
        description="Anthropic's powerful model for highly complex analysis, research, and nuanced creative writing.",
        lifecycle_status=LifecycleStatus.ACTIVE,
        source_url="https://docs.anthropic.com/en/docs/about-claude/models#claude-3-opus",
        verification_date="2026-10-26",
        last_audit_date="2026-10-26"
    ),

    # -----------------------------------------------------------------------
    # xAI Grok Active GA Models (Current Generation)
    # Source: https://docs.x.ai/
    # -----------------------------------------------------------------------
    "grok-2": ModelMetadata(
        model_id="grok-2",
        display_name="xAI Grok 2",
        provider="xai",
        family="grok-2",
        context_window_tokens=131_072,
        max_output_tokens=4_096,
        pricing=ModelPricing(
            input_cost_per_m=2.00,
            output_cost_per_m=10.00,
            pricing_effective_date="2024-12-12",
            pricing_verification_date="2026-10-26",
            source_url="https://docs.x.ai/"
        ),
        tokenizer_name="grok_calibrated_heuristic",
        capabilities=["general_instruction", "code_generation", "complex_reasoning"],
        strengths=["Frontier reasoning capabilities", "Fast generation speed", "Broad domain comprehension"],
        description="xAI's flagship text generation and reasoning model with 131K context window.",
        lifecycle_status=LifecycleStatus.ACTIVE,
        source_url="https://docs.x.ai/docs/models#grok-2",
        verification_date="2026-10-26",
        last_audit_date="2026-10-26"
    ),
    "grok-2-vision": ModelMetadata(
        model_id="grok-2-vision",
        display_name="xAI Grok 2 Vision",
        provider="xai",
        family="grok-2",
        context_window_tokens=32_768,
        max_output_tokens=4_096,
        pricing=ModelPricing(
            input_cost_per_m=2.00,
            output_cost_per_m=10.00,
            pricing_effective_date="2024-12-12",
            pricing_verification_date="2026-10-26",
            source_url="https://docs.x.ai/"
        ),
        tokenizer_name="grok_calibrated_heuristic",
        capabilities=["multimodal", "code_generation", "data_analysis"],
        strengths=["Multimodal understanding", "Visual document and code screenshot reasoning"],
        description="Multimodal model from xAI capable of processing images, diagrams, and text.",
        lifecycle_status=LifecycleStatus.ACTIVE,
        source_url="https://docs.x.ai/docs/models#grok-2-vision",
        verification_date="2026-10-26",
        last_audit_date="2026-10-26"
    ),

    # -----------------------------------------------------------------------
    # DeepSeek Active GA Models (Current Generation)
    # Source: https://api-docs.deepseek.com/quick_start/pricing
    # -----------------------------------------------------------------------
    "deepseek-chat": ModelMetadata(
        model_id="deepseek-chat",
        display_name="DeepSeek-V3",
        provider="deepseek",
        family="deepseek-v3",
        context_window_tokens=64_000,
        max_output_tokens=8_000,
        pricing=ModelPricing(
            input_cost_per_m=0.14,
            output_cost_per_m=0.28,
            pricing_effective_date="2024-12-26",
            pricing_verification_date="2026-10-26",
            source_url="https://api-docs.deepseek.com/quick_start/pricing"
        ),
        tokenizer_name="deepseek_calibrated_heuristic",
        capabilities=["code_generation", "general_instruction", "data_analysis", "structured_output"],
        strengths=["Extremely low token cost", "Top-tier code and language benchmarks", "Efficient MoE architecture"],
        description="DeepSeek's 671B Mixture-of-Experts foundation model delivering frontier performance at high economy.",
        lifecycle_status=LifecycleStatus.ACTIVE,
        source_url="https://api-docs.deepseek.com/quick_start/pricing",
        verification_date="2026-10-26",
        last_audit_date="2026-10-26"
    ),
    "deepseek-reasoner": ModelMetadata(
        model_id="deepseek-reasoner",
        display_name="DeepSeek-R1",
        provider="deepseek",
        family="deepseek-r1",
        context_window_tokens=64_000,
        max_output_tokens=8_000,
        pricing=ModelPricing(
            input_cost_per_m=0.55,
            output_cost_per_m=2.19,
            pricing_effective_date="2025-01-20",
            pricing_verification_date="2026-10-26",
            source_url="https://api-docs.deepseek.com/quick_start/pricing"
        ),
        tokenizer_name="deepseek_calibrated_heuristic",
        capabilities=["complex_reasoning", "stem_reasoning", "code_generation", "security_analysis"],
        strengths=["Open-weights reinforcement learning reasoning", "Competitive with OpenAI o1 on math and coding", "Affordable reasoning API"],
        description="DeepSeek's frontier reasoning model trained via large-scale reinforcement learning for complex STEM tasks.",
        lifecycle_status=LifecycleStatus.ACTIVE,
        source_url="https://api-docs.deepseek.com/quick_start/pricing",
        verification_date="2026-10-26",
        last_audit_date="2026-10-26"
    ),

    # -----------------------------------------------------------------------
    # Unverified / Preview Model Identifiers (Excluded from default catalogue)
    # Strictly flagged to prevent false claims of official GA verification
    # -----------------------------------------------------------------------
    "gemini-2.5-pro": ModelMetadata(
        model_id="gemini-2.5-pro",
        display_name="Google Gemini 2.5 Pro (Unverified)",
        provider="google",
        family="gemini-2.5",
        context_window_tokens=1_048_576,
        max_output_tokens=65_536,
        pricing=ModelPricing(
            input_cost_per_m=None,
            output_cost_per_m=None,
            is_known=False,
            notes="Unverified model identifier and pricing; excluded from default catalogue"
        ),
        tokenizer_name="gemini_256k_bpe_calibrated",
        capabilities=["deep_thinking"],
        strengths=["Unverified candidate model"],
        description="Unverified identifier: retained for catalogue audit tracking only. Excluded from default recommendations.",
        lifecycle_status=LifecycleStatus.UNVERIFIED,
        source_url="https://ai.google.dev/gemini-api/docs/models",
        verification_date="2026-10-26",
        last_audit_date="2026-10-26",
        unverified_fields=["model_id", "pricing", "context_window"]
    ),
    "gemini-2.5-flash": ModelMetadata(
        model_id="gemini-2.5-flash",
        display_name="Google Gemini 2.5 Flash (Unverified)",
        provider="google",
        family="gemini-2.5",
        context_window_tokens=1_048_576,
        max_output_tokens=65_536,
        pricing=ModelPricing(
            input_cost_per_m=None,
            output_cost_per_m=None,
            is_known=False,
            notes="Unverified model identifier and pricing; excluded from default catalogue"
        ),
        tokenizer_name="gemini_256k_bpe_calibrated",
        capabilities=["high_speed"],
        strengths=["Unverified candidate model"],
        description="Unverified identifier: retained for catalogue audit tracking only. Excluded from default recommendations.",
        lifecycle_status=LifecycleStatus.UNVERIFIED,
        source_url="https://ai.google.dev/gemini-api/docs/models",
        verification_date="2026-10-26",
        last_audit_date="2026-10-26",
        unverified_fields=["model_id", "pricing", "context_window"]
    ),
    "grok-3": ModelMetadata(
        model_id="grok-3",
        display_name="xAI Grok 3 (Preview/Unverified API)",
        provider="xai",
        family="grok-3",
        context_window_tokens=131_072,
        max_output_tokens=8_192,
        pricing=ModelPricing(
            input_cost_per_m=None,
            output_cost_per_m=None,
            is_known=False,
            notes="Preview tier without verified public API token pricing; excluded from default catalogue"
        ),
        tokenizer_name="grok_calibrated_heuristic",
        capabilities=["complex_reasoning"],
        strengths=["Next-generation frontier preview"],
        description="Preview tier / unverified public API identifier. Excluded from default recommendations until GA verified.",
        lifecycle_status=LifecycleStatus.PREVIEW,
        source_url="https://docs.x.ai/",
        verification_date="2026-10-26",
        last_audit_date="2026-10-26",
        unverified_fields=["api_endpoint", "public_pricing"]
    ),
    "grok-3-mini": ModelMetadata(
        model_id="grok-3-mini",
        display_name="xAI Grok 3 Mini (Preview/Unverified API)",
        provider="xai",
        family="grok-3",
        context_window_tokens=131_072,
        max_output_tokens=8_192,
        pricing=ModelPricing(
            input_cost_per_m=None,
            output_cost_per_m=None,
            is_known=False,
            notes="Preview tier without verified public API token pricing; excluded from default catalogue"
        ),
        tokenizer_name="grok_calibrated_heuristic",
        capabilities=["fast_reasoning"],
        strengths=["Next-generation lightweight preview"],
        description="Preview tier / unverified public API identifier. Excluded from default recommendations until GA verified.",
        lifecycle_status=LifecycleStatus.PREVIEW,
        source_url="https://docs.x.ai/",
        verification_date="2026-10-26",
        last_audit_date="2026-10-26",
        unverified_fields=["api_endpoint", "public_pricing"]
    ),

    # -----------------------------------------------------------------------
    # Historical Retired / Deprecated Models (Excluded from default recommendations)
    # Maintained for historical auditing and backward comparison
    # -----------------------------------------------------------------------
    "gpt-4-turbo": ModelMetadata(
        model_id="gpt-4-turbo",
        display_name="OpenAI GPT-4 Turbo (Deprecated)",
        provider="openai",
        family="gpt-4",
        context_window_tokens=128_000,
        max_output_tokens=4_096,
        pricing=ModelPricing(
            input_cost_per_m=10.00,
            output_cost_per_m=30.00,
            pricing_effective_date="2024-04-09",
            pricing_verification_date="2024-10-01",
            is_stale=True,
            source_url="https://platform.openai.com/docs/deprecations#gpt-4-turbo"
        ),
        tokenizer_name="cl100k_base",
        capabilities=["complex_reasoning", "code_generation"],
        strengths=["Historical architectural reasoning"],
        description="Deprecated by OpenAI. Superseded by GPT-4o.",
        lifecycle_status=LifecycleStatus.DEPRECATED,
        source_url="https://platform.openai.com/docs/deprecations#gpt-4-turbo",
        verification_date="2024-10-01",
        last_audit_date="2026-10-26",
        deprecation_date="2024-10-01",
        replacement_model_id="gpt-4o"
    ),
    "gpt-3.5-turbo": ModelMetadata(
        model_id="gpt-3.5-turbo",
        display_name="OpenAI GPT-3.5 Turbo (Retired)",
        provider="openai",
        family="gpt-3.5",
        context_window_tokens=16_385,
        max_output_tokens=4_096,
        pricing=ModelPricing(
            input_cost_per_m=0.50,
            output_cost_per_m=1.50,
            pricing_effective_date="2024-07-01",
            pricing_verification_date="2024-07-01",
            is_stale=True,
            source_url="https://platform.openai.com/docs/deprecations"
        ),
        tokenizer_name="cl100k_base",
        capabilities=["general_instruction"],
        strengths=["Legacy compatibility"],
        description="Retired legacy model. Superseded by GPT-4o-mini.",
        lifecycle_status=LifecycleStatus.RETIRED,
        source_url="https://platform.openai.com/docs/deprecations",
        verification_date="2024-07-01",
        last_audit_date="2026-10-26",
        retirement_date="2024-07-01",
        replacement_model_id="gpt-4o-mini"
    ),
    "gemini-1.0-pro": ModelMetadata(
        model_id="gemini-1.0-pro",
        display_name="Google Gemini 1.0 Pro (Retired)",
        provider="google",
        family="gemini-1.0",
        context_window_tokens=32_768,
        max_output_tokens=8_192,
        pricing=ModelPricing(
            input_cost_per_m=0.50,
            output_cost_per_m=1.50,
            pricing_effective_date="2024-02-15",
            pricing_verification_date="2025-02-15",
            is_stale=True,
            source_url="https://ai.google.dev/gemini-api/docs/deprecations"
        ),
        tokenizer_name="gemini_256k_bpe_calibrated",
        capabilities=["general_instruction"],
        strengths=["Legacy Google multimodal model"],
        description="Shut down and retired. Superseded by Gemini 2.0 Flash.",
        lifecycle_status=LifecycleStatus.RETIRED,
        source_url="https://ai.google.dev/gemini-api/docs/deprecations",
        verification_date="2025-02-15",
        last_audit_date="2026-10-26",
        retirement_date="2025-02-15",
        replacement_model_id="gemini-2.0-flash"
    )
}


# ===========================================================================
# Category B: IDEs & Coding Assistants Catalogue
# Audited: 2026-10-26 against official product documentation
# ===========================================================================
DEFAULT_IDES: Dict[str, IDEProductMetadata] = {
    "cursor": IDEProductMetadata(
        product_id="cursor",
        display_name="Cursor",
        vendor="Anysphere",
        category=ProductCategory.IDE_ASSISTANT,
        availability=ProductAvailability.GENERALLY_AVAILABLE,
        supported_os=["macOS", "Windows", "Linux"],
        editor_integration=["VS Code Fork (Native Editor)"],
        supported_models=["Claude 3.7 Sonnet", "Claude 3.5 Sonnet", "GPT-4o", "Cursor Fast Models", "Custom API Key"],
        key_capabilities=[
            "Composer multi-file agent",
            "Local codebase vector indexing",
            "Inline code edits",
            "Cloud Agents and Bugbot code review",
            "Terminal command generation and execution",
            "Origin cloud code hosting"
        ],
        context_features=[
            "Manual developer context curation (@files, @folders, @docs, @git)",
            "Local embedding indexing of entire repository",
            "Automatic relevance reranking"
        ],
        execution_capabilities=[
            "Direct terminal command generation and execution",
            "Interactive multi-file diff application",
            "Autonomous background agent execution"
        ],
        pricing_tier_summary="Free ($0 evaluation); Plus ($8/mo); Pro ($20/mo); Business ($40/user/mo)",
        pricing_model="freemium_subscription",
        free_tier="Free ($0, limited Tab completions and starter agent requests)",
        paid_tiers=["Plus ($8/mo)", "Pro ($20/mo)", "Business ($40/user/mo)"],
        privacy_statement="Privacy Mode by default on Teams/Enterprise; customer code not retained or used for model training. SOC 2 certified.",
        description="AI-first fork of VS Code featuring Composer for multi-file agentic code generation and local codebase indexing.",
        source_url="https://cursor.com/pricing",
        verification_date="2026-10-26",
        last_audit_date="2026-10-26",
        criteria_scores={
            "multi_file_agent": 0.95,
            "repo_indexing": 0.94,
            "terminal_execution": 0.88,
            "model_flexibility": 0.95,
            "ecosystem_integration": 0.85,
            "privacy_controls": 0.90,
            "plan_affordability": 0.88
        },
        unverified_fields=[]
    ),
    "windsurf": IDEProductMetadata(
        product_id="windsurf",
        display_name="Windsurf",
        vendor="Codeium",
        category=ProductCategory.IDE_ASSISTANT,
        availability=ProductAvailability.GENERALLY_AVAILABLE,
        supported_os=["macOS", "Windows", "Linux"],
        editor_integration=["AI-first Editor (VS Code Compatible)"],
        supported_models=["SWE-1 Cascade proprietary agent", "Claude 3.7 Sonnet", "Claude 3.5 Sonnet", "GPT-4o"],
        key_capabilities=[
            "Cascade autonomous flow agent",
            "Fast Context automatic routing",
            "Multi-file collaborative editing",
            "Integrated terminal execution and auto-debugging",
            "Codemaps deep architectural navigation",
            "Inline completions (Supercomplete)"
        ],
        context_features=[
            "Proactive context routing without manual @ tagging",
            "Cursor position and terminal error context tracking",
            "Deep semantic repository mapping (Codemaps)"
        ],
        execution_capabilities=[
            "Autonomous multi-step terminal execution",
            "Automatic compile error resolution",
            "Live web preview environments"
        ],
        pricing_tier_summary="Free ($0 unlimited completions + starter agent quota); Pro ($20/mo); Teams ($40/user/mo)",
        pricing_model="freemium_subscription",
        free_tier="Free ($0, unlimited tab completions + limited Cascade quota)",
        paid_tiers=["Pro ($20/month)", "Teams ($40/user/month)", "Enterprise (Custom)"],
        privacy_statement="Zero data retention option available for enterprise; paid accounts do not train models on code.",
        description="Agent-first IDE by Codeium built around continuous flow collaboration and proactive context routing.",
        source_url="https://windsurf.com/pricing",
        verification_date="2026-10-26",
        last_audit_date="2026-10-26",
        criteria_scores={
            "multi_file_agent": 0.94,
            "repo_indexing": 0.92,
            "terminal_execution": 0.93,
            "model_flexibility": 0.88,
            "ecosystem_integration": 0.84,
            "privacy_controls": 0.88,
            "plan_affordability": 0.92
        },
        unverified_fields=[]
    ),
    "github-copilot": IDEProductMetadata(
        product_id="github-copilot",
        display_name="GitHub Copilot",
        vendor="GitHub / Microsoft",
        category=ProductCategory.IDE_ASSISTANT,
        availability=ProductAvailability.GENERALLY_AVAILABLE,
        supported_os=["macOS", "Windows", "Linux"],
        editor_integration=["VS Code", "Visual Studio", "JetBrains IDEs", "Xcode", "Neovim", "Copilot Workspace"],
        supported_models=["GPT-4o", "Claude 3.7 Sonnet", "Claude 3.5 Sonnet", "Gemini 2.0 Flash"],
        key_capabilities=[
            "Copilot Chat side-panel & inline",
            "Agent Mode multi-file editing",
            "Automated PR code reviews on GitHub",
            "Copilot CLI terminal assistance",
            "Next-edit suggestions",
            "Enterprise policy controls and audit logs"
        ],
        context_features=[
            "Implicit workspace scanning across open tabs and project trees",
            "Copilot Spaces repository grounding",
            "GitHub remote repository indexing"
        ],
        execution_capabilities=[
            "Terminal CLI command synthesis",
            "Workspace Agent multi-file planning",
            "PR generation and automated review comments"
        ],
        pricing_tier_summary="Free ($0, 2k completions/mo); Pro ($10/mo); Business ($19/user/mo); Enterprise ($39/user/mo)",
        pricing_model="freemium_subscription",
        free_tier="Free ($0, 2,000 code completions/month, limited chat)",
        paid_tiers=["Pro ($10/month)", "Business ($19/user/month)", "Enterprise ($39/user/month)"],
        privacy_statement="Enterprise zero data retention; code snippets and prompts are not retained or used for training.",
        description="Comprehensive developer assistant by GitHub and Microsoft integrated natively across major IDEs and GitHub.",
        source_url="https://github.com/features/copilot",
        verification_date="2026-10-26",
        last_audit_date="2026-10-26",
        criteria_scores={
            "multi_file_agent": 0.87,
            "repo_indexing": 0.90,
            "terminal_execution": 0.84,
            "model_flexibility": 0.92,
            "ecosystem_integration": 0.98,
            "privacy_controls": 0.96,
            "plan_affordability": 0.95
        },
        unverified_fields=[]
    ),
    "google-antigravity": IDEProductMetadata(
        product_id="google-antigravity",
        display_name="Google Antigravity",
        vendor="Google",
        category=ProductCategory.IDE_ASSISTANT,
        availability=ProductAvailability.GENERALLY_AVAILABLE,
        supported_os=["macOS", "Windows", "Linux"],
        editor_integration=["Standalone Desktop Command Center", "Antigravity IDE", "Extensions (VS Code, JetBrains, Zed)"],
        supported_models=["Google Gemini 2.0 Flash", "Gemini 1.5 Pro"],
        key_capabilities=[
            "Autonomous multi-surface software engineering agents",
            "Asynchronous agent orchestration manager",
            "Verifiable artifacts, task lists, and visual diff reviews",
            "Terminal command and browser subagent execution",
            "Human-in-the-loop review and approval controls",
            "Antigravity CLI and Python SDK programmatic control"
        ],
        context_features=[
            "Multi-surface workspace indexing and tracking",
            "Subagent conversation transcript logging",
            "Repository-level knowledge item (KI) integration",
            "Browser DOM and terminal session state grounding"
        ],
        execution_capabilities=[
            "Autonomous shell execution with interactive approval",
            "Browser automation and live web verification",
            "Persistent background task management",
            "One-shot and recurring timer scheduling"
        ],
        pricing_tier_summary="Included in Google Developer / Cloud AI tiers; developer access",
        pricing_model="cloud_developer_tier",
        free_tier="Developer tier access with Google Cloud credentials",
        paid_tiers=["Google Cloud Vertex AI / Enterprise developer billing"],
        privacy_statement="Governed by Google Cloud Enterprise AI privacy terms; customer prompt code not used for model training.",
        description="Google's agent-first development platform featuring autonomous multi-surface agents and verifiable artifacts.",
        source_url="https://antigravity.google/",
        verification_date="2026-10-26",
        last_audit_date="2026-10-26",
        criteria_scores={
            "multi_file_agent": 0.98,
            "repo_indexing": 0.95,
            "terminal_execution": 0.97,
            "model_flexibility": 0.85,
            "ecosystem_integration": 0.92,
            "privacy_controls": 0.94,
            "plan_affordability": 0.85
        },
        unverified_fields=[]
    )
}


# ===========================================================================
# Category C: AI App Builders Catalogue
# Audited: 2026-10-25 against official product documentation
# ===========================================================================
DEFAULT_APP_BUILDERS: Dict[str, AppBuilderMetadata] = {
    "v0-dev": AppBuilderMetadata(
        product_id="v0-dev",
        display_name="Vercel v0",
        vendor="Vercel",
        category=ProductCategory.APP_BUILDER,
        availability=ProductAvailability.GENERALLY_AVAILABLE,
        primary_stacks=["React", "Next.js (App Router)", "Tailwind CSS", "shadcn/ui", "TypeScript"],
        supported_backends=["Next.js Server Actions", "Vercel Serverless Functions", "External REST/GraphQL APIs"],
        database_auth_support=["Bring-your-own database", "External Supabase/Neon integration"],
        deployment_options=["1-click native Vercel deployment", "Shareable preview URLs"],
        github_integration="Direct code export to GitHub repositories, copy-paste code snippets, CLI component add",
        source_code_export="100% clean Next.js React component code export with zero proprietary lock-in",
        key_capabilities=[
            "Prompt-to-component UI generation",
            "Visual chat iteration and element inspector",
            "Figma-to-code design fidelity",
            "Production-ready shadcn/ui components",
            "Interactive live canvas preview"
        ],
        supported_models=["Vercel Frontier AI Routing (Claude, GPT, Custom UI Models)"],
        pricing_tier_summary="Free ($0, monthly generation credits); Premium ($20/mo, expanded credits and team features)",
        pricing_model="freemium_credits",
        free_tier="Free ($0, limited monthly generation credits and public sharing)",
        paid_tiers=["Premium ($20/month)", "Enterprise (Custom)"],
        documented_limitations=[
            "Frontend-centric: does not provision managed relational databases or auth servers out-of-the-box",
            "Requires developer integration for complex multi-tier backend architectures"
        ],
        description="Vercel's generative UI tool outputting production-grade React, Next.js, and shadcn/ui component code.",
        source_url="https://v0.dev/",
        verification_date="2026-08-10",
        last_audit_date="2026-10-26",
        criteria_scores={
            "full_stack_backend": 0.70,
            "frontend_ui_quality": 0.98,
            "deployment_hosting": 0.95,
            "code_export_github": 0.95,
            "autonomous_iteration": 0.86,
            "pricing_accessibility": 0.90
        },
        unverified_fields=[]
    ),
    "bolt-new": AppBuilderMetadata(
        product_id="bolt-new",
        display_name="Bolt.new",
        vendor="StackBlitz",
        category=ProductCategory.APP_BUILDER,
        availability=ProductAvailability.GENERALLY_AVAILABLE,
        primary_stacks=["Node.js", "React", "Vite", "Next.js", "Express", "Vue", "Svelte"],
        supported_backends=["In-browser Node.js runtime (WebContainers)", "Express/Hono REST APIs"],
        database_auth_support=["SQLite local databases", "Local storage", "External API integrations"],
        deployment_options=["Instant .bolt.host hosting", "One-click Netlify deployment"],
        github_integration="Push to GitHub repository, fork projects, full zip download",
        source_code_export="Full zip project download and direct GitHub repo synchronization",
        key_capabilities=[
            "In-browser full-stack application execution via WebContainers",
            "Interactive Plan & Build agent modes",
            "Full container shell terminal and npm package installation",
            "Real-time live browser preview with hot-module reload",
            "Autonomous error fixing and package dependency resolution"
        ],
        supported_models=["Anthropic Claude, OpenAI GPT, DeepSeek"],
        pricing_tier_summary="Free ($0, daily token allowances ~300K/day); Pro ($20/mo, expanded runtime & tokens)",
        pricing_model="freemium_credits",
        free_tier="Free ($0, daily token quota ~300K tokens/day, public projects)",
        paid_tiers=["Pro ($20/month)", "Teams ($40/user/month)"],
        documented_limitations=[
            "WebContainer sandbox constraints: cannot run Docker daemon, native C++ extensions, or heavy background daemons",
            "Ephemeral browser-based container storage requires external database for production persistence"
        ],
        description="In-browser AI app builder powered by StackBlitz WebContainers for instant full-stack web prototypes.",
        source_url="https://bolt.new/",
        verification_date="2026-08-10",
        last_audit_date="2026-10-26",
        criteria_scores={
            "full_stack_backend": 0.84,
            "frontend_ui_quality": 0.89,
            "deployment_hosting": 0.92,
            "code_export_github": 0.92,
            "autonomous_iteration": 0.92,
            "pricing_accessibility": 0.92
        },
        unverified_fields=[]
    ),
    "lovable-dev": AppBuilderMetadata(
        product_id="lovable-dev",
        display_name="Lovable",
        vendor="Lovable",
        category=ProductCategory.APP_BUILDER,
        availability=ProductAvailability.GENERALLY_AVAILABLE,
        primary_stacks=["React", "Vite", "TypeScript", "Tailwind CSS", "shadcn/ui", "Supabase"],
        supported_backends=["Supabase Edge Functions", "PostgreSQL database triggers", "REST APIs"],
        database_auth_support=["Native Supabase PostgreSQL", "Supabase Auth (OAuth, Email)", "Row Level Security (RLS)", "Storage buckets"],
        deployment_options=["Lovable Cloud hosting", "Custom domain mapping", "1-click web deployment"],
        github_integration="Two-way GitHub repository synchronization with automatic pull requests and commit history",
        source_code_export="Full clean TypeScript React project export with standard Vite configuration",
        key_capabilities=[
            "Full-stack MVP generation with integrated PostgreSQL and Auth",
            "Visual canvas element selection and prompt-driven edits",
            "Automatic Supabase schema migrations and RLS generation",
            "Two-way GitHub synchronization enabling team PR workflows",
            "Autonomous error detection and repair during build"
        ],
        supported_models=["Lovable Custom Agent with Claude backend"],
        pricing_tier_summary="Free ($0, 5 daily credits); Starter ($20/mo, ~100 edits); Pro ($50/mo, high usage)",
        pricing_model="freemium_credits",
        free_tier="Free ($0, 5 daily generation credits)",
        paid_tiers=["Starter ($20/month)", "Pro ($50/month)", "Enterprise (Custom)"],
        documented_limitations=[
            "Optimized tightly around React/Vite/Supabase architecture; alternative backends require decoupling",
            "Credit depletion on deep complex multi-turn architectural iterations"
        ],
        description="Full-stack AI web app builder tightly integrated with Supabase for instant database, authentication, and UI generation.",
        source_url="https://lovable.dev/",
        verification_date="2026-08-10",
        last_audit_date="2026-10-26",
        criteria_scores={
            "full_stack_backend": 0.96,
            "frontend_ui_quality": 0.93,
            "deployment_hosting": 0.94,
            "code_export_github": 0.96,
            "autonomous_iteration": 0.93,
            "pricing_accessibility": 0.86
        },
        unverified_fields=[]
    ),
    "replit-agent": AppBuilderMetadata(
        product_id="replit-agent",
        display_name="Replit",
        vendor="Replit",
        category=ProductCategory.APP_BUILDER,
        availability=ProductAvailability.GENERALLY_AVAILABLE,
        primary_stacks=["Python (FastAPI, Flask, Django)", "Node.js (Express, Next.js)", "Go", "Rust", "React", "PostgreSQL"],
        supported_backends=["Full native Linux backend execution", "FastAPI / Django / Express servers", "Multi-process architectures"],
        database_auth_support=["Native Replit PostgreSQL database", "Replit Auth", "Secret management", "Persistent cloud storage"],
        deployment_options=["1-click .replit.app hosting", "Autoscale cloud deployments", "Custom domains", "Background workers"],
        github_integration="Full git version control, import/export to GitHub, GitLab, and Bitbucket",
        source_code_export="100% full project file download, standard language runtimes without proprietary frameworks",
        key_capabilities=[
            "Autonomous Replit Agent planning, scaffolding, installing dependencies, and running apps",
            "Full cloud Linux environment with persistent terminal and package managers",
            "Multi-language backend support beyond JavaScript/Node.js",
            "Integrated PostgreSQL database with automatic connection strings",
            "Real-time multiplayer collaborative coding"
        ],
        supported_models=["Replit Agent (Claude, GPT, Replit Code Models)"],
        pricing_tier_summary="Free ($0, basic cloud workspace); Core ($17–$20/mo); Teams ($95+/user/mo)",
        pricing_model="freemium_subscription",
        free_tier="Free ($0, public cloud workspaces with limited compute)",
        paid_tiers=["Core ($17–$20/month)", "Teams ($95+/user/month)"],
        documented_limitations=[
            "Cloud compute checkpoints and agent credits can consume quickly on extensive multi-file projects",
            "Cold start latency on sleeping free-tier deployments"
        ],
        description="Autonomous cloud software development platform supporting multi-language full-stack apps with persistent Linux execution.",
        source_url="https://replit.com/",
        verification_date="2026-08-10",
        last_audit_date="2026-10-26",
        criteria_scores={
            "full_stack_backend": 0.98,
            "frontend_ui_quality": 0.85,
            "deployment_hosting": 0.96,
            "code_export_github": 0.94,
            "autonomous_iteration": 0.94,
            "pricing_accessibility": 0.88
        },
        unverified_fields=[]
    )
}


class ModelRegistry:
    """
    Manages metadata across Foundation Models, IDE Assistants, and AI App Builders.
    Maintains strict category separation and lifecycle eligibility.
    """

    def __init__(self):
        self._models: Dict[str, ModelMetadata] = dict(DEFAULT_MODELS)
        self._ides: Dict[str, IDEProductMetadata] = dict(DEFAULT_IDES)
        self._app_builders: Dict[str, AppBuilderMetadata] = dict(DEFAULT_APP_BUILDERS)

    def get_model(self, item_id: str) -> Optional[Union[ModelMetadata, IDEProductMetadata, AppBuilderMetadata]]:
        """Retrieves an item by ID across all categories."""
        if item_id in self._models:
            return self._models[item_id]
        if item_id in self._ides:
            return self._ides[item_id]
        if item_id in self._app_builders:
            return self._app_builders[item_id]
        return None

    def get_foundation_model(self, model_id: str) -> Optional[ModelMetadata]:
        """Retrieves a foundation model by ID regardless of lifecycle status."""
        return self._models.get(model_id)

    def get_ide(self, product_id: str) -> Optional[IDEProductMetadata]:
        """Retrieves an IDE assistant by ID."""
        return self._ides.get(product_id)

    def get_app_builder(self, product_id: str) -> Optional[AppBuilderMetadata]:
        """Retrieves an AI app builder by ID."""
        return self._app_builders.get(product_id)

    def list_models(
        self,
        category: Optional[ProductCategory] = None,
        provider: Optional[str] = None,
        include_deprecated: bool = False,
        lifecycle_status: Optional[LifecycleStatus] = None
    ) -> List[ModelMetadata]:
        """
        Lists foundation models with lifecycle eligibility filtering.
        By default (include_deprecated=False), returns ONLY verified ACTIVE models.
        Retired, deprecated, preview, and unverified models are strictly excluded.
        """
        models = list(self._models.values())
        if provider:
            models = [m for m in models if m.provider.lower() == provider.lower()]

        if lifecycle_status is not None:
            models = [m for m in models if m.lifecycle_status == lifecycle_status]
        elif not include_deprecated:
            # Default recommendation pool: STRICTLY eligible ACTIVE models only
            models = [m for m in models if self.is_model_eligible(m.model_id, ProductCategory.FOUNDATION_MODEL)]

        return models

    def list_ides(self, include_discontinued: bool = False) -> List[IDEProductMetadata]:
        """Lists IDE coding assistants with availability filtering."""
        ides = list(self._ides.values())
        if not include_discontinued:
            ides = [
                i for i in ides
                if i.availability in {ProductAvailability.GENERALLY_AVAILABLE, ProductAvailability.PUBLIC_BETA}
            ]
        return ides

    def list_app_builders(self, include_discontinued: bool = False) -> List[AppBuilderMetadata]:
        """Lists AI app builders with availability filtering."""
        builders = list(self._app_builders.values())
        if not include_discontinued:
            builders = [
                b for b in builders
                if b.availability in {ProductAvailability.GENERALLY_AVAILABLE, ProductAvailability.PUBLIC_BETA}
            ]
        return builders

    def is_model_eligible(self, item_id: str, category: ProductCategory = ProductCategory.FOUNDATION_MODEL) -> bool:
        """
        Checks if an item is eligible for the default recommendation pool.
        Foundation models must be ACTIVE, verified, with known non-stale pricing.
        Products must be GENERALLY_AVAILABLE or PUBLIC_BETA.
        """
        if category == ProductCategory.FOUNDATION_MODEL:
            m = self.get_foundation_model(item_id)
            if not m:
                return False
            # Reject if not ACTIVE
            if m.lifecycle_status != LifecycleStatus.ACTIVE:
                return False
            # Reject if critical fields are unverified
            if getattr(m, "unverified_fields", []):
                critical = {"model_id", "pricing", "context_window", "api_endpoint", "public_pricing"}
                if any(f in critical for f in m.unverified_fields):
                    return False
            # Reject if pricing is unknown, stale, or missing
            if not m.pricing.is_known or m.pricing.is_stale:
                return False
            if m.pricing.input_cost_per_m is None or m.pricing.output_cost_per_m is None:
                return False
            return True

        elif category == ProductCategory.IDE_ASSISTANT:
            ide = self.get_ide(item_id)
            if not ide:
                return False
            return ide.availability in {ProductAvailability.GENERALLY_AVAILABLE, ProductAvailability.PUBLIC_BETA}

        elif category == ProductCategory.APP_BUILDER:
            builder = self.get_app_builder(item_id)
            if not builder:
                return False
            return builder.availability in {ProductAvailability.GENERALLY_AVAILABLE, ProductAvailability.PUBLIC_BETA}

        return False

    def register_model(self, model: ModelMetadata) -> None:
        """Registers or overwrites a foundation model in the registry."""
        self._models[model.model_id] = model

    def register_ide(self, ide: IDEProductMetadata) -> None:
        """Registers or overwrites an IDE assistant in the registry."""
        self._ides[ide.product_id] = ide

    def register_app_builder(self, builder: AppBuilderMetadata) -> None:
        """Registers or overwrites an App Builder in the registry."""
        self._app_builders[builder.product_id] = builder


# Singleton registry instance
registry = ModelRegistry()
