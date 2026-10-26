"""
PromptShield AI — Test Suite for Multi-Category Comparison, Freshness Audit & Lifecycle Verification.
=====================================================================================================
Tests:
1. Model Registry & Strict Lifecycle Eligibility (Active vs Deprecated/Retired).
2. Deep Audit of Target Models (gpt-4o, o1, o3-mini, gemini-2.0-flash, gemini-2.5, claude-3-7-sonnet, grok-3).
3. Stale Pricing and Unverified Metadata Rejection.
4. Category Separation: Foundation Models, IDE Assistants, and AI App Builders.
5. Local Tokenizer Adapters (Exact tiktoken vs Calibrated Heuristic estimates).
6. Published Pricing & Cost Calculations (No zero-cost fabrication).
7. Multi-Criteria Ranking (WSM and TOPSIS) with Missing Value Handling.
8. IDE and App Builder Multi-Criteria Ranking.
9. Lifecycle Filtering Enforcement (Default vs Historical Access).
10. Consistency between GET /models and POST /compare-models.
11. Provenance Compliance & Absence of Fabricated Metrics.
12. FastAPI Endpoints for all categories, specific lookups, and error conditions.
"""

import math
from pathlib import Path
import sys

# Add project root to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pytest
from fastapi.testclient import TestClient

from main import app
from analyzers.model_comparison import (
    registry,
    LifecycleStatus,
    ProductCategory,
    ProductAvailability,
    ModelPricing,
    ModelMetadata,
    IDEProductMetadata,
    AppBuilderMetadata,
    get_tokenizer_adapter,
    OpenAITokenizerAdapter,
    GeminiTokenizerAdapter,
    ClaudeTokenizerAdapter,
    feature_extractor,
    task_classifier,
    calculate_model_cost,
    calculate_suitability,
    RankingMethod,
    RankingPreset,
    rank_models,
    comparison_engine
)
from analyzers.model_comparison.ranker import (
    CandidateEvaluation,
    validate_and_normalize_weights,
    rank_models_weighted_sum,
    rank_models_topsis,
    rank_products
)
from analyzers.model_comparison.tokenizer_adapters import (
    UnsupportedTokenizerAdapter,
    GrokTokenizerAdapter
)


@pytest.fixture
def client():
    return TestClient(app)


# ---------------------------------------------------------------------------
# 1. Model Registry & Strict Lifecycle Tests
# ---------------------------------------------------------------------------

def test_model_registry_coverage_and_lifecycle():
    # 1. Default recommendation pool excludes deprecated and retired models
    active_models = registry.list_models(include_deprecated=False)
    assert len(active_models) == 10, f"Expected exactly 10 active models, found {len(active_models)}"
    assert all(m.lifecycle_status == LifecycleStatus.ACTIVE for m in active_models)
    assert all(registry.is_model_eligible(m.model_id) for m in active_models)

    # Verify key active frontier models
    active_ids = {m.model_id for m in active_models}
    assert "gpt-6-astra" in active_ids
    assert "gpt-6-sol" in active_ids
    assert "gpt-6-luna" in active_ids
    assert "gemini-4-argon" in active_ids
    assert "gemini-3-8-flash" in active_ids
    assert "claude-sonnet-5-5" in active_ids
    assert "claude-haiku-5-5" in active_ids
    assert "claude-opus-5-5" in active_ids
    assert "grok-4-7" in active_ids
    assert "grok-4-3" in active_ids

    # 2. Including deprecated/retired exposes full historical catalogue
    all_models = registry.list_models(include_deprecated=True)
    assert len(all_models) >= 20, f"Expected at least 20 total models, found {len(all_models)}"

    providers = {m.provider for m in all_models}
    assert "openai" in providers
    assert "google" in providers
    assert "anthropic" in providers
    assert "xai" in providers


# ---------------------------------------------------------------------------
# 2. Audit Target Models Lifecycle & Provenance Verification
# ---------------------------------------------------------------------------

def test_audit_target_models_lifecycle_and_provenance():
    """
    Validates lifecycle status, documented replacements, and ineligibility for default recommendations
    for all 11 audit targets from provider deprecation documentation.
    """
    audit_targets = [
        ("gemini-2.0-flash", LifecycleStatus.RETIRED, "gemini-3-8-flash"),
        ("claude-3-5-haiku", LifecycleStatus.RETIRED, "claude-haiku-5-5"),
        ("gpt-4o", LifecycleStatus.DEPRECATED, "gpt-6-sol"),
        ("gpt-4o-mini", LifecycleStatus.DEPRECATED, "gpt-6-luna"),
        ("o1", LifecycleStatus.RETIRED, "gpt-6-astra"),
        ("o3-mini", LifecycleStatus.RETIRED, "gpt-6-luna"),
        ("gemini-2.5-pro", LifecycleStatus.DEPRECATED, "gemini-4-argon"),
        ("gemini-2.5-flash", LifecycleStatus.DEPRECATED, "gemini-3-8-flash"),
        ("claude-3-7-sonnet", LifecycleStatus.RETIRED, "claude-sonnet-5-5"),
        ("grok-3", LifecycleStatus.RETIRED, "grok-4-7"),
        ("grok-3-mini", LifecycleStatus.RETIRED, "grok-4-3"),
    ]

    for model_id, expected_status, expected_replacement in audit_targets:
        m = registry.get_foundation_model(model_id)
        assert m is not None, f"Audit target '{model_id}' must exist in historical catalogue"
        assert m.lifecycle_status == expected_status, (
            f"Model '{model_id}' expected status {expected_status}, got {m.lifecycle_status}"
        )
        assert registry.is_model_eligible(model_id) is False, (
            f"Audit target '{model_id}' must NOT be eligible for default recommendation pool"
        )
        assert m.replacement_model_id == expected_replacement, (
            f"Model '{model_id}' expected replacement '{expected_replacement}', got '{m.replacement_model_id}'"
        )
        assert m.source_url.startswith("http"), f"Model '{model_id}' must have official source URL"
        assert m.last_audit_date == "2026-10-25"


# ---------------------------------------------------------------------------
# 3. Stale Pricing and Unverified Rejection Tests
# ---------------------------------------------------------------------------

def test_stale_pricing_and_unverified_exclusion():
    # 1. Model with stale pricing is excluded from default recommendation pool
    stale_model = ModelMetadata(
        model_id="test-stale-pricing-model",
        display_name="Test Stale Model",
        provider="openai",
        family="test",
        context_window_tokens=128000,
        max_output_tokens=4096,
        pricing=ModelPricing(
            input_cost_per_m=1.0,
            output_cost_per_m=2.0,
            is_stale=True  # Stale pricing flag
        ),
        tokenizer_name="cl100k_base",
        lifecycle_status=LifecycleStatus.ACTIVE
    )
    registry.register_model(stale_model)
    try:
        assert registry.is_model_eligible("test-stale-pricing-model") is False
        active_list = [m.model_id for m in registry.list_models(include_deprecated=False)]
        assert "test-stale-pricing-model" not in active_list
    finally:
        # Cleanup
        if "test-stale-pricing-model" in registry._models:
            del registry._models["test-stale-pricing-model"]

    # 2. Model with unverified lifecycle is excluded from default recommendation pool
    unverified_model = ModelMetadata(
        model_id="test-unverified-model",
        display_name="Test Unverified Model",
        provider="anthropic",
        family="test",
        context_window_tokens=128000,
        max_output_tokens=4096,
        pricing=ModelPricing(input_cost_per_m=1.0, output_cost_per_m=2.0),
        tokenizer_name="cl100k_base",
        lifecycle_status=LifecycleStatus.UNVERIFIED
    )
    registry.register_model(unverified_model)
    try:
        assert registry.is_model_eligible("test-unverified-model") is False
        active_list = [m.model_id for m in registry.list_models(include_deprecated=False)]
        assert "test-unverified-model" not in active_list
    finally:
        if "test-unverified-model" in registry._models:
            del registry._models["test-unverified-model"]


# ---------------------------------------------------------------------------
# 4. Category Separation & Product Catalogues
# ---------------------------------------------------------------------------

def test_category_separation_and_catalogues():
    # 1. Foundation models do not contain IDEs or App Builders
    f_models = registry.list_models()
    f_ids = {m.model_id for m in f_models}
    assert "cursor" not in f_ids
    assert "v0-dev" not in f_ids

    # 2. IDE catalogue contains verified coding assistants
    ides = registry.list_ides()
    assert len(ides) == 4
    ide_ids = {i.product_id for i in ides}
    assert "cursor" in ide_ids
    assert "windsurf" in ide_ids
    assert "github-copilot" in ide_ids
    assert "google-antigravity" in ide_ids
    assert all(i.category == ProductCategory.IDE_ASSISTANT for i in ides)
    assert all(i.availability == ProductAvailability.GENERALLY_AVAILABLE for i in ides)
    assert all(i.last_audit_date == "2026-10-25" for i in ides)

    # 3. App Builder catalogue contains verified platforms
    builders = registry.list_app_builders()
    assert len(builders) == 4
    builder_ids = {b.product_id for b in builders}
    assert "v0-dev" in builder_ids
    assert "bolt-new" in builder_ids
    assert "lovable-dev" in builder_ids
    assert "replit-agent" in builder_ids
    assert all(b.category == ProductCategory.APP_BUILDER for b in builders)
    assert all(b.availability == ProductAvailability.GENERALLY_AVAILABLE for b in builders)
    assert all(b.last_audit_date == "2026-10-25" for b in builders)

    # 4. Product eligibility checks
    assert registry.is_model_eligible("cursor", category=ProductCategory.IDE_ASSISTANT) is True
    assert registry.is_model_eligible("cursor", category=ProductCategory.FOUNDATION_MODEL) is False
    assert registry.is_model_eligible("v0-dev", category=ProductCategory.APP_BUILDER) is True
    assert registry.is_model_eligible("v0-dev", category=ProductCategory.FOUNDATION_MODEL) is False


# ---------------------------------------------------------------------------
# 5. Local Tokenizer Adapters Tests (OpenAI, Gemini, Claude, Grok)
# ---------------------------------------------------------------------------

def test_tokenizer_adapters_accuracy_and_provenance():
    test_prompt = (
        "Write a Python function to implement quicksort with in-place partitioning "
        "and explain its asymptotic time complexity."
    )

    openai_tok = get_tokenizer_adapter("gpt-6-astra")
    gemini_tok = get_tokenizer_adapter("gemini-4-argon")
    claude_tok = get_tokenizer_adapter("claude-sonnet-5-5")
    grok_tok = get_tokenizer_adapter("grok-4-7")

    res_openai = openai_tok.count_tokens(test_prompt)
    assert res_openai.is_exact is True
    assert res_openai.method == "exact"
    assert res_openai.tokenizer_name == "o200k_base"
    assert res_openai.token_count > 0

    res_gemini = gemini_tok.count_tokens(test_prompt)
    assert res_gemini.is_exact is False
    assert res_gemini.method == "estimated"
    assert "gemini" in res_gemini.tokenizer_name

    res_claude = claude_tok.count_tokens(test_prompt)
    assert res_claude.is_exact is False
    assert res_claude.method == "estimated"
    assert "claude" in res_claude.tokenizer_name

    res_grok = grok_tok.count_tokens(test_prompt)
    assert res_grok.is_exact is False
    assert res_grok.method == "estimated"

    # Unsupported adapter fallback
    unsupported = get_tokenizer_adapter("unknown-custom-architecture-99")
    assert isinstance(unsupported, UnsupportedTokenizerAdapter)
    res_unsup = unsupported.count_tokens(test_prompt)
    assert res_unsup.method == "unsupported"


# ---------------------------------------------------------------------------
# 6. Pricing & Cost Calculations (No zero-cost fabrication)
# ---------------------------------------------------------------------------

def test_cost_calculation_freshness_and_explicit_status():
    m = registry.get_foundation_model("gpt-6-astra")
    assert m is not None

    cost = calculate_model_cost(model=m, input_tokens=1000, estimated_output_tokens=500)
    assert cost.pricing_status == "known"
    assert cost.input_cost_usd is not None
    assert cost.output_cost_usd is not None
    assert cost.total_estimated_cost_usd is not None
    assert cost.total_estimated_cost_usd > 0
    assert cost.pricing_effective_date == "2026-09-15"

    # Unpriced model handling
    unpriced = ModelMetadata(
        model_id="unpriced-test-model",
        display_name="Unpriced Model",
        provider="custom",
        family="custom",
        context_window_tokens=128000,
        max_output_tokens=4096,
        pricing=ModelPricing(input_cost_per_m=None, output_cost_per_m=None, is_known=False),
        tokenizer_name="cl100k_base"
    )
    cost_unpriced = calculate_model_cost(model=unpriced, input_tokens=1000, estimated_output_tokens=500)
    assert cost_unpriced.pricing_status == "unknown"
    assert cost_unpriced.total_estimated_cost_usd is None


# ---------------------------------------------------------------------------
# 7. Multi-Criteria Ranking: Foundation Models
# ---------------------------------------------------------------------------

def test_foundation_model_ranking_wsm_and_topsis():
    test_prompt = "Build a scalable REST API in FastAPI with PostgreSQL, Redis caching, and Docker support."

    # 1. Test Weighted Sum Ranking
    report_wsm = comparison_engine.compare_prompt(
        prompt=test_prompt,
        category=ProductCategory.FOUNDATION_MODEL,
        method=RankingMethod.WEIGHTED_SUM,
        preset=RankingPreset.BALANCED
    )

    assert report_wsm.category == "foundation_model"
    assert report_wsm.ranking_method == "weighted_sum"
    assert len(report_wsm.ranked_models) == 10
    assert all(r.lifecycle_status == "active" for r in report_wsm.ranked_models)
    assert all(r.score_type == "heuristic_recommendation_score" for r in report_wsm.ranked_models)

    # Strictly monotonic sequential ranks
    ranks_wsm = [r.rank for r in report_wsm.ranked_models]
    assert ranks_wsm == list(range(1, 11))

    # Descending scores
    scores_wsm = [r.score for r in report_wsm.ranked_models]
    assert scores_wsm == sorted(scores_wsm, reverse=True)
    assert all(0.0 <= s <= 1.0 for s in scores_wsm)

    # 2. Test TOPSIS Ranking
    report_topsis = comparison_engine.compare_prompt(
        prompt=test_prompt,
        category=ProductCategory.FOUNDATION_MODEL,
        method=RankingMethod.TOPSIS,
        preset=RankingPreset.BALANCED
    )

    assert report_topsis.ranking_method == "topsis"
    ranks_topsis = [r.rank for r in report_topsis.ranked_models]
    assert ranks_topsis == list(range(1, 11))

    # 3. Cost-optimized preset favors affordable models
    report_cost = comparison_engine.compare_prompt(
        prompt=test_prompt,
        category=ProductCategory.FOUNDATION_MODEL,
        preset=RankingPreset.COST_OPTIMIZED
    )
    top_cost_model = report_cost.ranked_models[0].model_id
    assert top_cost_model in ["gpt-6-luna", "gemini-3-8-flash", "claude-haiku-5-5", "grok-4-3"]

    # 4. Quality-optimized preset favors frontier models
    report_quality = comparison_engine.compare_prompt(
        prompt=test_prompt,
        category=ProductCategory.FOUNDATION_MODEL,
        preset=RankingPreset.QUALITY_OPTIMIZED
    )
    top_quality_model = report_quality.ranked_models[0].model_id
    assert top_quality_model in ["gpt-6-astra", "claude-sonnet-5-5", "claude-opus-5-5", "gemini-4-argon"]


# ---------------------------------------------------------------------------
# 8. Missing Price Penalty in Ranking Tests
# ---------------------------------------------------------------------------

def test_missing_price_penalty_in_ranking():
    """
    Ensures models with missing or unverified pricing receive maximum cost penalties
    and are never treated as free or lowest cost.
    """
    candidates = [
        CandidateEvaluation(
            model_id="known-expensive",
            display_name="Known Expensive",
            provider="openai",
            token_count=100,
            token_count_method="exact",
            suitability_score=0.90,
            total_cost_usd=0.05,
            input_cost_usd=0.01,
            output_cost_usd=0.04,
            context_window_tokens=128000,
            pricing_status="known"
        ),
        CandidateEvaluation(
            model_id="known-cheap",
            display_name="Known Cheap",
            provider="google",
            token_count=100,
            token_count_method="estimated",
            suitability_score=0.88,
            total_cost_usd=0.0005,
            input_cost_usd=0.0001,
            output_cost_usd=0.0004,
            context_window_tokens=128000,
            pricing_status="known"
        ),
        CandidateEvaluation(
            model_id="unpriced-model",
            display_name="Unpriced Model",
            provider="custom",
            token_count=100,
            token_count_method="estimated",
            suitability_score=0.89,
            total_cost_usd=None,
            input_cost_usd=None,
            output_cost_usd=None,
            context_window_tokens=128000,
            pricing_status="unknown"
        )
    ]

    # In WSM:
    ranked_wsm = rank_models_weighted_sum(candidates, weights={"suitability": 0.3, "cost": 0.6, "context": 0.1})
    unpriced_rank = next(r for r in ranked_wsm if r.model_id == "unpriced-model")
    cheap_rank = next(r for r in ranked_wsm if r.model_id == "known-cheap")

    # The unpriced model must NOT outrank the verified cheap model on cost
    assert cheap_rank.score > unpriced_rank.score
    assert "Pricing metadata unverified" in unpriced_rank.reasoning

    # In TOPSIS:
    ranked_topsis = rank_models_topsis(candidates, weights={"suitability": 0.3, "cost": 0.6, "context": 0.1})
    unpriced_t_rank = next(r for r in ranked_topsis if r.model_id == "unpriced-model")
    cheap_t_rank = next(r for r in ranked_topsis if r.model_id == "known-cheap")
    assert cheap_t_rank.score > unpriced_t_rank.score


# ---------------------------------------------------------------------------
# 9. Multi-Criteria Ranking: IDEs & Coding Assistants
# ---------------------------------------------------------------------------

def test_ide_assistant_ranking():
    prompt = "Modify an existing TypeScript monorepo to add an authenticated API endpoint, update tests, and explain the affected files."

    # 1. WSM Balanced Ranking
    report_wsm = comparison_engine.compare_prompt(
        prompt=prompt,
        category=ProductCategory.IDE_ASSISTANT,
        method=RankingMethod.WEIGHTED_SUM,
        preset="balanced"
    )

    assert report_wsm.category == "ide_assistant"
    assert report_wsm.ranked_products is not None
    assert len(report_wsm.ranked_products) == 4
    assert len(report_wsm.ranked_models) == 0

    ranks = [rp.rank for rp in report_wsm.ranked_products]
    assert ranks == [1, 2, 3, 4]

    product_ids = {rp.product_id for rp in report_wsm.ranked_products}
    assert product_ids == {"cursor", "windsurf", "github-copilot", "google-antigravity"}
    assert all(rp.score_type == "heuristic_recommendation_score" for rp in report_wsm.ranked_products)

    # 2. TOPSIS Agentic Power Ranking
    report_topsis = comparison_engine.compare_prompt(
        prompt=prompt,
        category=ProductCategory.IDE_ASSISTANT,
        method=RankingMethod.TOPSIS,
        preset="agentic_power"
    )
    assert report_topsis.ranking_method == "topsis"
    assert report_topsis.ranked_products is not None
    assert len(report_topsis.ranked_products) == 4
    top_agentic = report_topsis.ranked_products[0].product_id
    assert top_agentic in ["google-antigravity", "cursor", "windsurf"]


# ---------------------------------------------------------------------------
# 10. Multi-Criteria Ranking: AI App Builders
# ---------------------------------------------------------------------------

def test_app_builder_ranking():
    prompt = "Build a full-stack event registration application with authentication, PostgreSQL database, admin dashboards, and deployment."

    # 1. Full-stack preset favors Lovable / Replit
    report_fs = comparison_engine.compare_prompt(
        prompt=prompt,
        category=ProductCategory.APP_BUILDER,
        method=RankingMethod.WEIGHTED_SUM,
        preset="full_stack"
    )

    assert report_fs.category == "app_builder"
    assert report_fs.ranked_products is not None
    assert len(report_fs.ranked_products) == 4
    assert len(report_fs.ranked_models) == 0

    top_builder = report_fs.ranked_products[0].product_id
    assert top_builder in ["lovable-dev", "replit-agent"]
    assert all(rp.score_type == "heuristic_recommendation_score" for rp in report_fs.ranked_products)

    # 2. Frontend UI prompt favors v0
    ui_prompt = "Generate a responsive React landing page with Tailwind CSS, hero banner, dark mode, and pricing tables."
    report_ui = comparison_engine.compare_prompt(
        prompt=ui_prompt,
        category=ProductCategory.APP_BUILDER,
        method=RankingMethod.WEIGHTED_SUM,
        preset="frontend_ui"
    )
    assert report_ui.ranked_products[0].product_id in ["v0-dev", "lovable-dev"]


# ---------------------------------------------------------------------------
# 11. Lifecycle Filtering Enforcement (Default vs Historical Access)
# ---------------------------------------------------------------------------

def test_retired_models_excluded_from_ranking():
    # If user requests retired models in default mode (include_deprecated=False),
    # they must be filtered out by shared eligibility policy
    report_default = comparison_engine.compare_prompt(
        prompt="Write a quicksort in C++",
        category=ProductCategory.FOUNDATION_MODEL,
        model_ids=["gemini-2.0-flash", "o1", "gpt-4o", "gpt-6-astra"],
        include_deprecated=False
    )
    ranked_ids = [r.model_id for r in report_default.ranked_models]
    assert "gpt-6-astra" in ranked_ids
    assert "gemini-2.0-flash" not in ranked_ids
    assert "o1" not in ranked_ids
    assert "gpt-4o" not in ranked_ids

    # If user explicitly requests historical access (include_deprecated=True),
    # retired models can be evaluated for historical trade-off analysis
    report_historical = comparison_engine.compare_prompt(
        prompt="Write a quicksort in C++",
        category=ProductCategory.FOUNDATION_MODEL,
        model_ids=["gemini-2.0-flash", "o1", "gpt-4o", "gpt-6-astra"],
        include_deprecated=True
    )
    hist_ranked_ids = [r.model_id for r in report_historical.ranked_models]
    assert "gemini-2.0-flash" in hist_ranked_ids
    assert "o1" in hist_ranked_ids
    assert "gpt-4o" in hist_ranked_ids


# ---------------------------------------------------------------------------
# 12. Consistency Between GET /models and POST /compare-models
# ---------------------------------------------------------------------------

def test_consistency_between_get_models_and_compare_models(client):
    # GET /models returns the active foundation models
    res_get = client.get("/models")
    assert res_get.status_code == 200
    get_models = res_get.json()
    get_ids = sorted([m["model_id"] for m in get_models])

    # POST /compare-models default pool should match exactly
    res_post = client.post("/compare-models", json={
        "prompt": "Evaluate API authentication.",
        "category": "foundation_model"
    })
    assert res_post.status_code == 200
    post_data = res_post.json()
    post_ids = sorted([m["model_id"] for m in post_data["ranked_models"]])

    assert get_ids == post_ids, f"Mismatch between GET /models ({get_ids}) and POST /compare-models ({post_ids})"


# ---------------------------------------------------------------------------
# 13. Provenance & Offline Guarantees
# ---------------------------------------------------------------------------

def test_provenance_and_offline_execution():
    test_prompt = "Analyze database query performance."
    report = comparison_engine.compare_prompt(test_prompt)

    assert "100% local execution" in report.provenance_declarations["offline_execution"]
    assert "No third-party quality benchmarks" in report.provenance_declarations["benchmarks"]
    assert "heuristic" in report.provenance_declarations["scoring_interpretation"].lower()

    for rm in report.ranked_models:
        assert rm.latency_status == "unavailable"
        assert "not fabricate" in rm.benchmark_status.lower()
        assert rm.score_type == "heuristic_recommendation_score"
        assert "heuristic" in rm.reasoning.lower()


# ---------------------------------------------------------------------------
# 14. FastAPI Endpoint Integration Tests
# ---------------------------------------------------------------------------

def test_fastapi_models_endpoint_categories(client):
    # 1. Default GET /models returns active foundation models
    res = client.get("/models")
    assert res.status_code == 200
    data = res.json()
    assert len(data) == 10
    assert all(m["category"] == "foundation_model" for m in data)
    assert all(m["lifecycle_status"] == "active" for m in data)
    assert all(m["last_audit_date"] == "2026-10-25" for m in data)

    # 2. GET /models?category=ide_assistant returns 4 IDEs
    res_ide = client.get("/models?category=ide_assistant")
    assert res_ide.status_code == 200
    data_ide = res_ide.json()
    assert len(data_ide) == 4
    assert all(m["category"] == "ide_assistant" for m in data_ide)

    # 3. GET /models?category=app_builder returns 4 App Builders
    res_b = client.get("/models?category=app_builder")
    assert res_b.status_code == 200
    data_b = res_b.json()
    assert len(data_b) == 4
    assert all(m["category"] == "app_builder" for m in data_b)

    # 4. Unknown category returns 400
    res_bad = client.get("/models?category=invalid_category")
    assert res_bad.status_code == 400


def test_fastapi_get_specific_items(client):
    # Active foundation model
    res_fm = client.get("/models/gpt-6-astra")
    assert res_fm.status_code == 200
    assert res_fm.json()["model_id"] == "gpt-6-astra"
    assert res_fm.json()["lifecycle_status"] == "active"
    assert res_fm.json()["last_audit_date"] == "2026-10-25"

    # Deprecated foundation model
    res_dep = client.get("/models/gpt-4o")
    assert res_dep.status_code == 200
    assert res_dep.json()["lifecycle_status"] == "deprecated"
    assert res_dep.json()["replacement_model_id"] == "gpt-6-sol"

    # Retired foundation model
    res_ret = client.get("/models/gemini-2.0-flash")
    assert res_ret.status_code == 200
    assert res_ret.json()["lifecycle_status"] == "retired"
    assert res_ret.json()["replacement_model_id"] == "gemini-3-8-flash"

    # IDE item
    res_ide = client.get("/models/cursor")
    assert res_ide.status_code == 200
    assert res_ide.json()["category"] == "ide_assistant"

    # App builder item
    res_app = client.get("/models/v0-dev")
    assert res_app.status_code == 200
    assert res_app.json()["category"] == "app_builder"

    # Nonexistent ID returns 404
    res_404 = client.get("/models/nonexistent-item-12345")
    assert res_404.status_code == 404


def test_fastapi_compare_models_endpoint_all_categories(client):
    # 1. Foundation model comparison
    payload_fm = {
        "prompt": "Review this FastAPI authentication design for security flaws and recommend fixes.",
        "category": "foundation_model",
        "method": "weighted_sum",
        "preset": "balanced"
    }
    res_fm = client.post("/compare-models", json=payload_fm)
    assert res_fm.status_code == 200
    data_fm = res_fm.json()
    assert data_fm["category"] == "foundation_model"
    assert len(data_fm["ranked_models"]) == 10
    assert data_fm["ranked_products"] is None
    assert all(r["score_type"] == "heuristic_recommendation_score" for r in data_fm["ranked_models"])

    # 2. IDE assistant comparison
    payload_ide = {
        "prompt": "Modify an existing TypeScript monorepo to add an authenticated API endpoint, update tests, and explain the affected files.",
        "category": "ide_assistant",
        "method": "weighted_sum",
        "preset": "balanced"
    }
    res_ide = client.post("/compare-models", json=payload_ide)
    assert res_ide.status_code == 200
    data_ide = res_ide.json()
    assert data_ide["category"] == "ide_assistant"
    assert len(data_ide["ranked_models"]) == 0
    assert len(data_ide["ranked_products"]) == 4
    assert data_ide["ranked_products"][0]["rank"] == 1
    assert all(r["score_type"] == "heuristic_recommendation_score" for r in data_ide["ranked_products"])

    # 3. App Builder comparison
    payload_builder = {
        "prompt": "Build a full-stack event registration application with authentication, PostgreSQL, admin dashboards, and deployment.",
        "category": "app_builder",
        "method": "weighted_sum",
        "preset": "full_stack"
    }
    res_builder = client.post("/compare-models", json=payload_builder)
    assert res_builder.status_code == 200
    data_builder = res_builder.json()
    assert data_builder["category"] == "app_builder"
    assert len(data_builder["ranked_models"]) == 0
    assert len(data_builder["ranked_products"]) == 4
    assert data_builder["ranked_products"][0]["rank"] == 1

    # 4. Unsupported category returns 422 (Pydantic validation error)
    res_bad_cat = client.post("/compare-models", json={"prompt": "Test", "category": "invalid_cat"})
    assert res_bad_cat.status_code == 422
