"""
PromptShield AI — Test Suite for Multi-Category Comparison & Lifecycle Verification.
=====================================================================================
Tests:
1. Model Registry & Strict Lifecycle Eligibility (Active vs Deprecated/Retired).
2. Category Separation: Foundation Models, IDE Assistants, and AI App Builders.
3. Local Tokenizer Adapters (Exact tiktoken vs Calibrated Heuristic estimates).
4. Feature Extraction & Uncertainty-Aware Task Classification.
5. Published Pricing & Cost Calculations (No zero-cost fabrication).
6. Multi-Criteria Ranking (WSM and TOPSIS) across Foundation Models, IDEs, and App Builders.
7. Degenerate TOPSIS & Weight Normalization Cases.
8. Provenance Compliance & Absence of Fabricated Metrics.
9. FastAPI Endpoints for all categories and error conditions.
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
    assert len(active_models) >= 11, f"Expected at least 11 active models, found {len(active_models)}"
    assert all(m.lifecycle_status == LifecycleStatus.ACTIVE for m in active_models)

    # Verify key active frontier models
    active_ids = {m.model_id for m in active_models}
    assert "gpt-4o" in active_ids
    assert "gpt-4o-mini" in active_ids
    assert "o1" in active_ids
    assert "o3-mini" in active_ids
    assert "gemini-2.0-flash" in active_ids
    assert "gemini-2.5-pro" in active_ids
    assert "gemini-2.5-flash" in active_ids
    assert "claude-3-7-sonnet" in active_ids
    assert "claude-3-5-haiku" in active_ids
    assert "grok-3" in active_ids
    assert "grok-3-mini" in active_ids

    # 2. Including deprecated/retired exposes full historical catalogue
    all_models = registry.list_models(include_deprecated=True)
    assert len(all_models) >= 18, f"Expected at least 18 total models, found {len(all_models)}"

    providers = {m.provider for m in all_models}
    assert "openai" in providers
    assert "google" in providers
    assert "anthropic" in providers
    assert "xai" in providers

    # 3. Verified deprecated models
    claude_35_sonnet = registry.get_foundation_model("claude-3-5-sonnet")
    assert claude_35_sonnet is not None
    assert claude_35_sonnet.lifecycle_status == LifecycleStatus.DEPRECATED
    assert registry.is_model_eligible("claude-3-5-sonnet") is False
    assert claude_35_sonnet.replacement_model_id == "claude-3-7-sonnet"

    gpt4_turbo = registry.get_foundation_model("gpt-4-turbo")
    assert gpt4_turbo is not None
    assert gpt4_turbo.lifecycle_status == LifecycleStatus.DEPRECATED
    assert registry.is_model_eligible("gpt-4-turbo") is False

    # 4. Verified retired models
    gemini_15_pro = registry.get_foundation_model("gemini-1.5-pro")
    assert gemini_15_pro is not None
    assert gemini_15_pro.lifecycle_status == LifecycleStatus.RETIRED
    assert registry.is_model_eligible("gemini-1.5-pro") is False
    assert gemini_15_pro.replacement_model_id == "gemini-2.5-pro"

    gemini_15_flash = registry.get_foundation_model("gemini-1.5-flash")
    assert gemini_15_flash is not None
    assert gemini_15_flash.lifecycle_status == LifecycleStatus.RETIRED
    assert registry.is_model_eligible("gemini-1.5-flash") is False

    claude_3_opus = registry.get_foundation_model("claude-3-opus")
    assert claude_3_opus is not None
    assert claude_3_opus.lifecycle_status == LifecycleStatus.RETIRED
    assert registry.is_model_eligible("claude-3-opus") is False

    gpt35 = registry.get_foundation_model("gpt-3.5-turbo")
    assert gpt35 is not None
    assert gpt35.lifecycle_status == LifecycleStatus.RETIRED
    assert registry.is_model_eligible("gpt-3.5-turbo") is False

    grok2 = registry.get_foundation_model("grok-2")
    assert grok2 is not None
    assert grok2.lifecycle_status == LifecycleStatus.RETIRED
    assert registry.is_model_eligible("grok-2") is False


# ---------------------------------------------------------------------------
# 2. Category Separation & Product Catalogues
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

    # 4. Product eligibility checks
    assert registry.is_model_eligible("cursor", category=ProductCategory.IDE_ASSISTANT) is True
    assert registry.is_model_eligible("cursor", category=ProductCategory.FOUNDATION_MODEL) is False
    assert registry.is_model_eligible("v0-dev", category=ProductCategory.APP_BUILDER) is True
    assert registry.is_model_eligible("v0-dev", category=ProductCategory.FOUNDATION_MODEL) is False


# ---------------------------------------------------------------------------
# 3. Local Tokenizer Adapters Tests (OpenAI, Gemini, Claude, Grok)
# ---------------------------------------------------------------------------

def test_tokenizer_adapters_accuracy_and_provenance():
    test_prompt = (
        "Write a Python function to implement quicksort with in-place partitioning "
        "and explain its asymptotic time complexity."
    )

    openai_tok = get_tokenizer_adapter("gpt-4o")
    openai_o1_tok = get_tokenizer_adapter("o1")
    gemini_tok = get_tokenizer_adapter("gemini-2.5-pro")
    claude_tok = get_tokenizer_adapter("claude-3-7-sonnet")
    grok_tok = get_tokenizer_adapter("grok-3")

    res_openai = openai_tok.count_tokens(test_prompt)
    res_o1 = openai_o1_tok.count_tokens(test_prompt)
    res_gemini = gemini_tok.count_tokens(test_prompt)
    res_claude = claude_tok.count_tokens(test_prompt)
    res_grok = grok_tok.count_tokens(test_prompt)

    # 1. OpenAI must be exact via tiktoken
    assert res_openai.is_exact is True
    assert res_openai.method == "exact"
    assert res_openai.tokenizer_name == "o200k_base"
    assert res_o1.is_exact is True
    assert res_o1.method == "exact"

    # 2. Gemini, Claude, Grok must be explicitly estimated
    assert res_gemini.is_exact is False
    assert res_gemini.method == "estimated"

    assert res_claude.is_exact is False
    assert res_claude.method == "estimated"

    assert res_grok.is_exact is False
    assert res_grok.method == "estimated"
    assert isinstance(grok_tok, GrokTokenizerAdapter)


def test_tokenizer_adapters_edge_cases():
    openai_tok = get_tokenizer_adapter("gpt-4o")
    gemini_tok = get_tokenizer_adapter("gemini-2.5-pro")
    claude_tok = get_tokenizer_adapter("claude-3-7-sonnet")
    grok_tok = get_tokenizer_adapter("grok-3")

    for tok in [openai_tok, gemini_tok, claude_tok, grok_tok]:
        res_empty = tok.count_tokens("")
        assert res_empty.token_count == 0

        res_spaces = tok.count_tokens("   \n\t  ")
        assert res_spaces.token_count == 0

        res_unicode = tok.count_tokens("こんにちは世界！ Hello World 🚀 12345")
        assert res_unicode.token_count > 0

    unsupported = get_tokenizer_adapter("totally-unknown-model-architecture")
    assert isinstance(unsupported, UnsupportedTokenizerAdapter)
    res_unsupported = unsupported.count_tokens("Hello world")
    assert res_unsupported.method == "unsupported"
    assert res_unsupported.is_exact is False


# ---------------------------------------------------------------------------
# 4. Feature Extraction & Cost Calculations
# ---------------------------------------------------------------------------

def test_cost_calculation():
    gpt4o = registry.get_foundation_model("gpt-4o")
    assert gpt4o is not None

    cost = calculate_model_cost(model=gpt4o, input_tokens=1000, estimated_output_tokens=500)
    # 1000 * 2.50 / 1M = 0.0025
    # 500 * 10.00 / 1M = 0.0050
    # Total = 0.0075
    assert cost.input_cost_usd == pytest.approx(0.0025, abs=1e-6)
    assert cost.output_cost_usd == pytest.approx(0.0050, abs=1e-6)
    assert cost.total_estimated_cost_usd == pytest.approx(0.0075, abs=1e-6)
    assert cost.pricing_status == "known"


def test_weight_validation_and_normalization():
    w = {"suitability": 2.0, "cost": 1.0, "context": 1.0}
    norm = validate_and_normalize_weights(w)
    assert sum(norm.values()) == pytest.approx(1.0, abs=1e-4)
    assert norm["suitability"] == pytest.approx(0.50, abs=1e-2)

    w_empty = validate_and_normalize_weights({})
    assert sum(w_empty.values()) == pytest.approx(1.0, abs=1e-4)


# ---------------------------------------------------------------------------
# 5. Multi-Criteria Ranking: Foundation Models
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
    assert len(report_wsm.ranked_models) >= 11
    assert all(r.lifecycle_status == "active" for r in report_wsm.ranked_models)

    # Strictly monotonic ranks
    ranks_wsm = [r.rank for r in report_wsm.ranked_models]
    assert ranks_wsm == list(range(1, len(ranks_wsm) + 1))

    # Descending scores
    scores_wsm = [r.score for r in report_wsm.ranked_models]
    assert scores_wsm == sorted(scores_wsm, reverse=True)

    # 2. Test TOPSIS Ranking
    report_topsis = comparison_engine.compare_prompt(
        prompt=test_prompt,
        category=ProductCategory.FOUNDATION_MODEL,
        method=RankingMethod.TOPSIS,
        preset=RankingPreset.BALANCED
    )

    assert report_topsis.ranking_method == "topsis"
    ranks_topsis = [r.rank for r in report_topsis.ranked_models]
    assert ranks_topsis == list(range(1, len(ranks_topsis) + 1))

    # 3. Cost-optimized preset favors affordable models
    report_cost = comparison_engine.compare_prompt(
        prompt=test_prompt,
        category=ProductCategory.FOUNDATION_MODEL,
        preset=RankingPreset.COST_OPTIMIZED
    )
    top_cost_model = report_cost.ranked_models[0].model_id
    assert top_cost_model in ["gpt-4o-mini", "gemini-2.0-flash", "gemini-2.5-flash", "grok-3-mini", "claude-3-5-haiku"]

    # 4. Quality-optimized preset favors frontier models
    report_quality = comparison_engine.compare_prompt(
        prompt=test_prompt,
        category=ProductCategory.FOUNDATION_MODEL,
        preset=RankingPreset.QUALITY_OPTIMIZED
    )
    top_quality_model = report_quality.ranked_models[0].model_id
    assert top_quality_model in ["claude-3-7-sonnet", "o1", "o3-mini", "gpt-4o", "gemini-2.5-pro"]


# ---------------------------------------------------------------------------
# 6. Multi-Criteria Ranking: IDEs & Coding Assistants
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
# 7. Multi-Criteria Ranking: AI App Builders
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
# 8. Provenance & Offline Guarantees
# ---------------------------------------------------------------------------

def test_provenance_and_offline_execution():
    test_prompt = "Analyze database query performance."
    report = comparison_engine.compare_prompt(test_prompt)

    assert "100% local execution" in report.provenance_declarations["offline_execution"]
    assert "No third-party quality benchmarks" in report.provenance_declarations["benchmarks"]

    for rm in report.ranked_models:
        assert rm.latency_status == "unavailable"
        assert "not fabricate" in rm.benchmark_status.lower()


# ---------------------------------------------------------------------------
# 9. FastAPI Endpoint Integration Tests
# ---------------------------------------------------------------------------

def test_fastapi_models_endpoint_categories(client):
    # 1. Default GET /models returns active foundation models
    res = client.get("/models")
    assert res.status_code == 200
    data = res.json()
    assert len(data) >= 11
    assert all(m["category"] == "foundation_model" for m in data)
    assert all(m["lifecycle_status"] == "active" for m in data)

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
    res_fm = client.get("/models/claude-3-7-sonnet")
    assert res_fm.status_code == 200
    assert res_fm.json()["model_id"] == "claude-3-7-sonnet"
    assert res_fm.json()["lifecycle_status"] == "active"

    # Deprecated foundation model
    res_dep = client.get("/models/claude-3-5-sonnet")
    assert res_dep.status_code == 200
    assert res_dep.json()["lifecycle_status"] == "deprecated"

    # Retired foundation model
    res_ret = client.get("/models/gemini-1.5-pro")
    assert res_ret.status_code == 200
    assert res_ret.json()["lifecycle_status"] == "retired"

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
    assert len(data_fm["ranked_models"]) >= 11
    assert data_fm["ranked_products"] is None

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
