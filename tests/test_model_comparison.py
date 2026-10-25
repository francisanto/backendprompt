"""
PromptShield AI — Test Suite for Local Model Comparison Engine.
==============================================================
Tests model registry, lifecycle eligibility, local tokenizer adapters,
feature extraction, local task classification, cost calculations, suitability
scoring, WSM and TOPSIS ranking algorithms, degenerate cases, provenance compliance,
and FastAPI endpoints.
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
    ModelPricing,
    ModelMetadata,
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
    rank_models_topsis
)
from analyzers.model_comparison.tokenizer_adapters import UnsupportedTokenizerAdapter


@pytest.fixture
def client():
    return TestClient(app)


# ---------------------------------------------------------------------------
# 1. Model Registry & Lifecycle Tests
# ---------------------------------------------------------------------------

def test_model_registry_coverage_and_lifecycle():
    # 1. Default recommendation pool excludes deprecated/retired models
    active_models = registry.list_models(include_deprecated=False)
    assert len(active_models) >= 8, f"Expected at least 8 active models, found {len(active_models)}"
    assert all(m.lifecycle_status in {LifecycleStatus.ACTIVE, LifecycleStatus.PREVIEW} for m in active_models)

    # 2. Including deprecated/retired exposes full catalogue
    all_models = registry.list_models(include_deprecated=True)
    assert len(all_models) >= 10, f"Expected at least 10 total models, found {len(all_models)}"

    providers = {m.provider for m in all_models}
    assert "openai" in providers
    assert "google" in providers
    assert "anthropic" in providers

    # 3. Active models have verified URLs and context
    gpt4o = registry.get_model("gpt-4o")
    assert gpt4o is not None
    assert gpt4o.lifecycle_status == LifecycleStatus.ACTIVE
    assert gpt4o.context_window_tokens == 128_000
    assert gpt4o.pricing.input_cost_per_m == 2.50
    assert gpt4o.pricing.output_cost_per_m == 10.00
    assert gpt4o.tokenizer_name == "o200k_base"
    assert "platform.openai.com" in gpt4o.source_url

    gemini_pro = registry.get_model("gemini-1.5-pro")
    assert gemini_pro is not None
    assert gemini_pro.lifecycle_status == LifecycleStatus.ACTIVE
    assert gemini_pro.context_window_tokens == 2_000_000
    assert "ai.google.dev" in gemini_pro.source_url

    claude_sonnet = registry.get_model("claude-3-5-sonnet")
    assert claude_sonnet is not None
    assert claude_sonnet.lifecycle_status == LifecycleStatus.ACTIVE
    assert claude_sonnet.context_window_tokens == 200_000
    assert "docs.anthropic.com" in claude_sonnet.source_url

    # 4. Check deprecated and retired entries
    gpt4_turbo = registry.get_model("gpt-4-turbo")
    assert gpt4_turbo is not None
    assert gpt4_turbo.lifecycle_status == LifecycleStatus.DEPRECATED
    assert gpt4_turbo.deprecation_date is not None

    gpt35 = registry.get_model("gpt-3.5-turbo")
    assert gpt35 is not None
    assert gpt35.lifecycle_status == LifecycleStatus.RETIRED
    assert registry.is_model_eligible("gpt-3.5-turbo") is False


# ---------------------------------------------------------------------------
# 2. Local Tokenizer Adapters Tests (Identical Prompts & Edge Cases)
# ---------------------------------------------------------------------------

def test_tokenizer_adapters_identical_prompts():
    test_prompt = (
        "Write a Python function to implement quicksort with in-place partitioning "
        "and explain its asymptotic time complexity."
    )

    openai_tok = get_tokenizer_adapter("gpt-4o")
    gemini_tok = get_tokenizer_adapter("gemini-1.5-pro")
    claude_tok = get_tokenizer_adapter("claude-3-5-sonnet")

    res_openai = openai_tok.count_tokens(test_prompt)
    res_gemini = gemini_tok.count_tokens(test_prompt)
    res_claude = claude_tok.count_tokens(test_prompt)

    # 1. OpenAI must be exact via tiktoken
    assert res_openai.is_exact is True
    assert res_openai.method == "exact"
    assert res_openai.tokenizer_name == "o200k_base"
    assert 15 <= res_openai.token_count <= 30
    assert "tiktoken" in res_openai.notes.lower()
    assert len(res_openai.billing_disclaimer) > 0

    # 2. Gemini must be explicitly estimated
    assert res_gemini.is_exact is False
    assert res_gemini.method == "estimated"
    assert 15 <= res_gemini.token_count <= 35
    assert "calibrated" in res_gemini.notes.lower()

    # 3. Claude must be explicitly estimated
    assert res_claude.is_exact is False
    assert res_claude.method == "estimated"
    assert 15 <= res_claude.token_count <= 35
    assert "calibrated" in res_claude.notes.lower()


def test_tokenizer_adapters_edge_cases():
    openai_tok = get_tokenizer_adapter("gpt-4o")
    gemini_tok = get_tokenizer_adapter("gemini-1.5-pro")

    # Empty and whitespace prompts
    for tok in [openai_tok, gemini_tok]:
        assert tok.count_tokens("").token_count == 0
        assert tok.count_tokens("   \n\t  ").token_count == 0

    # Unicode text
    unicode_prompt = "PromptShield AI: 🛡️ Análisis de Seguridad y Optimización (こんにちは)"
    res_u = openai_tok.count_tokens(unicode_prompt)
    assert res_u.token_count > 5
    assert res_u.is_exact is True

    # Unknown architecture adapter fallback
    unknown_tok = get_tokenizer_adapter("custom-llama-xyz")
    assert isinstance(unknown_tok, UnsupportedTokenizerAdapter)
    res_unk = unknown_tok.count_tokens("Some test text")
    assert res_unk.method == "unsupported"
    assert res_unk.is_exact is False


# ---------------------------------------------------------------------------
# 3. Prompt Feature Extractor Tests
# ---------------------------------------------------------------------------

def test_feature_extractor_code_vs_prose():
    code_prompt = """
    ```python
    def calculate_fibonacci(n: int) -> int:
        if n <= 1:
            return n
        return calculate_fibonacci(n - 1) + calculate_fibonacci(n - 2)
    ```
    Please optimize this recursive function using memoization.
    """

    prose_prompt = (
        "Write a descriptive, evocative poem about the quiet transition from autumn "
        "to winter in an ancient mountain village."
    )

    feat_code = feature_extractor.extract_features(code_prompt)
    feat_prose = feature_extractor.extract_features(prose_prompt)

    assert feat_code.has_code_block is True
    assert feat_code.code_proportion > 0.40
    assert "contains_fenced_code_block" in feat_code.detected_indicators

    assert feat_prose.has_code_block is False
    assert feat_prose.code_proportion < 0.10

    # Security sensitivity check
    sec_prompt = "Here is my secret token: api_key='sk-live-1234567890abcdef'. Can you verify its validity?"
    feat_sec = feature_extractor.extract_features(sec_prompt)
    assert feat_sec.security_sensitivity > 0.40
    assert "security_sensitive" in feat_sec.detected_indicators


# ---------------------------------------------------------------------------
# 4. Local Task Classifier Tests
# ---------------------------------------------------------------------------

def test_local_task_classifier_domains():
    prompts_and_expected = [
        ("Write a Rust function to parse JSON with serde and handle errors.", "code_generation"),
        ("Audit this smart contract for reentrancy bugs and integer overflows.", "security_analysis"),
        ("Write a Python script using pandas to calculate rolling 7-day sales averages.", "data_analysis"),
        ("Write a compelling short story about an astronaut discovering an alien library on Mars.", "creative_and_writing"),
        ("Prove that the square root of 2 is an irrational number by contradiction.", "reasoning_and_math"),
        ("Summarize the main arguments of this news article in three concise bullet points.", "general_instruction"),
    ]

    for p, expected_cat in prompts_and_expected:
        res = task_classifier.classify_task(p)
        assert res.predicted_category in [
            "code_generation", "security_analysis", "data_analysis",
            "creative_and_writing", "reasoning_and_math", "general_instruction"
        ]
        assert res.confidence > 0.0
        assert sum(res.category_probabilities.values()) == pytest.approx(1.0, abs=0.01)
        assert res.confidence_status in ["confident", "marginal", "low"]


# ---------------------------------------------------------------------------
# 5. Cost Calculator Tests (Known & Missing Pricing)
# ---------------------------------------------------------------------------

def test_cost_calculation():
    model = registry.get_model("gpt-4o")
    # 1,000 input tokens at $2.50/M = $0.0025
    # 500 output tokens at $10.00/M = $0.0050
    # Total = $0.0075
    cost = calculate_model_cost(model, input_tokens=1000, estimated_output_tokens=500)

    assert cost.input_tokens == 1000
    assert cost.estimated_output_tokens == 500
    assert cost.input_cost_usd == pytest.approx(0.0025, abs=1e-6)
    assert cost.output_cost_usd == pytest.approx(0.0050, abs=1e-6)
    assert cost.total_estimated_cost_usd == pytest.approx(0.0075, abs=1e-6)
    assert cost.currency == "USD"
    assert cost.pricing_status == "known"

    # Missing pricing handling
    unpriced_model = ModelMetadata(
        model_id="unpriced-model",
        display_name="Unpriced Model",
        provider="custom",
        family="custom",
        context_window_tokens=32000,
        max_output_tokens=4096,
        pricing=ModelPricing(input_cost_per_m=None, output_cost_per_m=None, is_known=False),
        tokenizer_name="custom"
    )
    cost_unpriced = calculate_model_cost(unpriced_model, input_tokens=1000, estimated_output_tokens=500)
    assert cost_unpriced.pricing_status == "unknown"
    assert cost_unpriced.total_estimated_cost_usd is None
    assert cost_unpriced.input_cost_usd is None


# ---------------------------------------------------------------------------
# 6. Weight Validation Tests
# ---------------------------------------------------------------------------

def test_weight_validation_and_normalization():
    # Negative weights clamped to zero
    w_bad = {"suitability": -0.5, "cost": 1.0, "context": 0.0}
    w_clean = validate_and_normalize_weights(w_bad)
    assert w_clean["suitability"] == 0.0
    assert w_clean["cost"] == 1.0
    assert sum(w_clean.values()) == pytest.approx(1.0, abs=1e-4)

    # Empty or zero sum falls back to preset defaults
    w_empty = validate_and_normalize_weights({})
    assert sum(w_empty.values()) == pytest.approx(1.0, abs=1e-4)
    assert w_empty["suitability"] > 0


# ---------------------------------------------------------------------------
# 7. Ranking Engine Tests (WSM & TOPSIS Identical Prompts)
# ---------------------------------------------------------------------------

def test_ranking_weighted_sum_and_topsis():
    test_prompt = "Build a scalable REST API in FastAPI with PostgreSQL, Redis caching, and Docker support."

    # 1. Test Weighted Sum Ranking
    report_wsm = comparison_engine.compare_prompt(
        prompt=test_prompt,
        method=RankingMethod.WEIGHTED_SUM,
        preset=RankingPreset.BALANCED
    )

    assert report_wsm.ranking_method == "weighted_sum"
    # By default, evaluates active models (at least 8)
    assert len(report_wsm.ranked_models) >= 8

    # Verify strictly monotonic ranks: 1, 2, ..., N
    ranks_wsm = [r.rank for r in report_wsm.ranked_models]
    assert ranks_wsm == list(range(1, len(ranks_wsm) + 1))

    # Verify scores are sorted descending
    scores_wsm = [r.score for r in report_wsm.ranked_models]
    assert scores_wsm == sorted(scores_wsm, reverse=True)

    # 2. Test TOPSIS Ranking on Identical Prompt
    report_topsis = comparison_engine.compare_prompt(
        prompt=test_prompt,
        method=RankingMethod.TOPSIS,
        preset=RankingPreset.BALANCED
    )

    assert report_topsis.ranking_method == "topsis"
    ranks_topsis = [r.rank for r in report_topsis.ranked_models]
    assert ranks_topsis == list(range(1, len(ranks_topsis) + 1))

    scores_topsis = [r.score for r in report_topsis.ranked_models]
    assert scores_topsis == sorted(scores_topsis, reverse=True)

    # 3. Test Cost-Optimized vs Quality-Optimized Presets
    report_cost = comparison_engine.compare_prompt(
        prompt=test_prompt,
        preset=RankingPreset.COST_OPTIMIZED
    )
    # The top model in cost_optimized should be a low-cost model
    top_cost_model = report_cost.ranked_models[0].model_id
    assert top_cost_model in ["gpt-4o-mini", "gemini-1.5-flash", "claude-3-5-haiku", "gemini-2.0-flash"]

    report_quality = comparison_engine.compare_prompt(
        prompt=test_prompt,
        preset=RankingPreset.QUALITY_OPTIMIZED
    )
    # Top models in quality_optimized should be flagship high-capability models
    top_quality_model = report_quality.ranked_models[0].model_id
    assert top_quality_model in ["claude-3-5-sonnet", "gpt-4o", "gemini-1.5-pro", "claude-3-opus"]


def test_ranking_degenerate_cases():
    # 1. Single candidate ranking
    single_candidate = [
        CandidateEvaluation(
            model_id="gpt-4o",
            display_name="OpenAI GPT-4o",
            provider="openai",
            token_count=100,
            token_count_method="exact",
            suitability_score=0.90,
            total_cost_usd=0.001,
            input_cost_usd=0.0005,
            output_cost_usd=0.0005,
            context_window_tokens=128000
        )
    ]
    ranked_single_wsm = rank_models_weighted_sum(single_candidate, {"suitability": 1.0})
    assert len(ranked_single_wsm) == 1
    assert ranked_single_wsm[0].rank == 1
    assert ranked_single_wsm[0].score == 1.0

    ranked_single_topsis = rank_models_topsis(single_candidate, {"suitability": 1.0})
    assert len(ranked_single_topsis) == 1
    assert ranked_single_topsis[0].rank == 1
    assert ranked_single_topsis[0].score == 1.0

    # 2. Identical criteria across candidates in TOPSIS (zero range)
    identical_candidates = [
        CandidateEvaluation(
            model_id=f"model-{i}",
            display_name=f"Model {i}",
            provider="test",
            token_count=100,
            token_count_method="exact",
            suitability_score=0.80,
            total_cost_usd=0.001,
            input_cost_usd=0.0005,
            output_cost_usd=0.0005,
            context_window_tokens=128000
        )
        for i in range(3)
    ]
    ranked_ident = rank_models_topsis(identical_candidates, {"suitability": 0.5, "cost": 0.5})
    assert len(ranked_ident) == 3
    # All scores should be finite non-NaN numbers
    for r in ranked_ident:
        assert not math.isnan(r.score)
        assert 0.0 <= r.score <= 1.0


# ---------------------------------------------------------------------------
# 8. Provenance & Non-Fabrication Tests
# ---------------------------------------------------------------------------

def test_provenance_and_no_fabricated_metrics():
    test_prompt = "Analyze database query performance."
    report = comparison_engine.compare_prompt(test_prompt)

    for rm in report.ranked_models:
        # Latency must be unavailable offline, NOT fabricated
        assert rm.latency_status == "unavailable"
        assert "not fabricate" in rm.benchmark_status.lower()

    # Model details provenance
    for mid, details in report.model_details.items():
        if "gpt" in mid:
            assert details.is_exact_token_count is True
            assert details.token_count_method == "exact"
        else:
            assert details.is_exact_token_count is False
            assert details.token_count_method == "estimated"

        assert "unavailable" in details.provenance["latency"]
        assert "unavailable" in details.provenance["external_benchmark_scores"]


# ---------------------------------------------------------------------------
# 9. FastAPI Endpoints Tests
# ---------------------------------------------------------------------------

def test_fastapi_models_endpoint(client):
    # 1. Default /models exposes active eligible models
    res = client.get("/models")
    assert res.status_code == 200
    data = res.json()
    assert isinstance(data, list)
    assert len(data) >= 8
    assert all(m["lifecycle_status"] in ["active", "preview"] for m in data)

    # 2. Querying with include_deprecated=true exposes full catalogue
    res_all = client.get("/models?include_deprecated=true")
    assert res_all.status_code == 200
    data_all = res_all.json()
    assert len(data_all) >= 10

    # 3. Filtering by specific lifecycle status
    res_dep = client.get("/models?status=deprecated")
    assert res_dep.status_code == 200
    data_dep = res_dep.json()
    assert len(data_dep) >= 1
    assert all(m["lifecycle_status"] == "deprecated" for m in data_dep)

    # 4. Invalid status returns 400
    res_bad = client.get("/models?status=invalid_status_xyz")
    assert res_bad.status_code == 400

    # 5. Filter by provider
    res_openai = client.get("/models?provider=openai")
    assert res_openai.status_code == 200
    data_openai = res_openai.json()
    assert all(m["provider"] == "openai" for m in data_openai)


def test_fastapi_get_specific_model(client):
    res = client.get("/models/gpt-4o")
    assert res.status_code == 200
    data = res.json()
    assert data["model_id"] == "gpt-4o"
    assert data["context_window_tokens"] == 128_000
    assert data["lifecycle_status"] == "active"
    assert data["source_url"] != ""

    # Deprecated model can still be retrieved by ID
    res_dep = client.get("/models/gpt-4-turbo")
    assert res_dep.status_code == 200
    assert res_dep.json()["lifecycle_status"] == "deprecated"

    # Nonexistent model 404
    res_404 = client.get("/models/fake-model-xyz")
    assert res_404.status_code == 404


def test_fastapi_compare_models_endpoint(client):
    payload = {
        "prompt": "Create a React modal dialog with keyboard accessibility (ESC to close).",
        "method": "weighted_sum",
        "preset": "balanced"
    }

    res = client.post("/compare-models", json=payload)
    assert res.status_code == 200
    data = res.json()

    assert data["status"] == "success"
    assert "prompt_features" in data
    assert "task_classification" in data
    assert "ranked_models" in data
    # Active models evaluated by default (at least 8)
    assert len(data["ranked_models"]) >= 8
    assert data["ranked_models"][0]["rank"] == 1

    # TOPSIS with specific models filter
    payload_topsis = {
        "prompt": "Explain the difference between Dijkstra and A* algorithms.",
        "method": "topsis",
        "preset": "quality_optimized",
        "models": ["gpt-4o", "claude-3-5-sonnet", "gemini-1.5-pro"]
    }
    res_topsis = client.post("/compare-models", json=payload_topsis)
    assert res_topsis.status_code == 200
    data_topsis = res_topsis.json()
    assert len(data_topsis["ranked_models"]) == 3
    assert set(data_topsis["model_details"].keys()) == {"gpt-4o", "claude-3-5-sonnet", "gemini-1.5-pro"}

    # Validation error for empty prompt
    res_bad = client.post("/compare-models", json={"prompt": "   "})
    assert res_bad.status_code == 422
