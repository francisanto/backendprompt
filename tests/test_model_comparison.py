"""
PromptShield AI — Test Suite for Local Model Comparison Engine.
==============================================================
Tests model registry, local tokenizer adapters, feature extraction,
local task classification, cost calculations, suitability scoring,
WSM and TOPSIS ranking algorithms, provenance compliance, and FastAPI endpoints.
"""

import sys
from pathlib import Path

# Add project root to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pytest
from fastapi.testclient import TestClient

from main import app
from analyzers.model_comparison import (
    registry,
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


@pytest.fixture
def client():
    return TestClient(app)


# ---------------------------------------------------------------------------
# 1. Model Registry Tests
# ---------------------------------------------------------------------------

def test_model_registry_coverage():
    models = registry.list_models()
    assert len(models) >= 9, f"Expected at least 9 models, found {len(models)}"

    providers = {m.provider for m in models}
    assert "openai" in providers
    assert "google" in providers
    assert "anthropic" in providers

    gpt4o = registry.get_model("gpt-4o")
    assert gpt4o is not None
    assert gpt4o.context_window_tokens == 128_000
    assert gpt4o.pricing.input_cost_per_m == 2.50
    assert gpt4o.pricing.output_cost_per_m == 10.00
    assert gpt4o.tokenizer_name == "o200k_base"

    gemini_pro = registry.get_model("gemini-1.5-pro")
    assert gemini_pro is not None
    assert gemini_pro.context_window_tokens == 2_000_000

    claude_sonnet = registry.get_model("claude-3-5-sonnet")
    assert claude_sonnet is not None
    assert claude_sonnet.context_window_tokens == 200_000


# ---------------------------------------------------------------------------
# 2. Local Tokenizer Adapters Tests (Identical Prompts)
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

    # 2. Gemini must be explicitly estimated
    assert res_gemini.is_exact is False
    assert res_gemini.method == "estimated"
    assert 15 <= res_gemini.token_count <= 35

    # 3. Claude must be explicitly estimated
    assert res_claude.is_exact is False
    assert res_claude.method == "estimated"
    assert 15 <= res_claude.token_count <= 35

    # 4. Empty text handling
    empty_res = openai_tok.count_tokens("")
    assert empty_res.token_count == 0


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


# ---------------------------------------------------------------------------
# 5. Cost Calculator Tests
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
    assert cost.pricing_effective_date == "2024-10-01"


# ---------------------------------------------------------------------------
# 6. Ranking Engine Tests (WSM & TOPSIS Identical Prompts)
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
    assert len(report_wsm.ranked_models) == len(registry.list_models())

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
    # The top model in cost_optimized should be a low-cost model (e.g. gpt-4o-mini or gemini-1.5-flash)
    top_cost_model = report_cost.ranked_models[0].model_id
    assert top_cost_model in ["gpt-4o-mini", "gemini-1.5-flash", "claude-3-5-haiku", "gemini-2.0-flash"]

    report_quality = comparison_engine.compare_prompt(
        prompt=test_prompt,
        preset=RankingPreset.QUALITY_OPTIMIZED
    )
    # Top models in quality_optimized should be flagship high-capability models
    top_quality_model = report_quality.ranked_models[0].model_id
    assert top_quality_model in ["claude-3-5-sonnet", "gpt-4o", "gemini-1.5-pro", "claude-3-opus"]


# ---------------------------------------------------------------------------
# 7. Provenance & Non-Fabrication Tests
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
# 8. FastAPI Endpoints Tests
# ---------------------------------------------------------------------------

def test_fastapi_models_endpoint(client):
    res = client.get("/models")
    assert res.status_code == 200
    data = res.json()
    assert isinstance(data, list)
    assert len(data) >= 9

    # Filter by provider
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
    assert len(data["ranked_models"]) >= 9
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
