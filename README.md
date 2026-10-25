# backendprompt

# PromptShield AI
## Intelligent Prompt Analysis, Security Assessment and Multi-Model Optimization Platform

PromptShield AI is a security-focused platform designed to analyze prompts used with AI coding agents and generative AI systems.

The system is designed to identify security risks arising from:

1. The prompt itself
2. The application or system that the prompt asks an AI coding agent to build
3. Sensitive information contained in the prompt

The platform will eventually provide:

- Security risk analysis
- Prompt security analysis
- Sensitive-data detection
- Security recommendations
- Prompt optimization
- Token analysis
- Cost estimation
- Multi-model comparison
- Model ranking
- Multi-model optimization

---

# Current Project Stage

## 🔄 CURRENT STAGE: Prompt Security Model + Application Risk Baseline

The project currently has a baseline **Application Risk Prediction Model** and is now implementing a separate **Prompt Security Model** using the MPDD malicious-prompt dataset.

---

# Current Progress

| Component | Status |
|---|---|
| FastAPI backend | ✅ Complete |
| `/analyze` API | ✅ Complete |
| Unified Terminal CLI (`cli.py`) | ✅ Complete |
| Basic prompt analyzer orchestrator | ✅ Complete |
| Initial Application Risk dataset | ✅ Complete |
| Application Risk dataset validation | ✅ Complete |
| Application Risk baseline model | ✅ Complete |
| TF-IDF for Application Risk | ✅ Complete |
| Multi-label Logistic Regression | ✅ Complete |
| Application Risk prediction script | ✅ Complete |
| Application Risk error analysis | ✅ Complete |
| MPDD dataset acquisition | ✅ Complete |
| MPDD dataset inspection | ✅ Complete |
| MPDD preprocessing pipeline | ✅ Complete |
| MPDD cleaned dataset | ✅ Complete |
| Prompt Security Model | ✅ Complete |
| Prompt Security model training | ✅ Complete |
| Prompt Security evaluation | ✅ Complete |
| Prompt Security testing | ✅ Complete |
| Sensitive Data Detector | ✅ Complete |
| Security recommendations engine | ✅ Complete |
| Probability calibration | ⬜ |
| Context extraction | ⬜ |
| Improved Application Risk Engine | ⬜ |
| 30-risk Application Risk taxonomy | 🔄 Planned |
| Risk aggregation | ⬜ |
| Security explanations | ⬜ |
| Prompt Optimizer | ⬜ Planned |
| Token analysis | ✅ Complete (Exact tiktoken + Calibrated Gemini/Claude) |
| Cost estimation | ✅ Complete (Separated input & estimated output) |
| Model registry & lifecycle | ✅ Complete (Active, Deprecated, Retired, Preview) |
| Multi-model comparison | ✅ Complete (`/compare-models`) |
| Model ranking (WSM & TOPSIS) | ✅ Complete (Balanced, Cost, Quality, Long-Context) |
| Frontend integration | ⬜ Planned |

---

# 🚀 Terminal CLI Usage (`cli.py`)

PromptShield AI provides an interactive, unified Terminal CLI suitable for local development and live demonstrations (e.g., college project expos).

### Starting the Interactive CLI

```bash
python cli.py
```

### Interactive Workflow

1. **PromptShield AI Banner**: Launches with the system header and diagnostic checks.
2. **Multiline Prompt Entry**:
   - Type or paste your prompt across multiple lines.
   - Type **`END`** on a new line by itself to submit and analyze.
   - Type **`exit`** or **`quit`** on the first line (or press `Ctrl+C`) to exit cleanly.
3. **Automated Multi-Model Analysis**:
   - **Prompt Security Model**: Evaluates adversarial malice, prompt injection, and jailbreak attempts (MPDD-trained binary classifier).
   - **Application Risk Model**: Evaluates 15 architectural attack surfaces implied by requested software features.
   - **Sensitive Data Detector**: Scans for leaked credentials, API keys, private keys, and database connection strings.
4. **Unified Security Report**:
   - Displays real confidence scores, classifications, and identified risk labels.
   - Automatically redacts sensitive credentials from all terminal output.
   - Clearly distinguishes *potential application attack surfaces* from *confirmed vulnerabilities*.
   - Never treats a missing or failed model as benign.
   - Delivers actionable, concrete security remediation recommendations.
5. **Next-Action Menu**:
   - `[1] Improve this prompt`: Displays the planned feature notice for automated prompt rewriting/optimization.
   - `[2] Analyze another prompt`: Prompts for the next prompt.
   - `[3] Exit PromptShield AI`: Exits cleanly.

### Non-Interactive / Scripting Mode

You can also run single-shot evaluations using command-line arguments:

```bash
# Analyze a direct prompt string
python cli.py --prompt "Build a web app with user login, password reset, and SQL database."

# Analyze a prompt from a file
python cli.py --file path/to/prompt.txt

# Disable ANSI colors (useful for pipelines or plain log files)
python cli.py --prompt "Build an API" --no-color
```

---

# Current ML Architecture

PromptShield uses two separate security prediction problems evaluated in parallel:

```text
                                  USER PROMPT
                                       │
        ┌──────────────────────────────┼──────────────────────────────┐
        │                              │                              │
        ▼                              ▼                              ▼
Prompt Security Model        Application Risk Model        Sensitive Data Detector
        │                              │                              │
        ▼                              ▼                              ▼
 "Is the prompt                "What risks could the         "Does the prompt contain
  malicious / jailbreak?"       requested application         hardcoded API keys,
                                potentially introduce?"       tokens, or passwords?"
        │                              │                              │
        ▼                              ▼                              ▼
  MPDD Dataset                  Contextual Dataset           Regex Pattern Heuristics
        │                              │                              │
        ▼                              ▼                              ▼
Binary Classification         Multi-Label Classification     Automated Secret Redaction
(Benign vs. Malicious)         (15 Risk Categories)           (Masked Output)
        │                              │                              │
        └──────────────────────────────┬──────────────────────────────┘
                                       │
                                       ▼
                       Unified Security Report & Recommendations
```

---

# 🧠 Local Model Comparison & Multi-Criteria Ranking Engine

PromptShield AI includes a **100% offline, local foundation model comparison and ranking engine** that evaluates prompts across leading LLMs (OpenAI GPT, Google Gemini, and Anthropic Claude) without requiring external provider generation API calls or API keys.

```text
                             USER PROMPT
                                  │
                                  ▼
               ┌─────────────────────────────────────┐
               │    FastAPI Validation & Router      │
               └──────────────────┬──────────────────┘
                                  │
         ┌────────────────────────┴────────────────────────┐
         │                                                 │
         ▼                                                 ▼
┌──────────────────────────────┐          ┌──────────────────────────────┐
│  Prompt Feature Extraction   │          │   Local Task Classifier      │
│  - Length (chars/words)      │          │   - TF-IDF + Multinomial NB  │
│  - Code proportion           │          │   - 6 domain categories      │
│  - Structural complexity     │          │   - Confidence scores        │
│  - Security sensitivity      │          │   - Uncertainty indicators   │
│  - Expected response length  │          │                              │
└──────────────┬───────────────┘          └──────────────┬───────────────┘
               │                                         │
               └──────────────────┬──────────────────────┘
                                  │
                                  ▼
                 ┌─────────────────────────────────┐
                 │     Model Registry Snapshot     │
                 │     - Active vs. Deprecated     │
                 │     - Verified 2025 Pricing     │
                 │     - Context limits (128k - 2M)│
                 └────────────────┬────────────────┘
                                  │
       ┌──────────────────────────┼──────────────────────────┐
       │                          │                          │
       ▼                          ▼                          ▼
┌──────────────┐           ┌──────────────┐           ┌──────────────┐
│ OpenAI BPE   │           │ Google 256k  │           │ Claude BPE   │
│ Exact Token  │           │ Calibrated   │           │ Calibrated   │
│ Count        │           │ Estimate     │           │ Estimate     │
└──────┬───────┘           └──────┬───────┘           └──────┬───────┘
       │                          │                          │
       └──────────────────────────┼──────────────────────────┘
                                  │
                                  ▼
                 ┌─────────────────────────────────┐
                 │     Cost Breakdown Calculator   │
                 │     - Separated input cost      │
                 │     - Estimated output cost     │
                 │     - Total estimated cost      │
                 └────────────────┬────────────────┘
                                  │
                                  ▼
                 ┌─────────────────────────────────┐
                 │   Architectural Suitability     │
                 │   - Task alignment              │
                 │   - Engineering/code aptitude   │
                 │   - Complexity tier fit         │
                 │   - Context headroom            │
                 └────────────────┬────────────────┘
                                  │
                                  ▼
                 ┌─────────────────────────────────┐
                 │ Multi-Criteria Ranking Engine   │
                 │ - Weighted Sum Model (WSM)      │
                 │ - TOPSIS Euclidean Distance     │
                 │ - Presets (Balanced, Cost,      │
                 │   Quality, Long-Context)        │
                 └────────────────┬────────────────┘
                                  │
                                  ▼
                 ┌─────────────────────────────────┐
                 │ Transparent Response & Rationale│
                 │ - Evidenced explanations        │
                 │ - Clear provenance declarations │
                 │ - No fabricated metrics         │
                 └─────────────────────────────────┘
```

### 1. Token Counting & Provenance Disclosures

| Provider | Adapter Engine | Accuracy | Provenance Disclosure |
|---|---|---|---|
| **OpenAI** (GPT-4o, GPT-4o-mini, GPT-4-turbo) | `tiktoken` (`o200k_base`, `cl100k_base`) | **100% Exact** | Exact local BPE count for prompt text. |
| **Google** (Gemini 1.5 Pro, 1.5 Flash, 2.0 Flash) | Calibrated SentencePiece heuristic | **Estimated** | Explicitly declared as calibrated estimate; official exact counts require remote Google API. |
| **Anthropic** (Claude 3.5 Sonnet, 3.5 Haiku, 3 Opus) | Calibrated BPE heuristic | **Estimated** | Explicitly declared as calibrated estimate; Anthropic provides no local open-source tokenizer. |

> **Production API Billing Note:** Raw prompt token counts do not reflect exact provider API billing. Final billing includes system instructions, tool declarations, chat framing formatting wrappers, and provider caching policies.

### 2. Model Registry & Verified Lifecycle Tracking

Model metadata is centralized in `analyzers/model_comparison/registry.py` with verified source documentation (February 2025):

- **Active Models (Default Recommendation Pool):**
  - `gpt-4o` (128k context, $2.50/$10.00 per M) — [OpenAI Models](https://platform.openai.com/docs/models#gpt-4o)
  - `gpt-4o-mini` (128k context, $0.15/$0.60 per M) — [OpenAI Pricing](https://platform.openai.com/docs/pricing)
  - `gemini-1.5-pro` (2M context, $1.25/$5.00 per M) — [Google Gemini](https://ai.google.dev/gemini-api/docs/models/gemini#gemini-1.5-pro)
  - `gemini-1.5-flash` (1M context, $0.075/$0.30 per M) — [Google Gemini](https://ai.google.dev/gemini-api/docs/models/gemini#gemini-1.5-flash)
  - `gemini-2.0-flash` (1M context, $0.10/$0.40 per M) — [Google Gemini](https://ai.google.dev/gemini-api/docs/models/gemini#gemini-2.0-flash)
  - `claude-3-5-sonnet` (200k context, $3.00/$15.00 per M) — [Anthropic Models](https://docs.anthropic.com/en/docs/about-claude/models#claude-3-5-sonnet)
  - `claude-3-5-haiku` (200k context, $0.80/$4.00 per M) — [Anthropic Pricing](https://www.anthropic.com/pricing)
  - `claude-3-opus` (200k context, $15.00/$75.00 per M) — [Anthropic Pricing](https://www.anthropic.com/pricing)
- **Deprecated / Retired Models (Excluded by default; available on request):**
  - `gpt-4-turbo` (Deprecated; superseded by GPT-4o) — [OpenAI Deprecations](https://platform.openai.com/docs/deprecations#gpt-4-turbo)
  - `gpt-3.5-turbo` (Retired; superseded by GPT-4o-mini) — [OpenAI Deprecations](https://platform.openai.com/docs/deprecations)

### 3. Multi-Criteria Ranking Algorithms

The platform implements two decision-making algorithms:

1. **Weighted Sum Model (WSM):**
   $$S_i = w_{\text{suit}} \cdot \bar{s}_i + w_{\text{cost}} \cdot \bar{c}_i + w_{\text{ctx}} \cdot \bar{x}_i$$
   - Normalized logarithmically across multi-order-of-magnitude pricing ($0.075/M to $75/M).
   - Cost is an inverted criterion (lower cost $\to$ higher score).
   - Suitability and context headroom are benefit criteria (higher $\to$ higher score).

2. **TOPSIS (Similarity to Ideal Solution):**
   $$C_i = \frac{D_i^-}{D_i^+ + D_i^-}$$
   - Measures Euclidean distance to the positive ideal solution ($D_i^+$) and negative ideal solution ($D_i^-$).
   - Handles degenerate zero-range and single-candidate cases cleanly.

#### Presets & Weight Allocations

| Preset | Suitability Weight | Cost Weight | Context Headroom Weight | Objective Focus |
|---|---|---|---|---|
| **`balanced`** | 0.45 | 0.35 | 0.20 | Balanced compromise between cost, quality, and context. |
| **`cost_optimized`** | 0.20 | 0.70 | 0.10 | Prioritizes economical models (e.g. Gemini 1.5 Flash, GPT-4o-mini). |
| **`quality_optimized`** | 0.80 | 0.05 | 0.15 | Prioritizes flagship capability tiers (Claude 3.5 Sonnet, GPT-4o). |
| **`long_context`** | 0.25 | 0.25 | 0.50 | Prioritizes massive context capacity (Gemini 1.5 Pro 2M window). |

### 4. What is NOT Measured in Offline Mode

PromptShield AI strictly enforces transparent provenance:
- **No Live Latency:** Network latency, time-to-first-token (TTFT), and token generation throughput are marked `unavailable` offline.
- **No Fabricated Benchmarks:** The platform does not invent or claim to measure third-party MMLU, SWE-bench, or HumanEval scores offline.
- **Relative Recommendation Scores:** The final ranking score is explicitly a multi-criteria decision trade-off index, not a probability of answer correctness.

---

# 🧪 Running Tests & API Server

### Running the FastAPI Backend

```bash
.venv/bin/uvicorn main:app --reload --port 8000
```

- Swagger UI Documentation: `http://localhost:8000/docs`
- Healthcheck: `GET http://localhost:8000/`
- List Eligible Models: `GET http://localhost:8000/models`
- Model Comparison: `POST http://localhost:8000/compare-models`

### Running the Pytest Suite

```bash
# Run all tests (unit, integration, benchmark, and model comparison)
.venv/bin/pytest -v

# Run only model comparison tests
.venv/bin/pytest tests/test_model_comparison.py -v
```

