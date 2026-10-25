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

### 2. Multi-Category AI Catalogue & Verified Lifecycle Policy

PromptShield AI organizes AI systems into three distinct categories with strict category isolation:

#### Category A: Foundation Models (`foundation_model`)
Centralized in `analyzers/model_comparison/registry.py` with verified provider documentation (Feb 2025):

- **Active Models (Default Recommendation Pool):**
  - `gpt-4o` (128k context, $2.50/$10.00 per M) — [OpenAI Models](https://platform.openai.com/docs/models#gpt-4o)
  - `gpt-4o-mini` (128k context, $0.15/$0.60 per M) — [OpenAI Pricing](https://openai.com/api/pricing)
  - `o1` (200k context, $15.00/$60.00 per M) — [OpenAI o1](https://platform.openai.com/docs/models#o1)
  - `o3-mini` (200k context, $1.10/$4.40 per M) — [OpenAI o3-mini](https://platform.openai.com/docs/models#o3-mini)
  - `gemini-2.0-flash` (1M context, $0.10/$0.40 per M) — [Google Gemini 2.0 Flash](https://ai.google.dev/gemini-api/docs/models/gemini#gemini-2.0-flash)
  - `gemini-2.5-pro` (2M context, $1.25/$5.00 per M) — [Google Gemini 2.5 Pro](https://ai.google.dev/gemini-api/docs/models#gemini-2.5-pro)
  - `gemini-2.5-flash` (1M context, $0.10/$0.40 per M) — [Google Gemini 2.5 Flash](https://ai.google.dev/gemini-api/docs/models#gemini-2.5-flash)
  - `claude-3-7-sonnet` (200k context, $3.00/$15.00 per M) — [Anthropic Claude 3.7 Sonnet](https://docs.anthropic.com/en/docs/about-claude/models#claude-3-7-sonnet)
  - `claude-3-5-haiku` (200k context, $0.80/$4.00 per M) — [Anthropic Pricing](https://www.anthropic.com/pricing)
  - `grok-3` (131k context, $3.00/$15.00 per M) — [xAI Grok](https://docs.x.ai/)
  - `grok-3-mini` (131k context, $0.30/$0.50 per M) — [xAI Grok](https://docs.x.ai/)

- **Deprecated / Retired Models (Excluded by default; accessible via `include_deprecated=True` or direct ID):**
  - `gemini-1.5-pro` (Retired Sep 2025; replaced by Gemini 2.5 Pro) — [Google Deprecations](https://ai.google.dev/gemini-api/docs/deprecations)
  - `gemini-1.5-flash` (Retired Sep 2025; replaced by Gemini 2.5 Flash) — [Google Deprecations](https://ai.google.dev/gemini-api/docs/deprecations)
  - `claude-3-5-sonnet` (Deprecated Aug 2025; replaced by Claude 3.7 Sonnet) — [Anthropic Deprecations](https://platform.claude.com/docs/en/docs/about-claude/model-deprecations)
  - `claude-3-opus` (Retired Jan 2026; replaced by Claude 3.7 Sonnet) — [Anthropic Deprecations](https://platform.claude.com/docs/en/docs/about-claude/model-deprecations)
  - `gpt-4-turbo` (Deprecated; replaced by GPT-4o) — [OpenAI Deprecations](https://platform.openai.com/docs/deprecations#gpt-4-turbo)
  - `gpt-3.5-turbo` (Retired; replaced by GPT-4o-mini) — [OpenAI Deprecations](https://platform.openai.com/docs/deprecations)
  - `grok-2` (Retired; replaced by Grok 3) — [xAI Deprecations](https://docs.x.ai/)

#### Category B: IDEs & Coding Assistants (`ide_assistant`)
Evaluated on developer environment workflows, multi-file agent autonomy, and repository understanding:
- **`cursor`** (Anysphere) — VS Code fork featuring Composer, local vector indexing, and Claude 3.7/GPT-4o selection. ($0 / $20/mo / $40/mo).
- **`windsurf`** (Codeium) — Continuous flow IDE with Cascade Fast Context routing, multi-file inline edits, and terminal execution. ($0 / $20/mo / $40/mo).
- **`github-copilot`** (GitHub / Microsoft) — Deepest editor ecosystem (VS Code, JetBrains, Visual Studio, Xcode), Copilot Workspace, PR review. ($0 / $10/mo / $19/mo / $39/mo).
- **`google-antigravity`** (Google) — Agent-first platform with autonomous multi-surface agents, asynchronous agent manager, verifiable artifacts, and browser subagents.

#### Category C: AI App Builders (`app_builder`)
Evaluated on full-stack scaffolding, UI generation, database/auth support, and code ownership:
- **`v0-dev`** (Vercel) — UI component generator outputting production-grade React, Next.js, and shadcn/ui. Frontend-focused ($0 / $20/mo).
- **`bolt-new`** (StackBlitz) — In-browser full-stack prototyping running Node.js in WebContainers with interactive Plan & Build modes ($0 / $20/mo).
- **`lovable-dev`** (Lovable) — Full-stack web MVP builder tightly integrated with Supabase PostgreSQL, Auth, and two-way GitHub sync ($0 / $20/mo / $50/mo).
- **`replit-agent`** (Replit) — Autonomous cloud development platform with full Linux backend execution, multi-language support (Python, Node, Go), and Postgres ($0 / $20/mo / $95/mo).

### 3. Multi-Criteria Ranking Algorithms

The platform implements two decision-making algorithms:

1. **Weighted Sum Model (WSM):**
   $$S_i = \sum_{j} w_j \cdot \bar{x}_{ij}$$
   - Normalized logarithmically across multi-order-of-magnitude foundation model pricing ($0.075/M to $75/M).
   - Evaluates category-specific benefit and cost criteria.

2. **TOPSIS (Similarity to Ideal Solution):**
   $$C_i = \frac{D_i^-}{D_i^+ + D_i^-}$$
   - Measures Euclidean distance to positive ideal solution ($D_i^+$) and negative ideal solution ($D_i^-$).
   - Handles degenerate zero-range and single-candidate cases cleanly.

#### Presets & Weight Allocations by Category

**Category A: Foundation Models**
| Preset | Suitability | Cost | Context Headroom | Objective Focus |
|---|---|---|---|---|
| **`balanced`** | 0.45 | 0.35 | 0.20 | Balanced compromise between cost, quality, and context. |
| **`cost_optimized`** | 0.20 | 0.70 | 0.10 | Prioritizes economical models (e.g. Gemini 2.0 Flash, GPT-4o-mini). |
| **`quality_optimized`** | 0.80 | 0.05 | 0.15 | Prioritizes frontier capability tiers (Claude 3.7 Sonnet, o1, o3-mini, GPT-4o). |
| **`long_context`** | 0.25 | 0.25 | 0.50 | Prioritizes massive context capacity (Gemini 2.5 Pro 2M window). |

**Category B: IDE Assistants**
| Preset | Multi-File Agent | Repo Indexing | Terminal Exec | Model Flex | Ecosystem | Privacy | Affordability |
|---|---|---|---|---|---|---|---|
| **`balanced`** | 0.20 | 0.15 | 0.15 | 0.15 | 0.15 | 0.10 | 0.10 |
| **`agentic_power`** | 0.35 | 0.20 | 0.25 | 0.10 | 0.05 | 0.02 | 0.03 |
| **`ecosystem_integration`** | 0.10 | 0.10 | 0.03 | 0.25 | 0.35 | 0.02 | 0.15 |
| **`privacy_enterprise`** | 0.10 | 0.20 | 0.05 | 0.05 | 0.15 | 0.40 | 0.05 |

**Category C: AI App Builders**
| Preset | Full-Stack Backend | Frontend UI Quality | Deployment/Hosting | Code Export/GitHub | Autonomous Iteration | Pricing Accessibility |
|---|---|---|---|---|---|---|
| **`balanced`** | 0.20 | 0.20 | 0.15 | 0.20 | 0.15 | 0.10 |
| **`full_stack`** | 0.40 | 0.05 | 0.15 | 0.15 | 0.20 | 0.05 |
| **`frontend_ui`** | 0.05 | 0.45 | 0.20 | 0.05 | 0.10 | 0.15 |
| **`code_ownership`** | 0.20 | 0.05 | 0.10 | 0.45 | 0.15 | 0.05 |

### 4. What is NOT Measured in Offline Mode

PromptShield AI strictly enforces transparent provenance:
- **No Live Latency:** Network latency, time-to-first-token (TTFT), and token generation throughput are marked `unavailable` offline.
- **No Fabricated Benchmarks:** The platform does not invent or claim to measure third-party MMLU, SWE-bench, or HumanEval scores offline.
- **Relative Recommendation Scores:** The final ranking score is explicitly a multi-criteria decision trade-off index, not a probability of answer correctness.
- **No Product Token Costs:** IDEs and App Builders disclose their real subscription and credit structures ($0 free, $10-$20/mo pro) rather than fabricated token costs.

---

# 🧪 Running Tests & API Server

### Running the FastAPI Backend

```bash
.venv/bin/uvicorn main:app --reload --port 8000
```

- Swagger UI Documentation: `http://localhost:8000/docs`
- Healthcheck: `GET http://localhost:8000/`
- List Eligible Models: `GET http://localhost:8000/models`
- List IDE Assistants: `GET http://localhost:8000/models?category=ide_assistant`
- List App Builders: `GET http://localhost:8000/models?category=app_builder`
- Multi-Category Comparison: `POST http://localhost:8000/compare-models`

### Running the Pytest Suite

```bash
# Run all tests (unit, integration, benchmark, and model comparison)
.venv/bin/pytest -v

# Run only model comparison tests
.venv/bin/pytest tests/test_model_comparison.py -v
```

