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
| Token analysis | ⬜ Planned |
| Cost estimation | ⬜ Planned |
| GPT/Claude/Gemini integration | ⬜ Planned |
| Model ranking | ⬜ Planned |
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
