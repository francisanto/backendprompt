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
| Basic prompt analyzer | ✅ Complete |
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
| Prompt Security Model | 🔄 Current |
| Prompt Security model training | 🔄 Current |
| Prompt Security evaluation | ⬜ Next |
| Prompt Security testing | ⬜ Next |
| Probability calibration | ⬜ |
| Context extraction | ⬜ |
| Improved Application Risk Engine | ⬜ |
| 30-risk Application Risk taxonomy | 🔄 Planned |
| Sensitive Data Detector | ⬜ |
| Risk aggregation | ⬜ |
| Security explanations | ⬜ |
| Security recommendations | ⬜ |
| Prompt Optimizer | ⬜ |
| Token analysis | ⬜ |
| Cost estimation | ⬜ |
| GPT/Claude/Gemini integration | ⬜ |
| Model ranking | ⬜ |
| Frontend integration | ⬜ |

---

# Current ML Architecture

PromptShield uses two separate security prediction problems.

```text
                         USER PROMPT
                              │
               ┌──────────────┴──────────────┐
               │                             │
               ▼                             ▼
      Prompt Security Model        Application Risk Model
               │                             │
               ▼                             ▼
        "Is the prompt              "What risks could the
         malicious?"                 requested application
                                    potentially introduce?"
               │                             │
               ▼                             ▼
          MPDD Dataset             Contextual Security Dataset
               │                             │
               ▼                             ▼
       Binary Classification       Multi-Label Classification
               │                             │
               ▼                             ▼
       Benign / Malicious          Multiple Security Risks
