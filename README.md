# backendprompt


## Intelligent Prompt Analysis, Security Assessment and Multi-Model Optimization Platform

PromptShield AI is a security-focused platform designed to analyze prompts used with AI coding agents and generative AI systems.

The system aims to identify security risks that may arise from:

1. The prompt itself
2. The application or system that the prompt asks an AI coding agent to build
3. Sensitive information contained in the prompt

It will also provide prompt improvements, token/cost analysis, model comparison and multi-model optimization.

---

# Current Project Stage

## 🔄 CURRENT STAGE: Baseline Security Risk ML Model

We are currently building the **Application Risk Prediction Model**.

### Current progress

| Component | Status |
|---|---|
| FastAPI backend | ✅ Complete |
| `/analyze` API | ✅ Complete |
| Basic prompt analyzer | ✅ Complete |
| Public security dataset acquisition | ✅ Complete |
| Public dataset cleaning | ✅ Complete |
| Public dataset preprocessing | ✅ Complete |
| Contextual security-risk dataset | ✅ Complete for baseline |
| Dataset validation | ✅ Complete |
| Train/test split | ✅ Complete |
| TF-IDF setup | 🔄 Current |
| Multi-label Logistic Regression | 🔄 Current |
| Model evaluation | ⬜ Next |
| Probability calibration | ⬜ |
| Context extraction | ⬜ |
| Application Risk Engine | ⬜ |
| Prompt Security Model | ⬜ |
| Sensitive Data Detector | ⬜ |
| Prompt Optimizer | ⬜ |
| Token analysis | ⬜ |
| Cost estimation | ⬜ |
| GPT/Claude/Gemini integration | ⬜ |
| Model ranking | ⬜ |
| Frontend integration | ⬜ |

---

# Project Architecture

The final system is planned as:

```text
                         USER PROMPT
                              │
                              ▼
                  ┌─────────────────────┐
                  │ Prompt Understanding │
                  └──────────┬──────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
        Application       Features       Security
          Context         Extraction       Controls
              │              │              │
              └──────────────┼──────────────┘
                             ▼
                    ┌─────────────────┐
                    │ Security Engine │
                    └────────┬────────┘
                             │
                ┌────────────┴────────────┐
                ▼                         ▼
       Prompt Security Model      Application Risk Model
                │                         │
                ▼                         ▼
       Prompt Injection            SQL Injection
       Jailbreak                   XSS
       Secret Exposure             SSRF
       Malicious Instructions      IDOR
                                   Authentication
                                   Authorization
                                   File Upload
                                   Path Traversal
                                   Command Injection
                                   CSRF
                                   AI/ML Risks
                │                         │
                └────────────┬────────────┘
                             ▼
                  Sensitive Data Detector
                             │
                             ▼
                    Risk Aggregation
                             │
                             ▼
                 Probability Calibration
                             │
                             ▼
                Risk Score + Explanation
                             │
                             ▼
                  Security Recommendations
                             │
                             ▼
                    Prompt Optimization
