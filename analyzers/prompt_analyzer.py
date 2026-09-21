"""
Prompt Analyzer Orchestrator for PromptShield AI.
Combines Prompt Security analysis with Application Risk assessment.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import joblib

from analyzers.prompt_security_analyzer import analyze_prompt_security

_APP_RISK_MODEL: Optional[Any] = None

APP_RISK_LABELS: List[str] = [
    "sqli",
    "xss",
    "ssrf",
    "idor",
    "auth_weakness",
    "authz_weakness",
    "file_upload",
    "path_traversal",
    "command_injection",
    "csrf",
    "sensitive_data",
    "prompt_injection",
    "data_poisoning",
    "model_extraction",
    "ai_tool_risk"
]


def get_app_risk_model_path() -> Path:
    """
    Returns path to existing baseline security_risk_model.pkl.
    """
    return Path(__file__).resolve().parent.parent / "security_risk_model.pkl"


def load_app_risk_model() -> Optional[Any]:
    """
    Safely loads the baseline Application Risk model if present.
    """
    global _APP_RISK_MODEL
    if _APP_RISK_MODEL is None:
        path = get_app_risk_model_path()
        if path.exists():
            try:
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    _APP_RISK_MODEL = joblib.load(path)
            except Exception as e:
                print(f"Warning: Could not load application risk model: {e}")
                return None
    return _APP_RISK_MODEL


def analyze_application_risks(prompt: str) -> Dict[str, Any]:
    """
    Infers application security risks using the baseline Application Risk Model.
    """
    if not prompt or not prompt.strip():
        return {
            "detected_risks": [],
            "risk_count": 0
        }

    model = load_app_risk_model()
    if model is None:
        return {
            "detected_risks": [],
            "risk_count": 0,
            "status": "baseline_model_unavailable"
        }

    try:
        prediction = model.predict([prompt])[0]
        detected = [
            label
            for label, value in zip(APP_RISK_LABELS, prediction)
            if value == 1
        ]
        return {
            "detected_risks": detected,
            "risk_count": len(detected)
        }
    except Exception as e:
        return {
            "detected_risks": [],
            "risk_count": 0,
            "error": str(e)
        }


def analyze_prompt(prompt: str) -> Dict[str, Any]:
    """
    Unified analysis endpoint handler.
    Runs Prompt Security Model and Application Risk Model concurrently.
    """
    prompt_security_result = analyze_prompt_security(prompt)
    app_risks_result = analyze_application_risks(prompt)

    return {
        "prompt": prompt,
        "task_type": "unknown",  # Preserved for backward compatibility
        "prompt_security": prompt_security_result,
        "application_risks": app_risks_result
    }