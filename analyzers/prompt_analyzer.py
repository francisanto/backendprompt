"""
Prompt Analyzer Orchestrator for PromptShield AI.
Combines Prompt Security analysis, Application Risk assessment,
and lightweight Sensitive Data detection.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import joblib

from analyzers.prompt_security_analyzer import analyze_prompt_security
from analyzers.sensitive_data_detector import detect_sensitive_data

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
    Returns the absolute path to the Application Risk Model artifact.
    """
    return Path(__file__).resolve().parent.parent / "models" / "security_risk_model.pkl"


def load_app_risk_model() -> Any:
    """
    Loads and caches the baseline Application Risk model.
    Raises FileNotFoundError if the model artifact is missing.
    """
    global _APP_RISK_MODEL
    if _APP_RISK_MODEL is None:
        path = get_app_risk_model_path()
        if not path.exists():
            raise FileNotFoundError(
                f"Application Risk Model artifact not found at: {path}. "
                "Train the model using '.venv/bin/python ml/train_security_model.py'."
            )
        _APP_RISK_MODEL = joblib.load(path)
    return _APP_RISK_MODEL


def analyze_application_risks(prompt: str) -> Dict[str, Any]:
    """
    Infers application security risks using the Application Risk Model.

    Returns:
        dict:
            - status (str): 'success', 'model_unavailable', or 'error'
            - detected_risks (list or None): List of predicted risk labels, or None if unavailable/error
            - risk_count (int or None): Count of detected risks, or None if unavailable/error
            - error (str, optional): Explanatory error message on failure
            - disclaimer (str): Contextual note on machine learning predictions
    """
    disclaimer = (
        "Statistical multi-label prediction based on application description patterns. "
        "These are model inferences, not proof of actual system vulnerabilities."
    )

    if not isinstance(prompt, str) or not prompt.strip():
        return {
            "status": "error",
            "detected_risks": None,
            "risk_count": None,
            "error": "Prompt cannot be empty or contain only whitespace.",
            "disclaimer": disclaimer
        }

    try:
        model = load_app_risk_model()
        pipeline = model if hasattr(model, "predict") else model.get("pipeline")
        if pipeline is None:
            raise ValueError("Loaded model artifact does not contain a valid predictor or pipeline.")

        prediction = pipeline.predict([prompt])[0]
        detected = [
            label
            for label, value in zip(APP_RISK_LABELS, prediction)
            if value == 1
        ]
        return {
            "status": "success",
            "detected_risks": detected,
            "risk_count": len(detected),
            "disclaimer": disclaimer
        }
    except FileNotFoundError as e:
        return {
            "status": "model_unavailable",
            "detected_risks": None,
            "risk_count": None,
            "error": str(e),
            "disclaimer": disclaimer
        }
    except Exception as e:
        return {
            "status": "error",
            "detected_risks": None,
            "risk_count": None,
            "error": f"Application risk inference error: {str(e)}",
            "disclaimer": disclaimer
        }


def analyze_prompt(prompt: str, analysis_mode: str = "both") -> Dict[str, Any]:
    """
    Unified analysis endpoint handler.
    Supports modes: 'prompt_security', 'application_risk', 'both'.

    Returns separate sections for:
    1. Prompt Security Model output
    2. Application Risk Model output
    3. Sensitive Data detector findings
    4. Analysis status and model availability summary
    """
    mode = (analysis_mode or "both").lower().strip()
    if mode not in {"prompt_security", "application_risk", "both"}:
        mode = "both"

    result: Dict[str, Any] = {
        "prompt": prompt,
        "analysis_mode": mode,
        "task_type": "unknown",  # Preserved for backward compatibility
    }

    if not isinstance(prompt, str) or not prompt.strip():
        result.update({
            "status": "error",
            "error": "Prompt cannot be empty or contain only whitespace.",
            "prompt_security": None,
            "application_risks": None,
            "sensitive_data": None
        })
        return result

    # 1. Prompt Security Analysis
    if mode in {"prompt_security", "both"}:
        prompt_sec_result = analyze_prompt_security(prompt)
    else:
        prompt_sec_result = {
            "status": "skipped",
            "message": f"Excluded by analysis_mode '{mode}'"
        }

    # 2. Application Risk Analysis
    if mode in {"application_risk", "both"}:
        app_risk_result = analyze_application_risks(prompt)
    else:
        app_risk_result = {
            "status": "skipped",
            "message": f"Excluded by analysis_mode '{mode}'"
        }

    # 3. Sensitive Data Detection
    sensitive_data_result = detect_sensitive_data(prompt)

    # 4. Overall status evaluation
    errors: List[str] = []
    if prompt_sec_result.get("status") in {"model_unavailable", "error"}:
        errors.append(f"Prompt Security: {prompt_sec_result.get('error')}")
    if app_risk_result.get("status") in {"model_unavailable", "error"}:
        errors.append(f"Application Risk: {app_risk_result.get('error')}")

    if not errors:
        overall_status = "success"
    elif len(errors) == (2 if mode == "both" else 1):
        overall_status = "error"
    else:
        overall_status = "partial_success"

    result.update({
        "status": overall_status,
        "model_availability": {
            "prompt_security_model": prompt_sec_result.get("status") == "success",
            "application_risk_model": app_risk_result.get("status") == "success"
        },
        "errors": errors if errors else None,
        "prompt_security": prompt_sec_result,
        "application_risks": app_risk_result,
        "sensitive_data": sensitive_data_result
    })

    return result