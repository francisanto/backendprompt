"""
Prompt Security Analyzer Module for PromptShield AI.
Provides real-time inference for binary malicious prompt classification using
the Prompt Security Model artifact (TF-IDF + Logistic Regression).
"""

from pathlib import Path
from typing import Any, Dict, Optional
import joblib


_MODEL_ARTIFACT: Optional[Dict[str, Any]] = None


def get_model_path() -> Path:
    """
    Returns the absolute path to the saved Prompt Security Model artifact.
    """
    return Path(__file__).resolve().parent.parent / "models" / "prompt_security_model.pkl"


def load_security_model() -> Dict[str, Any]:
    """
    Loads and caches the Prompt Security Model artifact.
    """
    global _MODEL_ARTIFACT
    if _MODEL_ARTIFACT is None:
        model_path = get_model_path()
        if not model_path.exists():
            raise FileNotFoundError(
                f"Prompt Security Model not found at: {model_path}. "
                "Ensure 'python ml/train_prompt_security_model.py' has been executed."
            )
        _MODEL_ARTIFACT = joblib.load(model_path)
    return _MODEL_ARTIFACT


def analyze_prompt_security(prompt: str) -> Dict[str, Any]:
    """
    Analyzes whether an input prompt is benign or malicious.

    Returns:
        dict:
            - is_malicious (bool): True if malicious probability >= 0.5, False otherwise
            - malicious_probability (float): Raw model probability/confidence for malicious class
            - benign_probability (float): Raw model probability/confidence for benign class
    """
    if not isinstance(prompt, str) or not prompt.strip():
        return {
            "is_malicious": False,
            "malicious_probability": 0.0,
            "benign_probability": 1.0,
            "warning": "Empty prompt received"
        }

    try:
        artifact = load_security_model()
        pipeline = artifact.get("pipeline")

        if pipeline is not None:
            proba = pipeline.predict_proba([prompt])[0]
        else:
            vectorizer = artifact["vectorizer"]
            classifier = artifact["classifier"]
            features = vectorizer.transform([prompt])
            proba = classifier.predict_proba(features)[0]

        benign_proba = float(proba[0])
        malicious_proba = float(proba[1])

        return {
            "is_malicious": bool(malicious_proba >= 0.5),
            "malicious_probability": round(malicious_proba, 4),
            "benign_probability": round(benign_proba, 4)
        }
    except Exception as e:
        return {
            "is_malicious": False,
            "malicious_probability": 0.0,
            "benign_probability": 0.0,
            "error": f"Prompt security analysis error: {str(e)}"
        }
