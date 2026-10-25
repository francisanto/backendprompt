"""
PromptShield AI — Local Task Classifier Interface.
=================================================
Loads the trained local task classification pipeline from `models/task_classifier_model.pkl`
to predict task domains:
- code_generation
- security_analysis
- data_analysis
- creative_and_writing
- reasoning_and_math
- general_instruction

Includes confidence scores, probability distribution, and fallback heuristics.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import joblib


@dataclass
class TaskClassificationResult:
    predicted_category: str
    confidence: float
    category_probabilities: Dict[str, float]
    model_source: str  # "local_ml_model" or "rule_heuristic_fallback"


class LocalTaskClassifier:
    """Predicts prompt task domain using local TF-IDF + LogisticRegression model."""

    def __init__(self, model_path: Optional[Path] = None):
        if model_path is None:
            repo_root = Path(__file__).resolve().parent.parent.parent
            self.model_path = repo_root / "models" / "task_classifier_model.pkl"
        else:
            self.model_path = model_path
        self._artifact = None

    def _load_model(self):
        if self._artifact is None:
            if self.model_path.exists():
                try:
                    self._artifact = joblib.load(self.model_path)
                except Exception:
                    self._artifact = None
        return self._artifact

    def classify_task(self, prompt: str) -> TaskClassificationResult:
        if not prompt or not prompt.strip():
            return TaskClassificationResult(
                predicted_category="general_instruction",
                confidence=0.5,
                category_probabilities={"general_instruction": 1.0},
                model_source="rule_heuristic_fallback"
            )

        artifact = self._load_model()
        if artifact and "pipeline" in artifact:
            pipeline = artifact["pipeline"]
            classes = artifact["classes"]

            probas = pipeline.predict_proba([prompt])[0]
            pred_idx = probas.argmax()
            predicted_class = classes[pred_idx]
            confidence = float(probas[pred_idx])

            prob_dict = {
                cls_name: round(float(prob), 4)
                for cls_name, prob in zip(classes, probas)
            }

            return TaskClassificationResult(
                predicted_category=predicted_class,
                confidence=round(confidence, 3),
                category_probabilities=prob_dict,
                model_source="local_ml_model"
            )

        # Fallback heuristic if artifact missing
        return self._rule_fallback(prompt)

    def _rule_fallback(self, prompt: str) -> TaskClassificationResult:
        low = prompt.lower()
        if any(k in low for k in ["def ", "class ", "function", "import ", "sql", "react", "api", "docker", "endpoint"]):
            cat = "code_generation"
        elif any(k in low for k in ["vulnerability", "exploit", "security", "xss", "sqli", "bypass", "injection", "threat"]):
            cat = "security_analysis"
        elif any(k in low for k in ["pandas", "dataset", "dataframe", "correlation", "regression", "clustering", "eda"]):
            cat = "data_analysis"
        elif any(k in low for k in ["poem", "story", "novel", "metaphor", "creative", "fiction", "character"]):
            cat = "creative_and_writing"
        elif any(k in low for k in ["integral", "derivative", "matrix", "probability", "theorem", "proof", "solve equation"]):
            cat = "reasoning_and_math"
        else:
            cat = "general_instruction"

        return TaskClassificationResult(
            predicted_category=cat,
            confidence=0.60,
            category_probabilities={cat: 0.60, "general_instruction": 0.40},
            model_source="rule_heuristic_fallback"
        )


# Singleton instance
task_classifier = LocalTaskClassifier()
