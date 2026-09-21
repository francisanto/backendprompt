"""
Prompt Security Model Training & Evaluation Pipeline.
Trains a binary classifier (TF-IDF + Logistic Regression) on the cleaned MPDD dataset
to classify prompts as benign (0) or malicious (1).
"""

import json
from datetime import datetime, timezone
from pathlib import Path
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


def train_prompt_security_model(
    data_path: Path,
    model_output_path: Path,
    metrics_output_path: Path,
    random_state: int = 42,
    test_size: float = 0.2
):
    """
    Loads cleaned MPDD data, splits with stratification, trains TF-IDF + Logistic Regression pipeline,
    computes detailed security metrics (including False Negatives and ROC-AUC), and persists model & metrics.
    """
    print("=" * 60)
    print("PROMPT SECURITY MODEL TRAINING")
    print("=" * 60)

    if not data_path.exists():
        raise FileNotFoundError(f"Cleaned dataset not found at: {data_path.resolve()}")

    print(f"Loading cleaned dataset: {data_path.resolve()}")
    df = pd.read_csv(data_path)

    prompt_col = "Prompt"
    label_col = "isMalicious"

    X = df[prompt_col].astype(str)
    y = df[label_col].astype(int)

    print(f"Total dataset records: {len(df):,}")
    print(f"Label distribution:\n{y.value_counts().to_dict()}\n")

    # 1. Stratified Train/Test Split
    print(f"Splitting data: {int((1-test_size)*100)}% Train / {int(test_size*100)}% Test (stratified, random_state={random_state})")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    print(f"Training samples: {len(X_train):,} ({y_train.value_counts().to_dict()})")
    print(f"Testing samples:  {len(X_test):,} ({y_test.value_counts().to_dict()})\n")

    # 2. Build Pipeline
    print("Configuring architecture: TfidfVectorizer -> LogisticRegression (balanced)")
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        max_features=10000,
        sublinear_tf=True
    )

    classifier = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=random_state
    )

    pipeline = Pipeline([
        ("tfidf", vectorizer),
        ("classifier", classifier)
    ])

    # 3. Train Model
    print("Fitting model...")
    pipeline.fit(X_train, y_train)
    print("Training complete.\n")

    # 4. Evaluation
    print("Evaluating on test set...")
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)
    # Probability of class 1 (malicious)
    y_proba_malicious = y_proba[:, 1]

    accuracy = float(accuracy_score(y_test, y_pred))
    precision = float(precision_score(y_test, y_pred, zero_division=0))
    recall = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    roc_auc = float(roc_auc_score(y_test, y_proba_malicious))

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]

    report_dict = classification_report(
        y_test,
        y_pred,
        target_names=["benign", "malicious"],
        output_dict=True,
        zero_division=0
    )
    report_text = classification_report(
        y_test,
        y_pred,
        target_names=["benign", "malicious"],
        zero_division=0
    )

    # Security focus: False Negative Rate
    total_malicious_test = int((y_test == 1).sum())
    fn_rate = float(fn / total_malicious_test) if total_malicious_test > 0 else 0.0

    print("=" * 60)
    print("MODEL EVALUATION RESULTS")
    print("=" * 60)
    print(f"Accuracy:  {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall:    {recall * 100:.2f}%")
    print(f"F1-Score:  {f1 * 100:.2f}%")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print("\nConfusion Matrix:")
    print(f"  True Negatives (TN):  {tn:,} (Benign correctly identified)")
    print(f"  False Positives (FP): {fp:,} (Benign flagged as Malicious)")
    print(f"  False Negatives (FN): {fn:,} (Malicious missed as Benign - Critical Security Risk)")
    print(f"  True Positives (TP):  {tp:,} (Malicious correctly identified)")
    print(f"\nFalse Negative Rate: {fn_rate * 100:.2f}% ({fn:,} of {total_malicious_test:,} malicious prompts missed)")
    print("\nClassification Report:\n")
    print(report_text)

    # 5. Save Metrics
    metrics_data = {
        "model_name": "Prompt Security Model",
        "evaluation_date": datetime.now(timezone.utc).isoformat(),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "metrics": {
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(roc_auc, 4)
        },
        "confusion_matrix": {
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "true_positives": tp,
            "false_negative_rate": round(fn_rate, 4)
        },
        "classification_report": report_dict,
        "scientific_notes": {
            "model_type": "Binary malicious-prompt classification based on MPDD training distribution",
            "calibration_status": "Uncalibrated raw logistic regression model confidence (calibration planned)",
            "security_focus": "False Negatives prioritized for security assessment"
        }
    }

    metrics_output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_output_path, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2)
    print(f"Metrics saved to: {metrics_output_path.resolve()}")

    # 6. Save Model Artifact
    model_artifact = {
        "vectorizer": pipeline.named_steps["tfidf"],
        "classifier": pipeline.named_steps["classifier"],
        "pipeline": pipeline,
        "label_mapping": {0: "benign", 1: "malicious"},
        "metadata": {
            "model_name": "Prompt Security Model",
            "version": "1.0.0",
            "algorithm": "TF-IDF + LogisticRegression",
            "description": "Binary malicious-prompt classification based on the MPDD training distribution",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "features_max": 10000,
            "metrics": metrics_data["metrics"]
        }
    }

    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_artifact, model_output_path)
    print(f"Model artifact saved to: {model_output_path.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    data_file = base_dir / "data" / "processed" / "mpdd_clean.csv"
    model_file = base_dir / "models" / "prompt_security_model.pkl"
    results_file = base_dir / "ml" / "results" / "prompt_security_metrics.json"

    train_prompt_security_model(data_file, model_file, results_file)
