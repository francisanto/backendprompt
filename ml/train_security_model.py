import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

data = pd.read_csv("data/security_risk_dataset.csv")

print("Dataset loaded successfully")
print("Number of examples:", len(data))
print(data.head())
risk_labels = [
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
X = data["prompt"]
y = data[risk_labels]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
print("\nTraining label counts:")
print(y_train.sum().sort_values())

print("\nTesting label counts:")
print(y_test.sum().sort_values())

print("Training examples:", len(X_train))
print("Testing examples:", len(X_test))
risk_labels = [
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

X = data["prompt"]
y = data[risk_labels]

print("\nInput shape:", X.shape)
print("Output shape:", y.shape)
# ---------------------------------------------------------
# Build the baseline multi-label security model
# ---------------------------------------------------------

model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            max_features=5000
        )
    ),
    (
        "classifier",
        OneVsRestClassifier(
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced"
            )
        )
    )
])

print("\nTraining model...")

model.fit(X_train, y_train)

print("Model training completed.")
# ---------------------------------------------------------
# Evaluate the model
# ---------------------------------------------------------

print("\nEvaluating model...")

y_pred = model.predict(X_test)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=risk_labels,
        zero_division=0
    )
)