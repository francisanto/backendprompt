import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report


# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------

data = pd.read_csv("data/security_risk_dataset.csv")

print("Dataset loaded successfully")
print("Number of examples:", len(data))
print(data.head())


# ---------------------------------------------------------
# Define security risk labels
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# Prepare input and output
# ---------------------------------------------------------

X = data["prompt"]
y = data[risk_labels]

print("\nInput shape:", X.shape)
print("Output shape:", y.shape)


# ---------------------------------------------------------
# Split dataset
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# Build baseline multi-label security model
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


# ---------------------------------------------------------
# Train model
# ---------------------------------------------------------

print("\nTraining model...")

model.fit(X_train, y_train)

print("Model training completed.")


# ---------------------------------------------------------
# Evaluate model
# ---------------------------------------------------------

print("\nEvaluating model...")

y_pred = model.predict(X_test)

print("\nIncorrect Predictions:")

for i in range(len(X_test)):
    actual = y_test.iloc[i].values
    predicted = y_pred[i]

    if not (actual == predicted).all():
        print("\nPrompt:")
        print(X_test.iloc[i])

        print("Actual risks:")
        print([
            label
            for label, value in zip(risk_labels, actual)
            if value == 1
        ])

        print("Predicted risks:")
        print([
            label
            for label, value in zip(risk_labels, predicted)
            if value == 1
        ])
        
print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=risk_labels,
        zero_division=0
    )
)


# ---------------------------------------------------------
# Save trained model
# ---------------------------------------------------------

MODEL_PATH = "security_risk_model.pkl"

joblib.dump(model, MODEL_PATH)

print(
    f"\nModel saved successfully to: "
    f"{os.path.abspath(MODEL_PATH)}"
)