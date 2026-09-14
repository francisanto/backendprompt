import joblib


# Load trained model
model = joblib.load("security_risk_model.pkl")


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


# Get prompt from user
prompt = input("\nEnter a prompt to analyze:\n> ")


# Predict risks
prediction = model.predict([prompt])[0]

print("\nPotential Security Risks:")

found = False

for label, value in zip(risk_labels, prediction):
    if value == 1:
        print("-", label)
        found = True

if not found:
    print("No significant risks detected by the baseline model.")