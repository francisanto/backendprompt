from fastapi import FastAPI

from models.schemas import PromptRequest
from analyzers.prompt_analyzer import analyze_prompt

app = FastAPI(
    title="PromptShield AI",
    description="Multi-model AI security analysis backend evaluating prompt injection/malice and application-level security risks.",
    version="1.0.0"
)


@app.get("/")
def home():
    return {"message": "PromptShield AI backend is running"}


@app.post("/analyze")
def analyze(data: PromptRequest):
    mode = data.analysis_mode.value if hasattr(data.analysis_mode, "value") else str(data.analysis_mode)
    return analyze_prompt(data.prompt, analysis_mode=mode)