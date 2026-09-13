from fastapi import FastAPI

from models.schemas import PromptRequest
from analyzers.prompt_analyzer import analyze_prompt

app = FastAPI(title="PromptShield AI")


@app.get("/")
def home():
    return {"message": "PromptShield AI backend is running"}


@app.post("/analyze")
def analyze(data: PromptRequest):
    return analyze_prompt(data.prompt)