from fastapi import FastAPI
from models.schemas import PromptRequest

app = FastAPI(title="PromptShield AI")


@app.get("/")
def home():
    return {"message": "PromptShield AI backend is running"}


@app.post("/analyze")
def analyze_prompt(data: PromptRequest):
    return {
        "prompt": data.prompt
    }