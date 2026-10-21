from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, field_validator


class AnalysisMode(str, Enum):
    PROMPT_SECURITY = "prompt_security"
    APPLICATION_RISK = "application_risk"
    BOTH = "both"


class PromptRequest(BaseModel):
    prompt: str = Field(
        ...,
        description="The prompt text to analyze"
    )
    analysis_mode: Optional[AnalysisMode] = Field(
        default=AnalysisMode.BOTH,
        description="Analysis mode: 'prompt_security', 'application_risk', or 'both'"
    )

    @field_validator("prompt")
    @classmethod
    def check_prompt(cls, v: str) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("Prompt cannot be empty or contain only whitespace.")
        return v