from pydantic import BaseModel, Field


class LLMAnswer(BaseModel):
    answer: str = Field(..., description="The LLM's answer.")


class Document(BaseModel):
    path: str = Field(..., description="Path to document.")
    text: str = Field(..., description="Document text.")
