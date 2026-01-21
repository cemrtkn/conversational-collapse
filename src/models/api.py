"""Pydantic models for LLM API requests and responses."""

from typing import Any, Dict, List

from pydantic import BaseModel, Field, PositiveInt


class LLMResponse(BaseModel):
    """Response model for LLM API calls."""

    content: str = Field(..., description="The generated text response")
    input_token_count: PositiveInt = Field(
        ..., description="Number of tokens in the input"
    )
    output_token_count: PositiveInt = Field(
        ..., description="Number of tokens in the output"
    )
    intervention_output: Dict[str, List[Any]] = Field(
        ..., description="Output of the intervention"
    )
