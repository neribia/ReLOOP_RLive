from pydantic import BaseModel, Field, ConfigDict


class Request(BaseModel):
    """Basic request body for POST endpoints."""

    model_config = ConfigDict(arbitrary_types_allowed=True)


class ResetRequest(Request):
    """Request body for POST /reset."""


class StepRequest(Request):
    """Request body for POST /step."""

    action: int = Field(..., description="Discrete action ID to execute in the world.")
