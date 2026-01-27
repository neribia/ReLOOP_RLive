from pydantic import BaseModel, Field, ConfigDict


class AttachHardwareRequest(BaseModel):
    """Request body for POST /connect."""
    camera_type: str | None = None
    robot_name: str | None = None
    use_dummy: bool = False


class DetachHardwareRequest(BaseModel):
    """Request body for POST /disconnect."""


class BaseRequest(BaseModel):
    """Basic request body for POST endpoints."""

    model_config = ConfigDict(arbitrary_types_allowed=True)


class ResetRequest(BaseRequest):
    """Request body for POST /reset."""


class StepRequest(BaseRequest):
    """Request body for POST /step."""

    action: int = Field(..., description="Discrete action ID to execute in the world.")
