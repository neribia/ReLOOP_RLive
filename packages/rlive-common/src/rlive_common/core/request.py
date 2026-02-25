from pydantic import BaseModel, Field, ConfigDict

from rlive_common.core.types import NumpyArray

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

    actions: list[NumpyArray] = Field(..., description="List of actions to execute during reset in the environment, where each action is represented as a NumPy array. The specific shape and meaning of each array depend on the environment's action space.")


class StepRequest(BaseRequest):
    """Request body for POST /step."""

    action: NumpyArray = Field(..., description="Action to take in the environment, represented as a NumPy array. The specific shape and meaning of the array depend on the environment's action space.")
