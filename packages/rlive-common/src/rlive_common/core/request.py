from pydantic import BaseModel, Field, ConfigDict

from rlive_common.core.types import NumpyArray
from rlive_common.core.hardware_config import WorldConfig

class AttachHardwareRequest(BaseModel):
    """Request body for POST /attach_hardware.

    Attributes:
        world_config: World configuration with camera and bolt settings.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    world_config: WorldConfig = Field(default_factory=WorldConfig, description="World configuration")


class DetachHardwareRequest(BaseModel):
    """Request body for POST /disconnect."""


class BaseRequest(BaseModel):
    """Basic request body for POST endpoints."""

    model_config = ConfigDict(arbitrary_types_allowed=True)


class ResetRequest(BaseRequest):
    """Request body for POST /reset."""

    actions: list[NumpyArray] = Field(default_factory=list, description="List of actions to execute during reset in the environment, where each action is represented as a NumPy array.")


class StepRequest(BaseRequest):
    """Request body for POST /step."""

    action: NumpyArray = Field(..., description="Action to take in the environment, represented as a NumPy array. The specific shape and meaning of the array depend on the environment's action space.")
