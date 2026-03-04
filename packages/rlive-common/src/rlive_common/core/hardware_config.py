"""Hardware configuration model for world setup."""

from typing import Any
from pydantic import BaseModel, Field, field_validator, ConfigDict

from rlive_common.core.enums import CameraType, CameraResolution


class WorldConfig(BaseModel):
    """Unified configuration for the entire world setup.

    Combines camera and robot (Bolt) settings into a single model for convenient
    initialization of the world environment. All None values use rlive-world defaults.

    Attributes:
        # Camera settings
        camera_type: Camera type (DUMMY, WEBCAM, PICAM). None uses env default.
        camera_id: Camera device ID. None uses env default.
        camera_resolution: Frame resolution as (width, height) tuple. Accepts CameraResolution enum
                          or custom tuple/list, but always stored and sent as tuple. None uses env default.
        camera_exposure_time_ms: Camera exposure time in milliseconds. None uses env default.
        
        # Bolt robot settings
        bolt_name: Robot name to connect to. None uses env default.
        bolt_scanning_time: Time in seconds to scan for robot. None uses env default.
        bolt_color_r: LED red component (0-255). None uses env default.
        bolt_color_g: LED green component (0-255). None uses env default.
        bolt_color_b: LED blue component (0-255). None uses env default.
        bolt_use_dummy: Use dummy robot instead of real hardware. Defaults to False.
        
        extra: Additional backend-specific configuration options.
    """

    model_config = ConfigDict(use_enum_values=True)

    # Camera settings
    camera_type: CameraType | str | None = Field(
        default=None, description="Camera type: DUMMY, WEBCAM, or PICAM"
    )
    camera_id: int | None = Field(default=None, ge=0, description="Camera device ID")
    camera_resolution: tuple[int, int] | None = Field(
        default=None, description="Camera resolution as (width, height) tuple. Accepts CameraResolution enum or custom tuple/list on input."
    )
    camera_exposure_time_ms: float | None = Field(
        default=None, gt=0, description="Exposure time in milliseconds"
    )

    # Bolt robot settings
    bolt_name: str | None = Field(default=None, description="Robot name to connect to")
    bolt_scanning_time: float | None = Field(
        default=None, gt=0, description="Scanning time in seconds"
    )
    bolt_color_r: int | None = Field(default=None, ge=0, le=255, description="Red (0-255)")
    bolt_color_g: int | None = Field(default=None, ge=0, le=255, description="Green (0-255)")
    bolt_color_b: int | None = Field(default=None, ge=0, le=255, description="Blue (0-255)")
    bolt_use_dummy: bool | None = Field(
        default=None, description="Use dummy robot instead of real hardware"
    )

    # Additional options
    extra: dict[str, Any] = Field(
        default_factory=dict, description="Additional backend-specific options"
    )

    @field_validator("camera_type", mode="before")
    @classmethod
    def validate_camera_type(cls, v: Any) -> CameraType | str | None:
        """Validate and convert camera type string to enum if valid."""
        if v is None:
            return None

        if isinstance(v, CameraType):
            return v

        if isinstance(v, str):
            v_lower = v.lower()
            valid_types = {e.value for e in CameraType}
            if v_lower not in valid_types:
                raise ValueError(
                    f"Invalid camera type '{v}'. Must be one of: {', '.join(sorted(valid_types))}"
                )
            return CameraType(v_lower)

        raise ValueError(f"Camera type must be string or CameraType enum, got {type(v).__name__}")

    @field_validator("camera_resolution", mode="before")
    @classmethod
    def validate_camera_resolution(cls, v: Any) -> tuple[int, int] | None:
        """Validate and normalize camera resolution to always be a tuple.

        Input can be:
        - None (uses server defaults)
        - CameraResolution enum (e.g., CameraResolution.RES_640x480) → converts to (640, 480)
        - Tuple (e.g., (640, 480)) → stays as tuple
        - List (e.g., [640, 480] from JSON) → JsonTuple wrapper converts to tuple

        Always returns: None or tuple[int, int]
        This ensures the field always contains a tuple for consistent serialization.
        """
        if v is None:
            return None

        # CameraResolution enum - convert to tuple
        if isinstance(v, CameraResolution):
            return tuple(v)  # CameraResolution inherits from tuple, so tuple(enum) works

        # Tuple or List - validate and return as tuple
        if isinstance(v, (tuple, list)) and len(v) == 2:
            width, height = v[0], v[1]
            if not isinstance(width, int) or not isinstance(height, int):
                raise ValueError("Camera resolution must contain two integers (width, height)")
            if width <= 0 or height <= 0:
                raise ValueError("Camera resolution must have positive values")
            return width, height

        raise ValueError(
            f"Camera resolution must be CameraResolution enum or (width, height) tuple, got {type(v).__name__}"
        )

    def _apply_camera_defaults(self, defaults: dict[str, Any]) -> dict[str, Any]:
        """Apply default values for None camera fields.

        Args:
            defaults: Dictionary of default camera values

        Returns:
            Dictionary with camera values, using defaults for None fields
        """
        # Handle resolution - convert enum or tuple to separate width/height
        resolution = self.camera_resolution if self.camera_resolution is not None else defaults['resolution']
        if isinstance(resolution, (tuple, list)):
            width, height = resolution[0], resolution[1]
        else:
            # Should not happen due to validation, but handle it
            width, height = resolution

        return {
            'type': self.camera_type if self.camera_type is not None else defaults['type'],
            'id': self.camera_id if self.camera_id is not None else defaults['id'],
            'width': width,
            'height': height,
            'exposure_time_ms': self.camera_exposure_time_ms if self.camera_exposure_time_ms is not None else defaults['exposure_time_ms'],
        }

    def _apply_bolt_defaults(self, defaults: dict[str, Any]) -> dict[str, Any]:
        """Apply default values for None bolt fields.

        Args:
            defaults: Dictionary of default bolt values

        Returns:
            Dictionary with bolt values, using defaults for None fields
        """
        return {
            'name': self.bolt_name if self.bolt_name is not None else defaults['name'],
            'scanning_time': self.bolt_scanning_time if self.bolt_scanning_time is not None else defaults['scanning_time'],
            'color_r': self.bolt_color_r if self.bolt_color_r is not None else defaults['color_r'],
            'color_g': self.bolt_color_g if self.bolt_color_g is not None else defaults['color_g'],
            'color_b': self.bolt_color_b if self.bolt_color_b is not None else defaults['color_b'],
            'use_dummy': self.bolt_use_dummy if self.bolt_use_dummy is not None else defaults['use_dummy'],
        }
