"""Enums for rlive-common core models."""

from enum import Enum


class CameraType(str, Enum):
    """Supported camera types."""
    DUMMY = "dummy"
    WEBCAM = "webcam"
    PICAM = "picam"

    def __str__(self) -> str:
        """Return the enum value as string."""
        return self.value


class CameraResolution(tuple, Enum):
    """Camera resolution enumeration with common presets.

    Each resolution is a tuple of (width, height).
    Custom resolutions can also be passed as tuples directly.

    Examples:
        CameraResolution.RES_1280x720  # (1280, 720)
        CameraResolution.RES_1920x1080  # (1920, 1080)
        (640, 480)  # Custom resolution as tuple
    """

    # Common resolutions
    RES_320x240 = (320, 240)
    RES_640x480 = (640, 480)
    RES_800x600 = (800, 600)
    RES_1024x768 = (1024, 768)
    RES_1280x720 = (1280, 720)
    RES_1920x1080 = (1920, 1080)

    @property
    def width(self) -> int:
        """Get width component (first element of tuple)."""
        return self[0]

    @property
    def height(self) -> int:
        """Get height component (second element of tuple)."""
        return self[1]

    def __tuple__(self) -> tuple:
        return int(self[0]), int(self[1])

    def __str__(self) -> str:
        """Return string representation as WIDTHxHEIGHT."""
        return f"{self.width}x{self.height}"

    def __repr__(self) -> str:
        """Return representation showing both enum name and tuple value."""
        return f"{self.__class__.__name__}.{self.name}({self.width}, {self.height})"

