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
        CameraResolution.HD  # (1280, 720)
        CameraResolution.FULL_HD  # (1920, 1080)
        (640, 480)  # Custom resolution as tuple
    """

    # Common resolutions
    QVGA = (320, 240)
    VGA = (640, 480)
    SVGA = (800, 600)
    XGA = (1024, 768)
    HD = (1280, 720)
    FULL_HD = (1920, 1080)

    def __new__(cls, width: int, height: int) -> "CameraResolution":
        """Create a new CameraResolution tuple."""
        obj = tuple.__new__(cls, (width, height))
        obj._value_ = (width, height)
        return obj

    @property
    def width(self) -> int:
        """Get width component."""
        return self[0]

    @property
    def height(self) -> int:
        """Get height component."""
        return self[1]

    def __str__(self) -> str:
        """Return string representation as WIDTHxHEIGHT."""
        return f"{self.width}x{self.height}"

    def __repr__(self) -> str:
        """Return repr as tuple."""
        return f"({self.width}, {self.height})"

