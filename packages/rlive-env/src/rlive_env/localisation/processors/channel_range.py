"""Single-channel range (threshold) processor with optional color-space conversion."""

import cv2 as cv
import numpy as np

from rlive_env.localisation.processors.base import AbstractProcessor


# Supported color-space conversions (input is always RGB)
_CONVERSIONS: dict[str, int | None] = {
    "rgb":    None,                        # no conversion needed
    "bgr":    cv.COLOR_RGB2BGR,
    "hsv":    cv.COLOR_RGB2HSV,
    "lab":    cv.COLOR_RGB2LAB,
    "ycrcb":  cv.COLOR_RGB2YCrCb,
    "gray":   cv.COLOR_RGB2GRAY,
}

_CHANNEL_NAMES: dict[str, list[str]] = {
    "bgr":   ["B", "G", "R"],
    "rgb":   ["R", "G", "B"],
    "hsv":   ["H", "S", "V"],
    "lab":   ["L", "A", "B"],
    "ycrcb": ["Y", "Cr", "Cb"],
    "gray":  ["Gray"],
}


class ChannelRangeProcessor(AbstractProcessor):
    """Extract one channel from a (converted) image and apply a min/max range mask.

    Combines three operations in a single step:
        1. Optionally convert the BGR input to another colour space.
        2. Extract a single channel.
        3. Return a binary mask: 255 where  lower <= pixel <= upper, else 0.

    This is particularly useful for shadow rejection:

        Example — keep only bright pixels from the V (Value) channel of HSV:
            processor = ChannelRangeProcessor(
                color_space="hsv", channel=2,   # V channel
                lower=90, upper=255,            # reject dark/shadow pixels
            )
            mask = processor.process(bgr_image)

        Example — keep the red channel only where it is saturated:
            processor = ChannelRangeProcessor(
                color_space="bgr", channel=2,   # R in BGR
                lower=150, upper=255,
            )
            mask = processor.process(bgr_image)

        Example — isolate bright, high-contrast areas via L (LAB):
            processor = ChannelRangeProcessor(
                color_space="lab", channel=0,   # L channel
                lower=100, upper=255,
            )
            mask = processor.process(bgr_image)

    Supported color spaces (color_space parameter):
        "bgr"    – no conversion; channels = B, G, R
        "rgb"    – channels = R, G, B
        "hsv"    – channels = H (0–180), S (0–255), V (0–255)
        "lab"    – channels = L, A, B
        "ycrcb"  – channels = Y, Cr, Cb
        "gray"   – single channel (channel must be 0)

    Args:
        color_space: Target colour space to convert to before extraction.
        channel: Index of the channel to extract (0, 1, or 2).
        lower: Minimum pixel value (inclusive) to keep → 255 in output.
        upper: Maximum pixel value (inclusive) to keep → 255 in output.
    """

    def __init__(
        self,
        color_space: str = "hsv",
        channel: int = 2,
        lower: int = 0,
        upper: int = 255,
    ):
        """Initialize ChannelRangeProcessor.

        Args:
            color_space: Colour space to convert to before extraction.
                         One of "bgr", "rgb", "hsv", "lab", "ycrcb", "gray".
            channel: Index of the channel to extract (0, 1, or 2).
            lower: Minimum pixel value (inclusive) that passes the mask.
            upper: Maximum pixel value (inclusive) that passes the mask.

        Raises:
            ValueError: If arguments are out of range or incompatible.
        """
        cs = color_space.lower().strip()
        if cs not in _CONVERSIONS:
            raise ValueError(
                f"Unsupported color_space '{color_space}'. "
                f"Choose from: {list(_CONVERSIONS.keys())}"
            )
        if cs == "gray" and channel != 0:
            raise ValueError("color_space='gray' only has channel 0")
        if channel not in (0, 1, 2):
            raise ValueError("channel must be 0, 1, or 2")
        if not (0 <= lower <= 255 and 0 <= upper <= 255):
            raise ValueError("lower and upper must be in 0–255")

        self.color_space = cs
        self.channel = channel
        self.lower = lower
        self.upper = upper

    @property
    def name(self) -> str:
        """Return processor name including colour space, channel and range."""
        ch_name = _CHANNEL_NAMES[self.color_space][self.channel]
        return f"ChannelRange_{self.color_space.upper()}_{ch_name}[{self.lower},{self.upper}]"

    def process(self, image: np.ndarray) -> np.ndarray:
        """Extract channel and apply range mask.

        Args:
            image: Input RGB image.

        Returns:
            Binary mask (uint8, 0 or 255): white where the channel value
            falls within [lower, upper], black elsewhere.
        """
        # 1. Convert colour space
        code = _CONVERSIONS[self.color_space]
        if code is not None:
            converted = cv.cvtColor(image, code)
        else:
            converted = image

        # 2. Extract channel
        if self.color_space == "gray":
            ch = converted  # already single-channel after cvtColor
        elif len(converted.shape) == 2:
            ch = converted  # already grayscale
        else:
            ch = converted[:, :, self.channel]

        # 3. Apply range threshold → binary mask
        lo = np.array([self.lower], dtype=np.uint8)
        hi = np.array([self.upper], dtype=np.uint8)
        mask: np.ndarray = cv.inRange(ch, lo, hi)
        return mask




