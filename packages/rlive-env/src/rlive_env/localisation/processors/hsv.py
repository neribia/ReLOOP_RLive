"""HSV color space processor for color-based filtering."""

import cv2 as cv
import numpy as np

from rlive_env.localisation.processors.base import AbstractProcessor
from rlive_env.config import processing_config


class HSVProcessor(AbstractProcessor):
    """Convert image to HSV color space and apply color range filtering.

    HSV (Hue, Saturation, Value) color space is useful for isolating colors
    that are robust to lighting changes. This processor converts RGB/BGR to HSV
    and optionally applies a color range mask.

    Loads default parameters from rlive_env.config.processing_config module.

    Examples:
        Basic HSV conversion without masking:
            processor = HSVProcessor()
            hsv_image = processor.process(bgr_image)

        HSV conversion with color range filtering:
            processor = HSVProcessor(
                lower_hue=10, upper_hue=25,      # Orange/Yellow hues
                lower_sat=50, upper_sat=255,
                lower_val=50, upper_val=255
            )
            mask = processor.process(bgr_image)  # Returns binary mask

        Red color detection (red wraps around 0-10 and 170-180):
            processor = HSVProcessor(
                lower_hue=0, upper_hue=10,
                lower_sat=50, upper_sat=255,
                lower_val=50, upper_val=255
            )
            mask = processor.process(image)
    """

    def __init__(
        self,
        lower_hue: int = getattr(processing_config, "HSV_LOWER_HUE", 0),
        upper_hue: int = getattr(processing_config, "HSV_UPPER_HUE", 180),
        lower_sat: int = getattr(processing_config, "HSV_LOWER_SAT", 0),
        upper_sat: int = getattr(processing_config, "HSV_UPPER_SAT", 255),
        lower_val: int = getattr(processing_config, "HSV_LOWER_VAL", 0),
        upper_val: int = getattr(processing_config, "HSV_UPPER_VAL", 255),
        apply_mask: bool = False,
    ):
        """Initialize HSV processor.

        Args:
            lower_hue: Lower bound for hue (0-180 in OpenCV).
            upper_hue: Upper bound for hue (0-180 in OpenCV).
            lower_sat: Lower bound for saturation (0-255).
            upper_sat: Upper bound for saturation (0-255).
            lower_val: Lower bound for value/brightness (0-255).
            upper_val: Upper bound for value/brightness (0-255).
            apply_mask: If True, returns binary mask of colors in range.
                       If False, returns HSV image.

        Raises:
            ValueError: If hue bounds are invalid (must be 0-180 in OpenCV).
        """
        if not (0 <= lower_hue <= 180 and 0 <= upper_hue <= 180):
            raise ValueError("Hue bounds must be 0-180 for OpenCV HSV")
        if not (0 <= lower_sat <= 255 and 0 <= upper_sat <= 255):
            raise ValueError("Saturation bounds must be 0-255")
        if not (0 <= lower_val <= 255 and 0 <= upper_val <= 255):
            raise ValueError("Value bounds must be 0-255")

        self.lower_hue = lower_hue
        self.upper_hue = upper_hue
        self.lower_sat = lower_sat
        self.upper_sat = upper_sat
        self.lower_val = lower_val
        self.upper_val = upper_val
        self.apply_mask = apply_mask
        self.kernal_size = 5

    @property
    def name(self) -> str:
        """Return processor name."""
        return "HSV"

    def process(self, image: np.ndarray) -> np.ndarray:
        """Convert image to HSV and optionally apply color range mask.

        Args:
            image: Input image in BGR format (standard OpenCV format).

        Returns:
            If apply_mask=False: HSV image (3-channel).
            If apply_mask=True: Binary mask of pixels within color range.
        """
        # Convert BGR to HSV
        hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)

        if not self.apply_mask:
            return hsv_image

        # Create color range mask
        lower_bound = np.array([self.lower_hue, self.lower_sat, self.lower_val], dtype=np.uint8)
        upper_bound = np.array([self.upper_hue, self.upper_sat, self.upper_val], dtype=np.uint8)

        mask = cv.inRange(hsv_image, lower_bound, upper_bound)
        return mask

    def get_hsv_ranges(self) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        """Get current HSV color range bounds.

        Returns:
            Tuple of (lower_bounds, upper_bounds) where each is (H, S, V).
        """
        return (
            (self.lower_hue, self.lower_sat, self.lower_val),
            (self.upper_hue, self.upper_sat, self.upper_val),
        )

    def set_hsv_ranges(
        self,
        lower_hue: int,
        upper_hue: int,
        lower_sat: int,
        upper_sat: int,
        lower_val: int,
        upper_val: int,
    ) -> None:
        """Update HSV color range bounds.

        Args:
            lower_hue: Lower hue bound (0-180).
            upper_hue: Upper hue bound (0-180).
            lower_sat: Lower saturation bound (0-255).
            upper_sat: Upper saturation bound (0-255).
            lower_val: Lower value bound (0-255).
            upper_val: Upper value bound (0-255).

        Raises:
            ValueError: If bounds are invalid.
        """
        if not (0 <= lower_hue <= 180 and 0 <= upper_hue <= 180):
            raise ValueError("Hue bounds must be 0-180 for OpenCV HSV")
        if not (0 <= lower_sat <= 255 and 0 <= upper_sat <= 255):
            raise ValueError("Saturation bounds must be 0-255")
        if not (0 <= lower_val <= 255 and 0 <= upper_val <= 255):
            raise ValueError("Value bounds must be 0-255")

        self.lower_hue = lower_hue
        self.upper_hue = upper_hue
        self.lower_sat = lower_sat
        self.upper_sat = upper_sat
        self.lower_val = lower_val
        self.upper_val = upper_val

