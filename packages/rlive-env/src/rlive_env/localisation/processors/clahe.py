"""CLAHE (Contrast Limited Adaptive Histogram Equalization) processor."""

import cv2 as cv
import numpy as np

from rlive_env.localisation.processors.base import AbstractProcessor


class CLAHEProcessor(AbstractProcessor):
    """Apply CLAHE to enhance local contrast in low-contrast images.

    CLAHE (Contrast Limited Adaptive Histogram Equalization) divides the image
    into small tiles and applies histogram equalization locally. This is very
    effective for images where the object and background have similar brightness,
    such as a gray ball on a gray surface.

    Input must be a single-channel (grayscale) image.

    Examples:
        Basic usage:
            processor = CLAHEProcessor()
            enhanced = processor.process(gray_image)

        Aggressive contrast enhancement:
            processor = CLAHEProcessor(clip_limit=4.0, tile_grid_size=(4, 4))
            enhanced = processor.process(gray_image)
    """

    def __init__(
        self,
        clip_limit: float = 3.0,
        tile_grid_size: tuple[int, int] = (8, 8),
    ):
        """Initialize CLAHE processor.

        Args:
            clip_limit: Contrast limit for histogram clipping.
                       Higher values = more contrast enhancement.
                       Range: 1.0-40.0 (default: 3.0).
            tile_grid_size: Grid size for local histogram equalization.
                           Smaller tiles = more local enhancement.
                           Typical values: (4,4), (8,8), (16,16).
        """
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    @property
    def name(self) -> str:
        """Return processor name."""
        return "CLAHE"

    def process(self, image: np.ndarray) -> np.ndarray:
        """Apply CLAHE to enhance local contrast.

        Args:
            image: Input single-channel (grayscale) image.

        Returns:
            Contrast-enhanced grayscale image.
        """
        clahe = cv.createCLAHE(
            clipLimit=self.clip_limit,
            tileGridSize=self.tile_grid_size,
        )
        return clahe.apply(image)