"""Grayscale conversion processor."""

import cv2 as cv
import numpy as np

from rlive_env.localisation.processors.base import AbstractProcessor


class GrayscaleProcessor(AbstractProcessor):
    """Convert RGB image to grayscale.

    Examples:
        processor = GrayscaleProcessor()
        gray = processor.process(rgb_image)
    """

    @property
    def name(self) -> str:
        """Return processor name."""
        return "Grayscale"

    def process(self, image: np.ndarray) -> np.ndarray:
        """Convert RGB image to grayscale.

        Args:
            image: Input RGB image.

        Returns:
            Grayscale image.
        """
        return cv.cvtColor(image, cv.COLOR_RGB2GRAY)

