"""Canny edge detection processor."""

import cv2 as cv
import numpy as np

from rlive_env.localisation.processors.base import AbstractProcessor
from rlive_env.config import processing_config


class CannyProcessor(AbstractProcessor):
    """Apply Canny edge detection.

    Loads default parameters from rlive_env.config.processing_config module.

    Examples:
        processor = CannyProcessor()  # Uses defaults from config
        edges = processor.process(gray_image)

        processor = CannyProcessor(threshold1=75, threshold2=200)  # Override defaults
        edges = processor.process(gray_image)
    """

    def __init__(
        self,
        threshold1: int = processing_config.CANNY_THRESHOLD1,
        threshold2: int = processing_config.CANNY_THRESHOLD2,
        aperture: int = processing_config.CANNY_APERTURE,
    ):
        """Initialize Canny edge detector with defaults from config module.

        Args:
            threshold1: First threshold for the hysteresis procedure.
            threshold2: Second threshold for the hysteresis procedure.
            aperture: Aperture size for the Sobel operator (3, 5, or 7).

        Raises:
            ValueError: If aperture is not 3, 5, or 7.
        """
        if aperture not in (3, 5, 7):
            raise ValueError("aperture must be 3, 5, or 7")
        self.threshold1 = threshold1
        self.threshold2 = threshold2
        self.aperture = aperture

    @property
    def name(self) -> str:
        """Return processor name."""
        return "Canny"

    def process(self, image: np.ndarray) -> np.ndarray:
        """Apply Canny edge detection.

        Args:
            image: Input grayscale image.

        Returns:
            Edge image.
        """
        return cv.Canny(image, self.threshold1, self.threshold2, apertureSize=self.aperture)

