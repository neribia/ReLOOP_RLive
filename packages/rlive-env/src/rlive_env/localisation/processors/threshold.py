"""Binary threshold processor."""

import cv2 as cv
import numpy as np

from rlive_env.localisation.processors.base import AbstractProcessor
from rlive_env.config import processing_config


class ThresholdProcessor(AbstractProcessor):
    """Apply binary thresholding.

    Loads default parameters from rlive_env.config.processing_config module.

    Examples:
        processor = ThresholdProcessor()  # Uses defaults from config
        binary = processor.process(gray_image)

        processor = ThresholdProcessor(threshold_value=100)  # Override default
        binary = processor.process(gray_image)
    """

    def __init__(
        self,
        threshold_value: int = processing_config.THRESHOLD_VALUE,
        max_value: int = processing_config.THRESHOLD_MAX,
        threshold_type: int = cv.THRESH_BINARY,
    ):
        """Initialize threshold processor with defaults from config module.

        Args:
            threshold_value: Threshold value (0-255).
            max_value: Maximum value for thresholding.
            threshold_type: OpenCV threshold type (e.g., cv.THRESH_BINARY).
        """
        self.threshold_value = threshold_value
        self.max_value = max_value
        self.threshold_type = threshold_type

    @property
    def name(self) -> str:
        """Return processor name."""
        return "Threshold"

    def process(self, image: np.ndarray) -> np.ndarray:
        """Apply binary thresholding.

        Args:
            image: Input grayscale image.

        Returns:
            Binary thresholded image.
        """
        _, thresh = cv.threshold(image, self.threshold_value, self.max_value, self.threshold_type)
        return thresh

