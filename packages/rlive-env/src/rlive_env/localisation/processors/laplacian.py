"""Laplacian edge detection processor."""

import cv2 as cv
import numpy as np

from rlive_env.localisation.processors.base import AbstractProcessor
from rlive_env.config import processing_config


class LaplacianProcessor(AbstractProcessor):
    """Apply Laplacian edge detection.

    Loads default parameters from rlive_env.config.processing_config module.

    Examples:
        processor = LaplacianProcessor()  # Uses defaults from config
        edges = processor.process(gray_image)

        processor = LaplacianProcessor(ksize=5)  # Override default
        edges = processor.process(gray_image)
    """

    def __init__(self, ksize: int = processing_config.LAPLACIAN_KSIZE):
        """Initialize Laplacian processor with defaults from config module.

        Args:
            ksize: Kernel size (1, 3, 5, or 7).

        Raises:
            ValueError: If ksize is not valid.
        """
        if ksize not in (1, 3, 5, 7):
            raise ValueError("ksize must be 1, 3, 5, or 7")
        self.ksize = ksize

    @property
    def name(self) -> str:
        """Return processor name."""
        return "Laplacian"

    def process(self, image: np.ndarray) -> np.ndarray:
        """Apply Laplacian edge detection.

        Args:
            image: Input grayscale image.

        Returns:
            Absolute Laplacian image (uint8).
        """
        laplacian = cv.Laplacian(image, cv.CV_64F, ksize=self.ksize)
        return cv.convertScaleAbs(laplacian)

