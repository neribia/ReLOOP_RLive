"""Gaussian blur processor."""

import cv2 as cv
import numpy as np

from rlive_env.localisation.processors.base import AbstractProcessor
from rlive_env.config import processing_config


class GaussianBlurProcessor(AbstractProcessor):
    """Apply Gaussian blur to reduce noise.

    Loads default parameters from rlive_env.config.processing_config module.

    Examples:
        processor = GaussianBlurProcessor()  # Uses defaults from config
        blurred = processor.process(image)

        processor = GaussianBlurProcessor(kernel_size=11, sigma=3.0)  # Override defaults
        blurred = processor.process(image)
    """

    def __init__(
        self,
        kernel_size: int = processing_config.BLUR_KERNEL_SIZE,
        sigma: float = processing_config.BLUR_SIGMA,
    ):
        """Initialize Gaussian blur processor with defaults from config module.

        Args:
            kernel_size: Size of the Gaussian kernel (must be odd).
            sigma: Gaussian kernel standard deviation.

        Raises:
            ValueError: If kernel_size is even.
        """
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        self.kernel_size = kernel_size
        self.sigma = sigma

    @property
    def name(self) -> str:
        """Return processor name."""
        return "GaussianBlur"

    def process(self, image: np.ndarray) -> np.ndarray:
        """Apply Gaussian blur.

        Args:
            image: Input image (grayscale or color).

        Returns:
            Blurred image.
        """
        return cv.GaussianBlur(image, (self.kernel_size, self.kernel_size), self.sigma)

