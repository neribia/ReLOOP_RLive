"""Morphological dilation processor."""

import cv2 as cv
import numpy as np

from rlive_env.localisation.processors.base import AbstractProcessor
from rlive_env.config import processing_config


class DilationProcessor(AbstractProcessor):
    """Apply morphological dilation.

    Loads default parameters from rlive_env.config.processing_config module.

    Examples:
        processor = DilationProcessor()  # Uses defaults from config
        dilated = processor.process(binary_image)

        processor = DilationProcessor(kernel_size=(5, 5), iterations=3)  # Override defaults
        dilated = processor.process(binary_image)
    """

    def __init__(
        self,
        kernel_size: tuple[int, int] = (processing_config.DILATION_KERNEL_WIDTH, processing_config.DILATION_KERNEL_HEIGHT),
        kernel_shape: int = cv.MORPH_ELLIPSE,
        iterations: int = processing_config.DILATION_ITERATIONS,
    ):
        """Initialize dilation processor with defaults from config module.

        Args:
            kernel_size: Size of the structuring element.
            kernel_shape: Shape of the structuring element (cv.MORPH_RECT, cv.MORPH_ELLIPSE, etc.).
            iterations: Number of times dilation is applied.
        """
        self.kernel_size = kernel_size
        self.kernel_shape = kernel_shape
        self.iterations = iterations

    @property
    def name(self) -> str:
        """Return processor name."""
        return "Dilation"

    def process(self, image: np.ndarray) -> np.ndarray:
        """Apply morphological dilation.

        Args:
            image: Input binary image.

        Returns:
            Dilated image.
        """
        kernel = cv.getStructuringElement(self.kernel_shape, self.kernel_size)
        return cv.dilate(image, kernel, iterations=self.iterations)

