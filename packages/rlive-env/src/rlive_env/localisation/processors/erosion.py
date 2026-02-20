"""Morphological erosion processor."""

import cv2 as cv
import numpy as np

from rlive_env.localisation.processors.base import AbstractProcessor
from rlive_env.config import processing_config


class ErosionProcessor(AbstractProcessor):
    """Apply morphological erosion to remove small noise.

    Erosion shrinks white regions in binary images, useful for removing
    small noise blobs after thresholding. Typically used together with
    DilationProcessor (erosion first to remove noise, dilation to restore size).

    Examples:
        processor = ErosionProcessor()
        eroded = processor.process(binary_image)

        processor = ErosionProcessor(kernel_size=(3, 3), iterations=1)
        eroded = processor.process(binary_image)
    """

    def __init__(
        self,
        kernel_size: tuple[int, int] = (
            processing_config.DILATION_KERNEL_WIDTH,
            processing_config.DILATION_KERNEL_HEIGHT,
        ),
        kernel_shape: int = cv.MORPH_ELLIPSE,
        iterations: int = 1,
    ):
        """Initialize erosion processor.

        Args:
            kernel_size: Size of the structuring element.
            kernel_shape: Shape of the structuring element.
            iterations: Number of times erosion is applied.
        """
        self.kernel_size = kernel_size
        self.kernel_shape = kernel_shape
        self.iterations = iterations

    @property
    def name(self) -> str:
        """Return processor name."""
        return "Erosion"

    def process(self, image: np.ndarray) -> np.ndarray:
        """Apply morphological erosion.

        Args:
            image: Input binary or grayscale image.

        Returns:
            Eroded image.
        """
        kernel = cv.getStructuringElement(self.kernel_shape, self.kernel_size)
        return cv.erode(image, kernel, iterations=self.iterations)
