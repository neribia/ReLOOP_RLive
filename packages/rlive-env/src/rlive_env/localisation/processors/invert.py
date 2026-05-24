"""Invert (bitwise NOT) processor."""

import cv2 as cv
import numpy as np

from rlive_env.localisation.processors.base import AbstractProcessor


class InvertProcessor(AbstractProcessor):
    """Invert an image by applying a bitwise NOT operation.

    Each pixel value p becomes 255 - p.  This is most commonly used to
    flip a binary mask so that white regions become black and vice versa,
    but it works equally well on grayscale and colour images.

    Examples:
        processor = InvertProcessor()
        inverted = processor.process(binary_mask)
    """

    @property
    def name(self) -> str:
        """Return processor name."""
        return "Invert"

    def process(self, image: np.ndarray) -> np.ndarray:
        """Invert the image.

        Args:
            image: Input image (binary mask, grayscale, or colour).

        Returns:
            Inverted image of the same shape and dtype.
        """
        return cv.bitwise_not(image)

