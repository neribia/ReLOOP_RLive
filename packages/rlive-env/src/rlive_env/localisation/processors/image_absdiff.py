"""Image difference-based ball detection extractor."""

import cv2 as cv
import numpy as np

from rlive_env.localisation.ball_location import BallLocation
from rlive_env.localisation.processors.base import AbstractProcessor
from rlive_env.config import extractor_config
from rlive_common.utils import get_logger

logger = get_logger(__name__)


class ImageAbsDiff(AbstractProcessor):
    """Extract ball location using frame differencing.

    Saves the last image and computes the difference between consecutive frames.
    Detects motion by finding contours in the difference image.

    This is useful for detecting moving objects when the background is static.

    Examples:
        Basic usage:
            extractor = ImageDiffExtractor()
            location = extractor.extract(current_image)

        After first frame, subsequent extractions detect motion.
    """

    def __init__(self, min_contour_area: int = extractor_config.MIN_CONTOUR_AREA):
        """Initialize image diff extractor.

        Args:
            min_contour_area: Minimum contour area for valid detection.
        """
        self.min_contour_area = min_contour_area
        self.last_image: np.ndarray | None = None

    @property
    def name(self) -> str:
        """Return extractor name."""
        return "ImageAbsDiff"

    def process(self, image: np.ndarray) -> np.ndarray | None:
        """Makes a absolut diff from two images

        Args:
            image: Current binary or grayscale image.

        Returns:
            image (np.ndarray): Absolute difference image, or None if no previous frame.
        """
        if image is None or image.size == 0:
            logger.debug(f"{self.name}: Empty image provided")
            return None

        try:
            # If no previous frame, save current and return None
            if self.last_image is None:
                self.last_image = image.copy()
                logger.debug(f"{self.name}: First frame saved, no diff available")
                return None

            # Ensure images have same shape
            if self.last_image.shape != image.shape:
                logger.debug(f"{self.name}: Image shape mismatch, resetting")
                self.last_image = image.copy()
                return None

            # Compute absolute difference
            diff = cv.absdiff(self.last_image, image)
            self.last_image = image
            return diff

        except Exception as e:
            logger.error(f"{self.name}: Process failed - {e}")
            return None
