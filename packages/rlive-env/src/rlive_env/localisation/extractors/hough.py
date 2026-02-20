"""Hough circle detection extractor."""

import cv2 as cv
import numpy as np

from rlive_env.localisation.ball_location import BallLocation
from rlive_env.localisation.extractors.base import AbstractBallExtractor
from rlive_env.config import extractor_config
from rlive_common.utils import get_logger

logger = get_logger(__name__)


class HoughCircleExtractor(AbstractBallExtractor):
    """Extract ball location using Hough circle detection.

    Uses OpenCV's HoughCircles to detect circular objects in preprocessed images.
    Well-suited for images with clear circular edges (e.g., Canny, Laplacian output).

    Loads default parameters from rlive_env.config.extractor_config module.

    Examples:
        Basic usage with defaults from config:
            extractor = HoughCircleExtractor()
            location = extractor.extract(processed_image)

        Override specific parameters:
            extractor = HoughCircleExtractor(hough_param2=25, hough_min_radius=15)
            location = extractor.extract(processed_image)
    """

    def __init__(
        self,
        hough_dp: float = extractor_config.HOUGH_DP,
        hough_min_dist: int = extractor_config.HOUGH_MIN_DIST,
        hough_param1: int = extractor_config.HOUGH_PARAM1,
        hough_param2: int = extractor_config.HOUGH_PARAM2,
        hough_min_radius: int = extractor_config.HOUGH_MIN_RADIUS,
        hough_max_radius: int = extractor_config.HOUGH_MAX_RADIUS,
    ):
        """Initialize Hough circle extractor with defaults from config module.

        Args:
            hough_dp: Inverse ratio of accumulator resolution.
            hough_min_dist: Minimum distance between circles.
            hough_param1: Canny edge threshold.
            hough_param2: Accumulator threshold.
            hough_min_radius: Minimum circle radius.
            hough_max_radius: Maximum circle radius.
        """
        self.hough_dp = hough_dp
        self.hough_min_dist = hough_min_dist
        self.hough_param1 = hough_param1
        self.hough_param2 = hough_param2
        self.hough_min_radius = hough_min_radius
        self.hough_max_radius = hough_max_radius

    @property
    def name(self) -> str:
        """Return extractor name."""
        return "HoughCircle"

    def extract(self, image: np.ndarray) -> BallLocation | None:
        """Extract ball location using Hough circle detection.

        Args:
            image: Grayscale or processed image (typically blurred for best results).

        Returns:
            BallLocation if circle detected, None otherwise.
        """
        if image is None or image.size == 0:
            logger.debug(f"{self.name}: Empty image provided")
            return None

        try:
            circles = cv.HoughCircles(
                image,
                cv.HOUGH_GRADIENT,
                dp=self.hough_dp,
                minDist=self.hough_min_dist,
                param1=self.hough_param1,
                param2=self.hough_param2,
                minRadius=self.hough_min_radius,
                maxRadius=self.hough_max_radius,
            )

            if circles is None:
                logger.debug(f"{self.name}: No circles detected")
                return None

            # Extract first (most confident) circle center
            circles = np.uint16(np.around(circles))
            x, y, _ = circles[0, 0]

            location = BallLocation(x=int(x), y=int(y))
            logger.debug(f"{self.name}: Ball detected at {location.as_tuple()}")
            return location

        except Exception as e:
            logger.error(f"{self.name}: Extraction failed - {e}")
            return None

    def get_debug_image(self, image: np.ndarray) -> np.ndarray | None:
        """Get debug visualization showing detected circles.

        Args:
            image: Grayscale image used for detection.

        Returns:
            RGB image with detected circles drawn, or None if no circles found.
        """
        if image is None or image.size == 0:
            return None

        try:
            circles = cv.HoughCircles(
                image,
                cv.HOUGH_GRADIENT,
                dp=self.hough_dp,
                minDist=self.hough_min_dist,
                param1=self.hough_param1,
                param2=self.hough_param2,
                minRadius=self.hough_min_radius,
                maxRadius=self.hough_max_radius,
            )

            # Convert grayscale to BGR for colored annotations
            if len(image.shape) == 2:
                debug_img = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
            else:
                debug_img = image.copy()

            if circles is not None:
                circles = np.uint16(np.around(circles))
                # Draw all detected circles
                for circle in circles[0, :]:
                    x, y, r = circle
                    # Draw circle
                    cv.circle(debug_img, (x, y), r, (0, 255, 0), 2)
                    # Draw center
                    cv.circle(debug_img, (x, y), 3, (0, 0, 255), -1)

            return debug_img

        except Exception as e:
            logger.error(f"{self.name}: Debug image generation failed - {e}")
            return None
