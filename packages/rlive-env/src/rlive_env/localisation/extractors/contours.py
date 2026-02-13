"""Contour-based ball detection extractor."""

import cv2 as cv
import numpy as np

from rlive_env.localisation.ball_location import BallLocation
from rlive_env.localisation.extractors.base import AbstractBallExtractor
from rlive_env.config import extractor_config
from rlive_common.utils import get_logger

logger = get_logger(__name__)


class ContourExtractor(AbstractBallExtractor):
    """Extract ball location using contour analysis.

    Finds contours in binary/edge-detected images, filters by area,
    and returns the centroid of the largest valid contour.

    Loads default parameters from rlive_env.config.extractor_config module.

    Examples:
        Basic usage with defaults from config:
            extractor = ContourExtractor()
            location = extractor.extract(binary_image)

        Override minimum area threshold:
            extractor = ContourExtractor(min_contour_area=150)
            location = extractor.extract(binary_image)
    """

    def __init__(self, min_contour_area: int = extractor_config.MIN_CONTOUR_AREA):
        """Initialize contour extractor with defaults from config module.

        Args:
            min_contour_area: Minimum contour area for valid detection.
        """
        self.min_contour_area = min_contour_area

    @property
    def name(self) -> str:
        """Return extractor name."""
        return "Contour"

    def extract(self, image: np.ndarray) -> BallLocation | None:
        """Extract ball location using contour analysis.

        Args:
            image: Binary or edge-detected image.

        Returns:
            BallLocation if valid contour found, None otherwise.
        """
        if image is None or image.size == 0:
            logger.debug(f"{self.name}: Empty image provided")
            return None

        try:
            # Find contours
            contours, _ = cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

            if not contours:
                logger.debug(f"{self.name}: No contours found")
                return None

            # Filter by area
            valid_contours = [
                c for c in contours if cv.contourArea(c) >= self.min_contour_area
            ]

            if not valid_contours:
                logger.debug(
                    f"{self.name}: No contours with area >= {self.min_contour_area}"
                )
                return None

            # Get largest contour
            largest = max(valid_contours, key=cv.contourArea)

            # Calculate centroid using moments
            M = cv.moments(largest)
            if M["m00"] == 0:
                logger.debug(f"{self.name}: Could not calculate centroid (m00=0)")
                return None

            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            location = BallLocation(x=cx, y=cy)
            logger.debug(f"{self.name}: Ball detected at {location.as_tuple()}")
            return location

        except Exception as e:
            logger.error(f"{self.name}: Extraction failed - {e}")
            return None

    def get_debug_image(self, image: np.ndarray) -> np.ndarray | None:
        """Get debug visualization showing detected contours.

        Args:
            image: Binary or edge-detected image used for detection.

        Returns:
            RGB image with contours drawn, or None if no contours found.
        """
        if image is None or image.size == 0:
            return None

        try:
            # Find contours
            contours, _ = cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

            # Convert grayscale to BGR for colored annotations
            if len(image.shape) == 2:
                debug_img = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
            else:
                debug_img = image.copy()

            if not contours:
                return debug_img

            # Filter by area
            valid_contours = [
                c for c in contours if cv.contourArea(c) >= self.min_contour_area
            ]

            # Draw all valid contours
            cv.drawContours(debug_img, valid_contours, -1, (0, 255, 0), 2)

            # Highlight largest contour
            if valid_contours:
                largest = max(valid_contours, key=cv.contourArea)
                cv.drawContours(debug_img, [largest], 0, (0, 0, 255), 3)

                # Draw centroid
                M = cv.moments(largest)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv.circle(debug_img, (cx, cy), 5, (255, 0, 0), -1)

            return debug_img

        except Exception as e:
            logger.error(f"{self.name}: Debug image generation failed - {e}")
            return None
