"""Polygon approximation-based ball detection extractor."""

import cv2 as cv
import numpy as np

from rlive_common.core.ball_location import BallLocation
from rlive_env.localisation.extractors.base import AbstractBallExtractor
from rlive_env.config import extractor_config
from rlive_common.utils import get_logger

logger = get_logger(__name__)


class PolygonApproxExtractor(AbstractBallExtractor):
    """Extract ball location using polygon approximation.

    Uses cv.approxPolyDP() to approximate contours as polygons,
    then selects contours that are approximately circular (near 8-sided polygons).

    This approach is more robust to noise and irregular contour shapes
    by focusing on overall contour shape rather than exact boundaries.

    Examples:
        Basic usage:
            extractor = PolygonApproxExtractor()
            location = extractor.extract(binary_image)

        Adjust polygon approximation tolerance:
            extractor = PolygonApproxExtractor(epsilon=0.02)
            location = extractor.extract(binary_image)
    """

    def __init__(
        self,
        min_contour_area: int = extractor_config.MIN_CONTOUR_AREA,
        epsilon_factor: float = 0.02,
        target_vertices: int = 8
    ):
        """Initialize polygon approximation extractor.

        Args:
            min_contour_area: Minimum contour area for valid detection.
            epsilon_factor: Approximation accuracy as fraction of contour perimeter.
            target_vertices: Expected number of vertices for circular objects.
        """
        self.min_contour_area = min_contour_area
        self.epsilon_factor = epsilon_factor
        self.target_vertices = target_vertices

    @property
    def name(self) -> str:
        """Return extractor name."""
        return "PolygonApprox"

    def extract(self, image: np.ndarray) -> BallLocation | None:
        """Extract ball location using polygon approximation.

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

            # Filter by area and approximate
            valid_approx = []
            for c in contours:
                area = cv.contourArea(c)
                if area >= self.min_contour_area:
                    epsilon = self.epsilon_factor * cv.arcLength(c, True)
                    approx = cv.approxPolyDP(c, epsilon, True)
                    valid_approx.append((c, approx, area))

            if not valid_approx:
                logger.debug(
                    f"{self.name}: No contours with area >= {self.min_contour_area}"
                )
                return None

            # Select contour with polygon vertices closest to target
            best_contour, best_approx, best_area = min(
                valid_approx,
                key=lambda x: abs(len(x[1]) - self.target_vertices)
            )

            # Calculate centroid from approximated polygon
            M = cv.moments(best_approx)
            if M["m00"] == 0:
                # Fall back to original contour if approximation fails
                M = cv.moments(best_contour)
                if M["m00"] == 0:
                    logger.debug(f"{self.name}: Could not calculate centroid (m00=0)")
                    return None

            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            vertices = len(best_approx)
            logger.debug(
                f"{self.name}: Ball at {(cx, cy)}, vertices={vertices}, area={best_area:.1f}"
            )

            location = BallLocation(x=cx, y=cy)
            return location

        except Exception as e:
            logger.error(f"{self.name}: Extraction failed - {e}")
            return None

    def get_debug_image(self, image: np.ndarray) -> np.ndarray | None:
        """Get debug visualization showing polygon approximations.

        Args:
            image: Binary or edge-detected image.

        Returns:
            RGB image with approximated polygons, or None if failed.
        """
        if image is None or image.size == 0:
            return None

        try:
            # Find contours
            contours, _ = cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

            # Convert to RGB
            if len(image.shape) == 2:
                debug_img = cv.cvtColor(image, cv.COLOR_GRAY2RGB)
            else:
                debug_img = image.copy()

            if not contours:
                return debug_img

            # Approximate and draw all valid contours
            valid_approx = []
            for c in contours:
                area = cv.contourArea(c)
                if area >= self.min_contour_area:
                    epsilon = self.epsilon_factor * cv.arcLength(c, True)
                    approx = cv.approxPolyDP(c, epsilon, True)
                    valid_approx.append((c, approx, area))

            # Draw original contours in green
            for c, _, _ in valid_approx:
                cv.drawContours(debug_img, [c], 0, (0, 255, 0), 2)

            if valid_approx:
                # Highlight best approximation
                best_contour, best_approx, best_area = min(
                    valid_approx,
                    key=lambda x: abs(len(x[1]) - self.target_vertices)
                )

                # Draw approximated polygon in red
                cv.polylines(debug_img, [best_approx], True, (255, 0, 0), 3)

                # Draw centroid in blue
                M = cv.moments(best_approx)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv.circle(debug_img, (cx, cy), 5, (0, 0, 255), -1)

                    vertices = len(best_approx)
                    text = f"Vertices: {vertices}"
                    cv.putText(debug_img, text, (cx + 10, cy - 10),
                              cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            return debug_img

        except Exception as e:
            logger.error(f"{self.name}: Debug image generation failed - {e}")
            return None

