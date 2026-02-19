"""Moments-based ball detection extractor."""

import cv2 as cv
import numpy as np

from rlive_env.localisation.ball_location import BallLocation
from rlive_env.localisation.extractors.base import AbstractBallExtractor
from rlive_env.config import extractor_config
from rlive_common.utils import get_logger

logger = get_logger(__name__)


class MomentsExtractor(AbstractBallExtractor):
    """Extract ball location using image moments analysis.

    Uses cv.moments() to compute spatial moments of the contour and calculate
    geometric properties like centroid, area, and circularity.

    This extractor is optimized for circular or near-circular objects
    and provides additional geometric information.

    Examples:
        Basic usage:
            extractor = MomentsExtractor()
            location = extractor.extract(binary_image)

        Custom moment analysis parameters:
            extractor = MomentsExtractor(min_contour_area=200)
            location = extractor.extract(binary_image)
    """

    def __init__(
        self,
        min_contour_area: int = extractor_config.MIN_CONTOUR_AREA,
        min_circularity: float = 0.4,
    ):
        """Initialize moments-based extractor.

        Args:
            min_contour_area: Minimum contour area for valid detection.
            min_circularity: Minimum circularity (0.0-1.0) for a contour to be
                            considered a ball candidate. 1.0 = perfect circle.
                            Box corners typically have circularity < 0.3.
                            Default 0.4 rejects rectangles and lines.
        """
        self.min_contour_area = min_contour_area
        self.min_circularity = min_circularity

    @property
    def name(self) -> str:
        """Return extractor name."""
        return "Moments"

    def _calculate_circularity(self, contour: np.ndarray, area: float) -> float:
        """Calculate circularity of contour (0=line, 1=circle).

        Args:
            contour: Contour array.
            area: Contour area.

        Returns:
            Circularity value between 0 and 1.
        """
        if area == 0:
            return 0.0

        perimeter = cv.arcLength(contour, True)
        if perimeter == 0:
            return 0.0

        circularity = 4 * np.pi * area / (perimeter ** 2)
        return min(circularity, 1.0)

    def extract(self, image: np.ndarray) -> BallLocation | None:
        """Extract ball location using moment-based analysis.

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

            # Filter by circularity to reject non-round shapes (box corners, lines)
            circular_contours = [
                c for c in valid_contours
                if self._calculate_circularity(c, cv.contourArea(c)) >= self.min_circularity
            ]

            if not circular_contours:
                logger.debug(
                    f"{self.name}: No contours with circularity >= {self.min_circularity}"
                )
                return None

            # Select contour with best circularity (most likely the ball)
            best_contour = max(
                circular_contours,
                key=lambda c: self._calculate_circularity(c, cv.contourArea(c))
            )

            # Calculate moments
            M = cv.moments(best_contour)

            if M["m00"] == 0:
                logger.debug(f"{self.name}: Could not calculate moments (m00=0)")
                return None

            # Calculate centroid
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            # Log moment information
            area = cv.contourArea(best_contour)
            circularity = self._calculate_circularity(best_contour, area)
            logger.debug(
                f"{self.name}: Ball at {(cx, cy)}, area={area:.1f}, circularity={circularity:.3f}"
            )

            location = BallLocation(x=cx, y=cy)
            return location

        except Exception as e:
            logger.error(f"{self.name}: Extraction failed - {e}")
            return None

    def get_debug_image(self, image: np.ndarray) -> np.ndarray | None:
        """Get debug visualization with moment analysis results.

        Args:
            image: Binary or edge-detected image.

        Returns:
            RGB image with contours and moment information, or None if failed.
        """
        if image is None or image.size == 0:
            return None

        try:
            # Find contours
            contours, _ = cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

            # Convert to BGR
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

            if not valid_contours:
                return debug_img

            # Draw all valid contours (area-filtered) in gray
            cv.drawContours(debug_img, valid_contours, -1, (128, 128, 128), 1)

            # Filter by circularity
            circular_contours = [
                c for c in valid_contours
                if self._calculate_circularity(c, cv.contourArea(c)) >= self.min_circularity
            ]

            # Draw circular contours in green
            cv.drawContours(debug_img, circular_contours, -1, (0, 255, 0), 2)

            if not circular_contours:
                return debug_img

            # Highlight best contour by circularity
            best_contour = max(
                circular_contours,
                key=lambda c: self._calculate_circularity(c, cv.contourArea(c))
            )
            cv.drawContours(debug_img, [best_contour], 0, (0, 0, 255), 3)

            # Draw centroid and moment info
            M = cv.moments(best_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv.circle(debug_img, (cx, cy), 5, (255, 0, 0), -1)

                area = cv.contourArea(best_contour)
                circularity = self._calculate_circularity(best_contour, area)
                text = f"Circ: {circularity:.2f}"
                cv.putText(debug_img, text, (cx + 10, cy - 10),
                          cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            return debug_img

        except Exception as e:
            logger.error(f"{self.name}: Debug image generation failed - {e}")
            return None

