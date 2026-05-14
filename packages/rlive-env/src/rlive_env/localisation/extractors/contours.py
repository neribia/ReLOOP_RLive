"""Contour-based ball detection extractor."""

import cv2 as cv
import numpy as np

from rlive_common.core.ball_location import BallLocation
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

    def __init__(
        self,
        min_contour_area: int = extractor_config.MIN_CONTOUR_AREA,
        min_circularity: float = 0.0,
    ):
        """Initialize contour extractor with defaults from config module.

        Args:
            min_contour_area: Minimum contour area in pixels² for a contour to
                be considered a valid detection candidate.  Contours smaller
                than this value are discarded before any other filtering.

            min_circularity: Minimum circularity score required for a contour
                to be accepted as the ball.

                Circularity is defined as::

                    C = (4π · A) / P²

                where A is the contour area and P is its perimeter.  The value
                lies in [0.0, 1.0]:

                * ``1.0`` — perfect circle
                * ``0.7 – 0.95`` — Sphero BOLT+ under normal conditions
                * ``0.05 – 0.35`` — shadow edges / elongated blobs (rejected)

                Recommended starting values:

                ================  ===============================================
                Value             When to use
                ================  ===============================================
                ``0.0`` (default) Disabled; keeps original behaviour
                ``0.5``           Lenient; removes most blobs, keeps blurry ball
                ``0.6``           Good default for real camera footage
                ``0.75``          Strict; use when lighting is stable
                ================  ===============================================
        """
        self.min_contour_area = min_contour_area
        self.min_circularity = min_circularity
        self.last_circularity: float | None = None  # C of the selected contour, updated after each extract()
        self.last_area: float | None = None          # pixel² area of the selected contour, updated after each extract()

    @property
    def name(self) -> str:
        """Return extractor name."""
        return "Contour"

    @staticmethod
    def _circularity(contour) -> float:
        """Compute the circularity score of a contour.

        Circularity is a dimensionless shape descriptor defined as:

            C = (4π · A) / P²

        where
            A  = contour area       (cv.contourArea)
            P  = contour perimeter  (cv.arcLength, closed)
            π  = 3.14159…

        Properties
        ----------
        - C = 1.0  →  perfect circle (theoretical maximum)
        - C → 0.0  →  increasingly elongated or irregular shape
        - C is scale-invariant: doubling the object size leaves C unchanged
        - C is rotation-invariant

        Typical values observed in this system
        ---------------------------------------
        =========================================  ===========
        Shape                                      C (approx.)
        =========================================  ===========
        Perfect circle                             1.00
        Sphero BOLT+ (real camera, good lighting)  0.70 – 0.95
        Sphero BOLT+ (motion blur / compression)   0.50 – 0.70
        Shadow edge blob / irregular region        0.05 – 0.35
        Rectangle / square                        ~0.79
        =========================================  ===========

        Returns
        -------
        float
            Circularity in the range [0.0, 1.0].
            Returns 0.0 if the perimeter is zero (degenerate contour).
        """
        area = cv.contourArea(contour)
        perimeter = cv.arcLength(contour, True)
        if perimeter == 0:
            return 0.0
        return (4 * np.pi * area) / (perimeter ** 2)

    def _filter_contours(self, contours) -> list:
        """Filter contours by area and circularity."""
        valid = []
        for c in contours:
            if cv.contourArea(c) < self.min_contour_area:
                continue
            if self.min_circularity > 0.0 and self._circularity(c) < self.min_circularity:
                logger.debug(
                    f"{self.name}: Contour rejected (circularity={self._circularity(c):.2f} "
                    f"< min={self.min_circularity:.2f})"
                )
                continue
            valid.append(c)
        return valid

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

            # Filter by area and circularity
            valid_contours = self._filter_contours(contours)

            if not valid_contours:
                logger.debug(
                    f"{self.name}: No contours with area >= {self.min_contour_area}"
                )
                return None

            # Get largest contour
            largest = max(valid_contours, key=cv.contourArea)
            self.last_circularity = round(self._circularity(largest), 3)
            self.last_area = round(cv.contourArea(largest), 1)

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

            # Draw ALL contours (area >= min) in orange so rejected ones are still visible
            area_filtered = [c for c in contours if cv.contourArea(c) >= self.min_contour_area]
            cv.drawContours(debug_img, area_filtered, -1, (0, 140, 255), 1)

            # Filter by area and circularity
            valid_contours = self._filter_contours(contours)

            # Draw accepted contours in green
            cv.drawContours(debug_img, valid_contours, -1, (0, 255, 0), 2)

            # Draw C value on every area-filtered contour (green = accepted, orange = rejected)
            for c in area_filtered:
                circ = self._circularity(c)
                M = cv.moments(c)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    accepted = circ >= self.min_circularity or self.min_circularity == 0.0
                    colour = (0, 220, 0) if accepted else (0, 140, 255)
                    area = int(cv.contourArea(c))
                    cv.putText(debug_img, f"C={circ:.2f}", (cx - 28, cy + 8),
                               cv.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1)
                    cv.putText(debug_img, f"A={area}", (cx - 28, cy + 22),
                               cv.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1)

            # Highlight the selected (largest valid) contour in red
            if valid_contours:
                largest = max(valid_contours, key=cv.contourArea)
                cv.drawContours(debug_img, [largest], 0, (0, 0, 255), 3)

                M = cv.moments(largest)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv.circle(debug_img, (cx, cy), 5, (255, 0, 0), -1)

            return debug_img

        except Exception as e:
            logger.error(f"{self.name}: Debug image generation failed - {e}")
            return None