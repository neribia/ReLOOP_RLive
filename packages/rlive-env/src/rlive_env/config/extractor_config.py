"""Extraction configuration for ball localisation - environment variables with defaults."""
import os

# --- Extraction Method ---
EXTRACTION_METHOD: str = os.getenv("BALL_EXTRACTION_METHOD", "hough")
"""
Detection method ('hough' or 'contours').
default: hough
"""

# --- Hough Circle Detection ---
HOUGH_DP: float = float(os.getenv("BALL_HOUGH_DP", "1.0"))
"""
Hough accumulator inverse ratio.
default: 1.0
"""

HOUGH_MIN_DIST: int = int(os.getenv("BALL_HOUGH_MIN_DIST", "50"))
"""
Minimum distance between detected circles.
default: 50
"""

HOUGH_PARAM1: int = int(os.getenv("BALL_HOUGH_PARAM1", "100"))
"""
Hough Canny edge detection threshold.
default: 100
"""

HOUGH_PARAM2: int = int(os.getenv("BALL_HOUGH_PARAM2", "30"))
"""
Hough accumulator threshold.
default: 30
"""

HOUGH_MIN_RADIUS: int = int(os.getenv("BALL_HOUGH_MIN_RADIUS", "10"))
"""
Minimum circle radius.
default: 10
"""

HOUGH_MAX_RADIUS: int = int(os.getenv("BALL_HOUGH_MAX_RADIUS", "100"))
"""
Maximum circle radius.
default: 100
"""

# --- Contour Detection ---
MIN_CONTOUR_AREA: int = int(os.getenv("BALL_MIN_CONTOUR_AREA", "2000"))
"""
Minimum contour area for valid detection.
default: 2000
"""

MAX_CONTOUR_AREA: int | None = int(os.getenv("BALL_MAX_CONTOUR_AREA", "5000"))
"""
Maximum contour area for valid detection. Contours larger than this are rejected
(e.g. arena border, large shadows). Set BALL_MAX_CONTOUR_AREA=0 to disable.
default: 5000
"""

MIN_CIRCULARITY: float = float(os.getenv("BALL_MIN_CIRCULARITY", "0.3"))
"""
Minimum circularity score (0.0–1.0) for a contour to be accepted as the ball.
C = (4π·A) / P²  where A=area, P=perimeter.
  0.0        → disabled (default)
  0.5        → lenient, removes most blobs
  0.6        → good default for real camera footage
  0.75       → strict, stable lighting only
default: 0.0 (disabled)
"""

# --- Logging ---
LOGGING_LEVEL: str = os.getenv("BALL_LOGGING_LEVEL", "INFO")
"""
Logging level (DEBUG, INFO, WARNING, ERROR).
default: INFO
"""
