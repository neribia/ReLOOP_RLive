"""Ball extraction strategies and utilities."""

from rlive_env.localisation.extractors.base import AbstractBallExtractor
from rlive_env.localisation.extractors.hough import HoughCircleExtractor
from rlive_env.localisation.extractors.contours import ContourExtractor
from rlive_env.localisation.extractors.moments import MomentsExtractor
from rlive_env.localisation.extractors.polygon_approx import PolygonApproxExtractor

# Import stubs for future deep learning implementations
try:
    from rlive_env.localisation.extractors.yolo import YOLOv8Extractor
except NotImplementedError:
    YOLOv8Extractor = None

__all__ = [
    "AbstractBallExtractor",
    "HoughCircleExtractor",
    "ContourExtractor",
    "MomentsExtractor",
    "PolygonApproxExtractor",
    "YOLOv8Extractor"
]

