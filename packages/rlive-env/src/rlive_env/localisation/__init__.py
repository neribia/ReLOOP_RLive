"""Ball localisation module for determining Sphero ball position in images.

This module provides a pluggable architecture for ball detection with support for:
- Classical computer vision methods (Hough circles, contours)
- Future deep learning approaches (YOLO, regression, segmentation)
- Configurable image processing pipelines via dependency injection
- Intelligent fallback strategy when detection fails

Core Components:
    - BallLocalisator: Main orchestrator for ball localization
    - BallLocation: Data class representing ball position
    - Processors: Image processing transformations (Gaussian blur, Canny, etc.)
    - Extractors: Ball detection strategies (Hough, contours, future DL methods)

Quick Start:
    Basic usage with defaults from config:
        from rlive_env.localisation import BallLocalisator, BallLocation
        localiser = BallLocalisator()
        location = localiser.get_position(image)

    With custom extractor:
        from rlive_env.localisation import BallLocalisator
        from rlive_env.localisation.extractors import HoughCircleExtractor

        extractor = HoughCircleExtractor(hough_param2=25)
        localiser = BallLocalisator(extractor=extractor)
        location = localiser.get_position(image)

    With custom pipeline and extractor:
        from rlive_env.localisation import BallLocalisator
        from rlive_env.localisation.processors import ImagePipeline, GrayscaleProcessor
        from rlive_env.localisation.extractors import HoughCircleExtractor

        pipeline = ImagePipeline([GrayscaleProcessor(), ...])
        extractor = HoughCircleExtractor()
        localiser = BallLocalisator(extractor=extractor, pipeline=pipeline)
        location = localiser.get_position(image)
"""

from rlive_env.localisation.ball_location import BallLocation
from rlive_env.localisation.localisator import BallLocalisator, DetectionResult
from rlive_env.localisation.processors.pipeline import ImagePipeline

__all__ = [
    # Core classes
    "BallLocalisator",
    "BallLocation",
    "DetectionResult",
    # Processing
    "ImagePipeline",
]
