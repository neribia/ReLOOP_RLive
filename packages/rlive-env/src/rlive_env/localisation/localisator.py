"""Main ball localization orchestrator."""

from typing import Optional

import cv2 as cv
import numpy as np

from rlive_common.core.ball_location import BallLocation
from rlive_env.localisation.extractors.base import AbstractBallExtractor
from rlive_env.localisation.extractors import *
from rlive_env.localisation.processors.pipeline import ImagePipeline
from rlive_env.localisation.processors import *
from rlive_common.utils import get_logger

logger = get_logger(__name__)


class DetectionResult:
    """Result of a detection attempt, containing position and debug images."""

    def __init__(
        self,
        location: Optional[BallLocation] = None,
        debug_image: Optional[np.ndarray] = None,
        extractor_debug_image: Optional[np.ndarray] = None,
    ):
        """Initialize detection result.

        Args:
            location: Detected ball location, or None if not found.
            debug_image: Debug image from processing pipeline.
            extractor_debug_image: Debug image from extractor.
        """
        self.location = location
        self.debug_image = debug_image
        self.extractor_debug_image = extractor_debug_image


class BallLocalisator:
    """Localizes the Sphero ball in images using injected dependencies.

    Combines image processing pipeline with a single extraction strategy.
    Uses dependency injection for flexible configuration without config objects.

    Attributes:
        extractor: Injected extractor instance for ball detection.
        pipeline: Injected ImagePipeline for preprocessing.
        last_result: Result from last detection attempt.

    Examples:
        Basic usage with default dependencies:
            localiser = BallLocalisator()
            location = localiser.get_position(image)

        Inject custom extractor:
            from rlive_env.localisation.extractors import HoughCircleExtractor
            from rlive_env.localisation.config import ExtractionConfig

            config = ExtractionConfig(hough_param2=25)
            extractor = HoughCircleExtractor(config)
            localiser = BallLocalisator(extractor=extractor)
            location = localiser.get_position(image)

        Inject custom pipeline:
            from rlive_env.localisation.processors import ImagePipeline

            pipeline = ImagePipeline([
                GrayscaleProcessor(),
                GaussianBlurProcessor(kernel_size=11),
                LaplacianProcessor(),
            ])
            localiser = BallLocalisator(pipeline=pipeline)
            location = localiser.get_position(image)

        Both custom:
            localiser = BallLocalisator(
                extractor=my_extractor,
                pipeline=my_pipeline
            )
            location = localiser.get_position(image)
    """

    def __init__(
        self,
        extractor: Optional[AbstractBallExtractor] = None,
        pipeline: Optional[ImagePipeline] = None,
        target_width: int = 640,
        target_height: int = 480,
    ):
        """Initialize the BallLocalisator with dependency injection.

        Args:
            extractor: Extractor instance for ball detection.
                      If None, uses default HoughCircleExtractor.
            pipeline: ImagePipeline for preprocessing.
                     If None, uses default pipeline with standard processors.
            target_width: Fixed width to resize images to. Default 640.
            target_height: Fixed height to resize images to. Default 480.
        """
        self.extractor = extractor or self._create_default_extractor()
        self.pipeline = pipeline or self._create_default_pipeline()
        self.last_result: Optional[DetectionResult] = None
        self.target_width = target_width
        self.target_height = target_height

    @staticmethod
    def _create_default_extractor() -> AbstractBallExtractor:
        """Create default extractor (ContourExtractor).

        Returns:
            AbstractBallExtractor instance.
        """
        return ContourExtractor()

    @staticmethod
    def _create_default_pipeline() -> ImagePipeline:
        """Create default image processing pipeline.

        Returns:
            ImagePipeline with standard processors.
        """
        processors = [
            HSVProcessor(
                lower_hue=90, upper_hue=130,  # blue hue range
                lower_sat=50, upper_sat=255,  # require some color (not gray)
                lower_val=50, upper_val=255,  # require some brightness (not black)
                apply_mask=True,
            ),
            DilationProcessor(kernel_size=(7, 7), iterations=3),
        ]
        return ImagePipeline(processors)

    def get_position(self, image: np.ndarray) -> Optional[BallLocation]:
        """Get the ball position from an image.

        Executes the pipeline to process the image, then uses the injected
        extractor to detect the ball. Image is resized to a fixed size
        before processing to ensure consistent behavior.

        Args:
            image: BGR or RGB image as numpy array.

        Returns:
            BallLocation if ball is detected, None otherwise.

        Examples:
            Get ball position from image:
                localiser = BallLocalisator()
                location = localiser.get_position(image)
                if location:
                    print(f"Ball at: ({location.x}, {location.y})")

            Process video stream:
                cap = cv2.VideoCapture(0)
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    location = localiser.get_position(frame)
                    if location:
                        cv2.circle(frame, location.as_tuple(), 5, (0, 255, 0), -1)
                    cv2.imshow("Ball Localization", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        """
        try:
            # Validate input
            if image is None or image.size == 0:
                logger.warning("Empty image provided")
                raise ValueError("Empty image provided")

            # Resize image to fixed size for consistent processing
            resized = cv.resize(image, (self.target_width, self.target_height))
            logger.debug(f"Image resized to {self.target_width}x{self.target_height}")

            # Execute processing pipeline
            processed = self.pipeline.execute(resized)

            if processed is None:
                logger.error("Pipeline execution failed")
                raise RuntimeError("Pipeline execution failed")

            # Extract ball position
            logger.debug(f"Extracting ball using {self.extractor.name}...")
            location = self.extractor.extract(processed)

            # Get extractor debug image
            extractor_debug_img = self.extractor.get_debug_image(processed)

            if location is not None:
                logger.debug(f"Ball located at {location.as_tuple()}")
                self.last_result = DetectionResult(
                    location=location,
                    debug_image=processed,
                    extractor_debug_image=extractor_debug_img,
                )
                return location
            else:
                logger.debug(f"{self.extractor.name} returned None")
                self.last_result = DetectionResult(
                    location=None,
                    debug_image=processed,
                    extractor_debug_image=extractor_debug_img,
                )
                return None

        except Exception as e:
            logger.error(f"Ball localization failed: {e}")
            self.last_result = DetectionResult(location=None, debug_image=None)
            return None

    def get_position_with_debug(self, image: np.ndarray) -> DetectionResult:
        """Get ball position and debug information.

        Args:
            image: BGR or RGB image as numpy array.

        Returns:
            DetectionResult with location and debug images.

        Examples:
            Get position with debug images:
                localiser = BallLocalisator()
                result = localiser.get_position_with_debug(image)
                if result.location:
                    print(f"Ball at: {result.location.as_tuple()}")
                    cv2.imshow("Debug", result.debug_image)
                    if result.extractor_debug_image is not None:
                        cv2.imshow("Extractor Debug", result.extractor_debug_image)
        """
        self.get_position(image)

        # Add extractor debug image as last element to pipeline debug list
        if self.last_result and self.last_result.extractor_debug_image is not None:
            self.pipeline.intermediate_results.append(self.last_result.extractor_debug_image)

        return self.last_result
