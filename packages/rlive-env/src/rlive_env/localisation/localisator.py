"""Main ball localization orchestrator."""

from typing import Optional

import cv2 as cv
import numpy as np

from rlive_env.localisation.ball_location import BallLocation
from rlive_env.localisation.extractors.base import AbstractBallExtractor
from rlive_env.localisation.extractors import HoughCircleExtractor, ContourExtractor
from rlive_env.localisation.processors.pipeline import ImagePipeline
from rlive_env.localisation.processors import (
    GrayscaleProcessor,
    GaussianBlurProcessor,
    LaplacianProcessor,
    ThresholdProcessor,
    DilationProcessor,
)
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
    ):
        """Initialize the BallLocalisator with dependency injection.

        Args:
            extractor: Extractor instance for ball detection.
                      If None, uses default HoughCircleExtractor.
            pipeline: ImagePipeline for preprocessing.
                     If None, uses default pipeline with standard processors.
        """
        self.extractor = extractor or self._create_default_extractor()
        self.pipeline = pipeline or self._create_default_pipeline()
        self.last_result: Optional[DetectionResult] = None

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
            GrayscaleProcessor(),
            GaussianBlurProcessor(kernel_size=5, sigma=2.0),
            LaplacianProcessor(ksize=3),
            ThresholdProcessor(threshold_value=30, max_value=255),
            DilationProcessor(kernel_size=(5, 5), iterations=2),
        ]
        return ImagePipeline(processors)

    def get_position(self, image: np.ndarray) -> Optional[BallLocation]:
        """Get the ball position from an image.

        Executes the pipeline to process the image, then uses the injected
        extractor to detect the ball.

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
        if image is None or image.size == 0:
            logger.warning("Empty image provided")
            self.last_result = DetectionResult(location=None, debug_image=None)
            return None

        # Execute processing pipeline
        processed = self.pipeline.execute(image)

        if processed is None:
            logger.error("Pipeline execution failed")
            self.last_result = DetectionResult(location=None, debug_image=None)
            return None

        # Extract ball position
        try:
            logger.debug(f"Extracting ball using {self.extractor.name}...")
            location = self.extractor.extract(processed)

            # Get extractor debug image
            extractor_debug_img = self.extractor.get_debug_image(processed)

            if location is not None:
                logger.info(f"Ball located at {location.as_tuple()}")
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
            logger.error(f"{self.extractor.name} extraction failed: {e}")
            self.last_result = DetectionResult(location=None, debug_image=processed)
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

    def annotate_image(
        self,
        image: np.ndarray,
        ball_location: Optional[BallLocation] = None,
        goal_position: Optional[tuple[int, int]] = None,
        goal_radius: int = 50,
    ) -> np.ndarray:
        """Draw detection results on image for visualization. Position of the ball and the distance.

        Args:
            image: Input image to annotate.
            ball_location: Detected ball position (if any).
            goal_position: Goal position to draw (if any).
            goal_radius: Radius of goal circle.

        Returns:
            Annotated image copy.

        Examples:
            Annotate image with results:
                location = localiser.get_position(image)
                annotated = localiser.annotate_image(
                    image,
                    ball_location=location,
                    goal_position=(320, 240),
                    goal_radius=50
                )
                cv2.imshow("Annotated", annotated)
        """
        annotated = image.copy()

        # Draw ball if detected
        if ball_location is not None:
            pos = ball_location.as_tuple()
            cv.circle(annotated, pos, 15, (255, 0, 0), 2)
            cv.circle(annotated, pos, 5, (255, 0, 0), -1)

            # Draw distance line to goal if both available
            if goal_position is not None:
                cv.line(annotated, pos, goal_position, (255, 255, 0), 1)
                distance = ball_location.distance_to(goal_position)
                mid_x = (pos[0] + goal_position[0]) // 2
                mid_y = (pos[1] + goal_position[1]) // 2
                cv.putText(
                    annotated,
                    f"{distance:.1f}px",
                    (mid_x, mid_y - 10),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 0),
                    1,
                )

        return annotated

