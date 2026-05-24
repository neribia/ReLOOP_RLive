"""Main ball localization orchestrator."""


import cv2 as cv
import numpy as np

from rlive_common.core.ball_location import BallLocation
from rlive_env.localisation.extractors.base import AbstractBallExtractor
from rlive_env.localisation.extractors import ContourExtractor
from rlive_env.localisation.processors.pipeline import ImagePipeline
from rlive_env.localisation.processors import GrayscaleProcessor, ThresholdProcessor, InvertProcessor, DilationProcessor
from rlive_env.config import config
from rlive_common.utils import get_logger

logger = get_logger(__name__)


class DetectionResult:
    """Result of a detection attempt, containing position and debug images."""

    def __init__(
        self,
        location: BallLocation | None = None,
        debug_image: np.ndarray | None = None,
        extractor_debug_image: np.ndarray | None = None,
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
        extractor: AbstractBallExtractor | None = None,
        pipeline: ImagePipeline | None = None,
        target_width: int = 640,
        target_height: int = 480,
        debug: bool = False,
    ):
        """Initialize the BallLocalisator with dependency injection.

        Args:
            extractor: Extractor instance for ball detection.
                      If None, uses default HoughCircleExtractor.
            pipeline: ImagePipeline for preprocessing.
                     If None, uses default pipeline with standard processors.
            target_width: Fixed width to resize images to. Default 640.
            target_height: Fixed height to resize images to. Default 480.
            debug: If True, shows pipeline and extractor debug images in
                   OpenCV windows after every detection call. Default False.
        """
        self.extractor = extractor or self._create_default_extractor()
        self.pipeline = pipeline or self._create_default_pipeline()
        self.last_result: DetectionResult | None = None
        self.target_width = target_width
        self.target_height = target_height
        self.debug = debug

    @staticmethod
    def _create_default_extractor() -> AbstractBallExtractor:
        """Create default extractor (ContourExtractor).

        Returns:
            AbstractBallExtractor instance.
        """
        return ContourExtractor(
            min_contour_area=config.BALL_MIN_CONTOUR_AREA,
            max_contour_area=config.BALL_MAX_CONTOUR_AREA,
            min_circularity=config.BALL_MIN_CIRCULARITY,
        )

    @staticmethod
    def _create_default_pipeline() -> ImagePipeline:
        """Create default image processing pipeline.

        Returns:
            ImagePipeline with standard processors.
        """
        processors = [
            GrayscaleProcessor(),
            ThresholdProcessor(threshold_value=config.BALL_THRESHOLD_VALUE),
            InvertProcessor(),
            DilationProcessor(
                kernel_size=(config.BALL_DILATION_KERNEL_WIDTH, config.BALL_DILATION_KERNEL_HEIGHT),
                iterations=config.BALL_DILATION_ITERATIONS,
            ),
        ]
        return ImagePipeline(processors)

    def get_position(self, image: np.ndarray) -> BallLocation | None:
        """Get the ball position from an image.

        Executes the pipeline to process the image, then uses the injected
        extractor to detect the ball. Image is resized to a fixed size
        before processing to ensure consistent behavior.

        Args:
            image: RGB image as numpy array.

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
            else:
                logger.debug(f"{self.extractor.name} returned None")
                self.last_result = DetectionResult(
                    location=None,
                    debug_image=processed,
                    extractor_debug_image=extractor_debug_img,
                )

            if self.debug:
                self._show_debug_windows()

            # location is None when else-branch was taken — both paths correct
            return location

        except Exception as e:
            logger.error(f"Ball localization failed: {e}")
            self.last_result = DetectionResult(location=None, debug_image=None)
            return None

    def _show_debug_windows(self) -> None:
        """Display pipeline and extractor debug images in OpenCV windows.

        Called automatically after each detection when ``self.debug`` is True.
        Windows are non-blocking (waitKey(1)).
        """
        if self.last_result is None:
            return
        if self.last_result.debug_image is not None:
            cv.imshow("Localiser: pipeline", self.last_result.debug_image)
        if self.last_result.extractor_debug_image is not None:
            # extractor image is RGB — convert to BGR for correct colours
            bgr = cv.cvtColor(self.last_result.extractor_debug_image, cv.COLOR_RGB2BGR)
            cv.imshow("Localiser: extractor", bgr)
        cv.waitKey(1)

    def get_position_with_debug(self, image: np.ndarray) -> DetectionResult | None:
        """Get ball position and debug information.

        Args:
            image: RGB image as numpy array.

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
