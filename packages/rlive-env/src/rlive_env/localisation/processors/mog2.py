"""MOG2 background subtraction processor."""

import cv2 as cv
import numpy as np

from rlive_env.localisation.processors.base import AbstractProcessor


class MOG2BackgroundSubtractorProcessor(AbstractProcessor):
    """Background subtraction using MOG2 (Mixture of Gaussians) algorithm.

    MOG2 (Mixture of Gaussians v2) is a robust background subtraction method
    that models the background as a mixture of Gaussian distributions. It can
    adapt to gradual scene changes and is effective for detecting moving objects
    against static or gradually changing backgrounds.

    This processor maintains state between frames, so it should be reused
    for the same video/stream sequence.

    Examples:
        Basic usage:
            processor = MOG2BackgroundSubtractorProcessor()
            fg_mask = processor.process(frame)

        With custom detection thresholds:
            processor = MOG2BackgroundSubtractorProcessor(
                detect_shadows=False,
                var_threshold=16.0,
                complexity_reduction_threshold=0.05
            )
            fg_mask = processor.process(frame)

        In a pipeline:
            pipeline = ImagePipeline([
                MOG2BackgroundSubtractorProcessor(),
                DilationProcessor(kernel_size=(5, 5), iterations=2),
            ])
    """

    def __init__(
        self,
        detect_shadows: bool = True,
        var_threshold: float = 16.0,
        complexity_reduction_threshold: float = 0.05,
    ):
        """Initialize MOG2 background subtractor processor.

        Args:
            detect_shadows: If True, the algorithm also detects shadows.
                          If False, only foreground objects are detected.
            var_threshold: Variance threshold for pixel classification.
                          Higher values = more pixels classified as foreground.
                          Range: 1-500 (default: 16)
            complexity_reduction_threshold: Complexity reduction threshold for
                                           background model update.
                                           Range: 0.0-1.0 (default: 0.05)

        Notes:
            - detect_shadows adds computational cost but improves detection
            - var_threshold should be tuned based on your specific scenario
            - Higher var_threshold = more sensitive to changes
        """
        self.detect_shadows = detect_shadows
        self.var_threshold = var_threshold
        self.complexity_reduction_threshold = complexity_reduction_threshold

        # Create the MOG2 background subtractor
        self.bg_subtractor = cv.createBackgroundSubtractorMOG2(
            detectShadows=detect_shadows
        )

        # Set parameters
        self.bg_subtractor.setVarThreshold(var_threshold)
        self.bg_subtractor.setComplexityReductionThreshold(complexity_reduction_threshold)

        self.frame_count = 0

    @property
    def name(self) -> str:
        """Return processor name."""
        return "MOG2BackgroundSubtractor"

    def process(self, image: np.ndarray) -> np.ndarray:
        """Apply MOG2 background subtraction to extract foreground mask.

        Args:
            image: Input frame (RGB color image or grayscale).

        Returns:
            Binary foreground mask where:
            - White (255): Foreground pixels (moving objects)
            - Black (0): Background pixels (static scene)
            - Gray (127): Shadow pixels (only if detect_shadows=True)
        """
        self.frame_count += 1

        try:
            # Apply background subtraction
            fg_mask = self.bg_subtractor.apply(image)

            return fg_mask

        except Exception as e:
            from rlive_common.utils import get_logger
            logger = get_logger(__name__)
            logger.error(f"{self.name}: Process failed - {e}")
            # Return black image on error
            return np.zeros(image.shape[:2], dtype=np.uint8)

    def reset(self) -> None:
        """Reset the background model.

        Use this when switching to a different scene or need to restart
        background learning.
        """
        self.bg_subtractor = cv.createBackgroundSubtractorMOG2(
            detectShadows=self.detect_shadows
        )
        self.bg_subtractor.setVarThreshold(self.var_threshold)
        self.bg_subtractor.setComplexityReductionThreshold(self.complexity_reduction_threshold)
        self.frame_count = 0

    def set_var_threshold(self, threshold: float) -> None:
        """Update variance threshold.

        Args:
            threshold: New variance threshold (1-500).
                      Higher = more sensitive to changes.
        """
        if not 1 <= threshold <= 500:
            raise ValueError("var_threshold must be between 1 and 500")
        self.var_threshold = threshold
        self.bg_subtractor.setVarThreshold(threshold)

    def set_complexity_reduction_threshold(self, threshold: float) -> None:
        """Update complexity reduction threshold.

        Args:
            threshold: New threshold (0.0-1.0).
                      Controls background model update speed.
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("complexity_reduction_threshold must be between 0.0 and 1.0")
        self.complexity_reduction_threshold = threshold
        self.bg_subtractor.setComplexityReductionThreshold(threshold)

    def get_background_model(self) -> np.ndarray | None:
        """Get the current background model image.

        Returns:
            The average background image learned by the model (BGR format),
            or None if unavailable.
            Useful for debugging/visualization.
        """
        try:
            return self.bg_subtractor.getBackgroundImage()
        except Exception as e:
            from rlive_common.utils import get_logger
            logger = get_logger(__name__)
            logger.warning(f"{self.name}: Could not retrieve background model - {e}")
            return None

    def get_frame_count(self) -> int:
        """Get number of frames processed.

        Returns:
            Number of frames processed since initialization or last reset.
        """
        return self.frame_count

    def save_model(self, path: str) -> None:
        """Save the learned background model to a file.

        Saves the background image so it can be reloaded later without
        having to re-learn from scratch.

        Args:
            path: File path to save the model image (e.g. 'bg_model.png').

        Raises:
            RuntimeError: If no background model is available yet.
        """
        bg = self.get_background_model()
        if bg is None:
            raise RuntimeError(
                "No background model available. "
                "Process at least ~30 frames before saving."
            )
        cv.imwrite(path, bg)

    def load_model(self, path: str, warmup_iterations: int = 50) -> None:
        """Load a previously saved background model from a file.

        Feeds the saved background image into the subtractor multiple times
        so it learns it as the stable background. This avoids the warm-up
        period when restarting.

        Args:
            path: File path to the saved background image.
            warmup_iterations: Number of times to feed the image to the model.
                              More iterations = more stable model.
                              Default 50 is usually sufficient.

        Raises:
            FileNotFoundError: If the model file does not exist.
        """
        import os
        if not os.path.exists(path):
            raise FileNotFoundError(f"Background model file not found: {path}")

        bg_image = cv.imread(path)
        if bg_image is None:
            raise RuntimeError(f"Failed to read background model: {path}")

        # cv.imread returns BGR — convert to RGB to match pipeline convention
        bg_image = cv.cvtColor(bg_image, cv.COLOR_BGR2RGB)

        # Reset the subtractor
        self.reset()

        # Feed the saved background image repeatedly so the model learns it
        for _ in range(warmup_iterations):
            self.bg_subtractor.apply(bg_image)

        self.frame_count = warmup_iterations


