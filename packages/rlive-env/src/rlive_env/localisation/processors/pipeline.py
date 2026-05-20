"""Image processing pipeline orchestration."""

from typing import Callable, Optional

import numpy as np

from rlive_common.utils import get_logger

logger = get_logger(__name__)


class ImagePipeline:
    """Executes a chain of image processing transformations.

    Pure image processing - no detection logic. Takes an image and applies
    a sequence of transformations, returning the processed result.

    Attributes:
        steps: List of processing steps (callable functions or processors).
        intermediate_results: Results from each pipeline stage (if debug=True).

    Examples:
        Create a simple pipeline with grayscale and blur:
            from rlive_env.localisation.processors import (
                GrayscaleProcessor, GaussianBlurProcessor, LaplacianProcessor
            )
            pipeline = ImagePipeline([
                GrayscaleProcessor(),
                GaussianBlurProcessor(kernel_size=9, sigma=2),
                LaplacianProcessor(ksize=3),
            ])
            processed = pipeline.execute(image)

        Use debug mode to inspect intermediate stages:
            processed = pipeline.execute(image, debug=True)
            stages = pipeline.get_intermediate_results()
            # stages[0] = original image
            # stages[1] = grayscale
            # stages[2] = blurred
            # stages[3] = laplacian

        Use with callable functions (backward compatible):
            pipeline = ImagePipeline([
                lambda img: cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
                lambda img: cv2.GaussianBlur(img, (9, 9), 2),
            ])
            processed = pipeline.execute(image)
    """

    def __init__(self, steps: list[Callable[[np.ndarray], np.ndarray] | object]):
        """Initialize pipeline with processing steps.

        Args:
            steps: List of callable functions or processor objects that transform images.
                  Each should take np.ndarray and return np.ndarray.
                  Can also be processor objects with a .process() method.
        """
        self.steps = steps
        self.intermediate_results: list[np.ndarray] = []

    def execute(self, image: np.ndarray, debug: bool = False) -> Optional[np.ndarray]:
        """Execute all pipeline steps on the image.

        Args:
            image: Input image (RGB).
            debug: If True, store intermediate results for debugging.

        Returns:
            Processed image after all steps applied, or None if error occurs.
        """
        if debug:
            self.intermediate_results = [image.copy()]

        result = image
        for i, step in enumerate(self.steps):
            try:
                # Handle both callable functions and processor objects
                if hasattr(step, "process"):
                    # It's a processor object
                    result = step.process(result)
                else:
                    # It's a callable function
                    result = step(result)

                if debug and result is not None:
                    self.intermediate_results.append(result.copy())
            except Exception as e:
                logger.error(f"Error in pipeline step {i}: {e}")
                return None

        return result

    def get_intermediate_results(self) -> list[np.ndarray]:
        """Get intermediate results from last debug execution.

        Returns:
            List of images at each pipeline stage (if debug=True was used).
        """
        return self.intermediate_results

