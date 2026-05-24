"""Mask intersection processor — combines two binary masks with bitwise AND."""

import cv2 as cv
import numpy as np

from rlive_env.localisation.processors.base import AbstractProcessor
from rlive_env.localisation.processors.pipeline import ImagePipeline
from rlive_common.utils import get_logger

logger = get_logger(__name__)

_NDIM_COLOR = 3  # number of dimensions for a colour (3-channel) image


class MaskIntersectionProcessor(AbstractProcessor):
    """Compute the intersection (bitwise AND) of two independently produced masks.

    Each sub-pipeline receives the **same original RGB image** that was passed
    into this processor.  Both sub-pipelines must produce a single-channel
    (grayscale / binary) image of equal size.  Only pixels that are non-zero in
    *both* masks survive in the output.

    This is useful for combining complementary cues, e.g.:
      • A grayscale-threshold mask (bright pixels)   AND
      • An HSV color-range mask  (correctly-coloured pixels)

    so that only bright *and* correctly-coloured pixels reach the extractor,
    reducing false positives from bright non-target objects.

    Attributes:
        pipeline_a: First sub-pipeline (e.g. Grayscale → Threshold → Invert).
        pipeline_b: Second sub-pipeline (e.g. HSV with apply_mask=True).

    Examples:
        Grayscale-threshold AND HSV color intersection:
            from rlive_env.localisation.processors import (
                GrayscaleProcessor, ThresholdProcessor, InvertProcessor,
                HSVProcessor, ImagePipeline, MaskIntersectionProcessor,
                DilationProcessor,
            )

            gray_pipeline = ImagePipeline([
                GrayscaleProcessor(),
                ThresholdProcessor(threshold_value=80),
                InvertProcessor(),
            ])
            hsv_pipeline = ImagePipeline([
                HSVProcessor(
                    lower_hue=10, upper_hue=25,
                    lower_sat=80, upper_sat=255,
                    lower_val=80, upper_val=255,
                    apply_mask=True,
                ),
            ])

            processor = MaskIntersectionProcessor(gray_pipeline, hsv_pipeline)

            # Use as a single step inside a main pipeline
            main_pipeline = ImagePipeline([
                processor,
                DilationProcessor(kernel_size=(5, 5), iterations=2),
            ])
            result = main_pipeline.execute(rgb_image)
    """

    def __init__(self, pipeline_a: ImagePipeline, pipeline_b: ImagePipeline):
        """Initialize the intersection processor.

        Args:
            pipeline_a: Sub-pipeline that produces binary mask A from the RGB image.
            pipeline_b: Sub-pipeline that produces binary mask B from the RGB image.
        """
        self.pipeline_a = pipeline_a
        self.pipeline_b = pipeline_b

    @property
    def name(self) -> str:
        """Return processor name."""
        return "MaskIntersection"

    def process(self, image: np.ndarray) -> np.ndarray:
        """Run both sub-pipelines on the input image and AND their results.

        Args:
            image: Input RGB image.

        Returns:
            Binary mask that is the bitwise AND of mask_a and mask_b.
            Returns an all-black mask if either sub-pipeline fails.

        Raises:
            ValueError: If the two masks have incompatible shapes.
        """
        if image is None or image.size == 0:
            logger.warning(f"{self.name}: Empty image received")
            return np.zeros(image.shape[:2], dtype=np.uint8) if image is not None else np.zeros((1, 1), dtype=np.uint8)

        mask_a = self.pipeline_a.execute(image.copy())
        if mask_a is None:
            logger.error(f"{self.name}: pipeline_a returned None — using empty mask")
            mask_a = np.zeros(image.shape[:2], dtype=np.uint8)

        mask_b = self.pipeline_b.execute(image.copy())
        if mask_b is None:
            logger.error(f"{self.name}: pipeline_b returned None — using empty mask")
            mask_b = np.zeros(image.shape[:2], dtype=np.uint8)

        # Ensure both masks are single-channel
        if mask_a.ndim == _NDIM_COLOR:
            mask_a = cv.cvtColor(mask_a, cv.COLOR_RGB2GRAY)
        if mask_b.ndim == _NDIM_COLOR:
            mask_b = cv.cvtColor(mask_b, cv.COLOR_RGB2GRAY)

        if mask_a.shape != mask_b.shape:
            raise ValueError(
                f"{self.name}: mask shapes do not match — "
                f"mask_a={mask_a.shape}, mask_b={mask_b.shape}"
            )

        intersection = cv.bitwise_and(mask_a, mask_b)
        logger.debug(
            f"{self.name}: non-zero pixels — "
            f"A={cv.countNonZero(mask_a)}, "
            f"B={cv.countNonZero(mask_b)}, "
            f"AND={cv.countNonZero(intersection)}"
        )
        return intersection




