"""YOLOv8 ball detection extractor (stub for future implementation).

This module provides a placeholder for YOLOv8-based ball detection.
When implemented, it should:
1. Load a pre-trained or custom YOLOv8 model
2. Run inference on input images
3. Extract ball bounding box and return its center

Example implementation outline (future):
    class YOLOv8Extractor(AbstractBallExtractor):
        def __init__(self, model_path: str = "ball_detector.pt", ...):
            from ultralytics import YOLO
            self.model = YOLO(model_path)

        def extract(self, image: np.ndarray) -> Optional[BallLocation]:
            results = self.model(image)
            if results[0].boxes:
                x1, y1, x2, y2 = results[0].boxes[0].xyxy[0]
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                return BallLocation(x=center_x, y=center_y)
            return None
"""

from typing import Optional

import numpy as np

from rlive_env.localisation.ball_location import BallLocation
from rlive_env.localisation.extractors.base import AbstractBallExtractor


class YOLOv8Extractor(AbstractBallExtractor):
    """YOLOv8-based ball detection extractor (stub).

    This is a placeholder for future deep learning implementation.
    It demonstrates the required interface and usage patterns.

    Note:
        - Requires: ultralytics, torch (install separately to avoid adding as hard dependency)
        - Model: Can use pre-trained weights or custom fine-tuned model

    Examples:
        Usage once implemented:
            extractor = YOLOv8Extractor(model_path="ball_detector.pt")
            location = extractor.extract(image)

        In fallback chain:
            localiser = BallLocalisator(extractors=[
                HoughCircleExtractor(),  # Try classical first
                YOLOv8Extractor(),       # Fall back to deep learning
            ])
    """

    def __init__(self, model_path: str = "ball_detector.pt", **kwargs):
        """Initialize YOLOv8 extractor (stub).

        Args:
            model_path: Path to YOLOv8 model weights.
            **kwargs: Additional arguments for model initialization.

        Raises:
            NotImplementedError: This is a stub; not yet implemented.
        """
        raise NotImplementedError(
            "YOLOv8Extractor is not yet implemented. "
            "Install ultralytics (pip install ultralytics) and implement this class."
        )

    @property
    def name(self) -> str:
        """Return extractor name."""
        return "YOLOv8"

    def extract(self, image: np.ndarray) -> Optional[BallLocation]:
        """Extract ball location using YOLOv8 (not implemented).

        Args:
            image: Input image (BGR).

        Returns:
            BallLocation if ball detected, None otherwise.

        Raises:
            NotImplementedError: Not yet implemented.
        """
        raise NotImplementedError("YOLOv8Extractor.extract() not yet implemented")

