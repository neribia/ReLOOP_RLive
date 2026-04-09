"""Abstract base class for ball extraction strategies."""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from rlive_common.core.ball_location import BallLocation


class AbstractBallExtractor(ABC):
    """Abstract base for ball extraction strategies.

    Defines the interface that all concrete extractors (classical CV, deep learning, etc.)
    must implement. Each extractor takes a processed image and returns the ball location.

    Examples:
        Implement a custom extractor:
            class CustomExtractor(AbstractBallExtractor):
                def extract(self, image: np.ndarray) -> Optional[BallLocation]:
                    # Custom detection logic here
                    return BallLocation(x=100, y=200)

        Use in BallLocalisator:
            extractor = CustomExtractor()
            localiser = BallLocalisator(extractors=[extractor])
            location = localiser.get_position(image)
    """

    @abstractmethod
    def extract(self, image: np.ndarray) -> Optional[BallLocation]:
        """Extract ball location from image.

        Args:
            image: Input image (typically preprocessed by pipeline).
                   Format depends on extractor implementation.

        Returns:
            BallLocation if ball detected, None otherwise.
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of this extractor.

        Returns:
            String identifier (e.g., 'HoughCircle', 'YOLOv8').
        """
        pass

    def get_debug_image(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Get debug visualization of the extraction process.

        Optional method for visualization of intermediate steps.
        Default implementation returns None (no debug image).

        Args:
            image: Input image used for extraction.

        Returns:
            Debug image showing detection process, or None if not implemented.
        """
        return None
