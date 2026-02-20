"""Abstract base class for image processing strategies."""

from abc import ABC, abstractmethod

import numpy as np


class AbstractProcessor(ABC):
    """Abstract base for image processing strategies.

    Defines the interface for image transformation operations.
    Each processor takes an image and returns a processed image.

    Examples:
        Implement a custom processor:
            class CustomProcessor(AbstractProcessor):
                def process(self, image: np.ndarray) -> np.ndarray:
                    return cv2.GaussianBlur(image, (5, 5), 0)
    """

    @abstractmethod
    def process(self, image: np.ndarray) -> np.ndarray:
        """Process an image.

        Args:
            image: Input image (typically numpy array).

        Returns:
            Processed image (numpy array).
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of this processor.

        Returns:
            String identifier (e.g., 'GaussianBlur', 'CannyEdge').
        """
        pass

