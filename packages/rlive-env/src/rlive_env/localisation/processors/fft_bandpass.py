"""FFT-based bandpass filtering processor."""

import numpy as np

from rlive_env.localisation.processors.base import AbstractProcessor
from rlive_env.config import processing_config


class FFTBandpassProcessor(AbstractProcessor):
    """Apply FFT-based bandpass filtering to isolate frequency components.

    Performs Fast Fourier Transform (FFT) on the image, applies a bandpass
    filter in the frequency domain, and transforms back to spatial domain.
    Useful for enhancing circular objects and reducing noise.

    Loads default parameters from rlive_env.config.processing_config module.

    Examples:
        processor = FFTBandpassProcessor()  # Uses defaults from config
        filtered = processor.process(gray_image)

        processor = FFTBandpassProcessor(low_freq=20, high_freq=80)  # Override defaults
        filtered = processor.process(gray_image)
    """

    def __init__(
        self,
        low_freq: int = processing_config.FFT_LOW_FREQ,
        high_freq: int = processing_config.FFT_HIGH_FREQ,
    ):
        """Initialize FFT bandpass processor with defaults from config module.

        Args:
            low_freq: Lower frequency cutoff (exclude frequencies below this).
            high_freq: Upper frequency cutoff (exclude frequencies above this).
        """
        self.low_freq = low_freq
        self.high_freq = high_freq

    @property
    def name(self) -> str:
        """Return processor name."""
        return "FFTBandpass"

    def process(self, image: np.ndarray) -> np.ndarray:
        """Apply FFT-based bandpass filtering.

        Args:
            image: Input grayscale image.

        Returns:
            Bandpass-filtered image (grayscale, uint8).
        """
        # Ensure image is uint8
        if image.dtype != np.uint8:
            image = np.uint8(image)

        # Compute FFT
        fft = np.fft.fft2(image)
        fft_shifted = np.fft.fftshift(fft)

        # Get dimensions
        rows, cols = image.shape
        crow, ccol = rows // 2, cols // 2

        # Create bandpass filter mask
        y = np.arange(-crow, rows - crow)
        x = np.arange(-ccol, cols - ccol)
        X, Y = np.meshgrid(x, y)
        distance = np.sqrt(X ** 2 + Y ** 2)

        # Create bandpass mask: 1 where low_freq <= distance <= high_freq
        mask = np.logical_and(distance >= self.low_freq, distance <= self.high_freq).astype(np.uint8)

        # Apply mask
        fft_filtered = fft_shifted * mask

        # Inverse FFT
        fft_ishift = np.fft.ifftshift(fft_filtered)
        filtered = np.fft.ifft2(fft_ishift)
        filtered = np.abs(filtered)

        # Normalize to 0-255
        filtered_normalized = (
            np.uint8(255 * filtered / np.max(filtered))
            if np.max(filtered) > 0
            else np.uint8(filtered)
        )

        return filtered_normalized

