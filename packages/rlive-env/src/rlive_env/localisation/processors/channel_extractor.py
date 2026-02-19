"""Channel extraction processor for multi-channel images."""

import cv2 as cv
import numpy as np

from rlive_env.localisation.processors.base import AbstractProcessor


class ChannelExtractorProcessor(AbstractProcessor):
    """Extract a single channel from a multi-channel image.

    Useful for isolating specific channels from HSV, BGR, or other
    multi-channel color spaces. For example, extracting the Hue channel
    from an HSV image or the Blue channel from a BGR image.

    Examples:
        Extract Hue channel from HSV image:
            processor = ChannelExtractorProcessor(channel=0)  # H is channel 0
            hue_image = processor.process(hsv_image)

        Extract Saturation channel from HSV:
            processor = ChannelExtractorProcessor(channel=1)  # S is channel 1
            sat_image = processor.process(hsv_image)

        Extract Value/Brightness from HSV:
            processor = ChannelExtractorProcessor(channel=2)  # V is channel 2
            value_image = processor.process(hsv_image)

        Extract Red channel from BGR:
            processor = ChannelExtractorProcessor(channel=2)  # R is channel 2 in BGR
            red_image = processor.process(bgr_image)
    """

    def __init__(self, channel: int = 0):
        """Initialize channel extractor.

        Args:
            channel: Channel index to extract (0, 1, or 2 for typical 3-channel images).
                    For HSV: 0=Hue, 1=Saturation, 2=Value
                    For BGR: 0=Blue, 1=Green, 2=Red

        Raises:
            ValueError: If channel is not 0, 1, or 2.
        """
        if channel not in (0, 1, 2):
            raise ValueError("Channel must be 0, 1, or 2")
        self.channel = channel

    @property
    def name(self) -> str:
        """Return processor name."""
        channel_names = {0: "Channel0", 1: "Channel1", 2: "Channel2"}
        return f"ChannelExtractor_{channel_names[self.channel]}"

    def process(self, image: np.ndarray) -> np.ndarray:
        """Extract single channel from image.

        Args:
            image: Multi-channel input image (typically 3-channel).

        Returns:
            Single-channel (grayscale) image containing the specified channel.

        Raises:
            ValueError: If image doesn't have enough channels.
        """
        if len(image.shape) != 3:
            raise ValueError(f"Expected 3-channel image, got {len(image.shape)}-channel")

        if image.shape[2] <= self.channel:
            raise ValueError(
                f"Channel {self.channel} not available in image with {image.shape[2]} channels"
            )

        # Extract channel
        channel_image = cv.split(image)[self.channel]
        return channel_image

