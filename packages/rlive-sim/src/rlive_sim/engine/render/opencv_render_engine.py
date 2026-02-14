"""OpenCV Render Engine implementation.

This module provides a simple 2D render engine using OpenCV to draw
a ball in a box. The ball is blue, the box is gray, and the background
is black.
"""

from typing import Any

import cv2
import numpy as np

from rlive_sim.engine.base_render_engine import BaseRenderEngine
from rlive_sim.engine.registry import register_render_backend
from rlive_sim.config import RenderBackend



@register_render_backend(RenderBackend.OPENCV)
class OpenCVRenderEngine(BaseRenderEngine):
    """OpenCV-based 2D render engine.

    Renders a simple scene with a blue ball inside a gray box on a
    black background. Fast and lightweight for simple simulations.

    Attributes:
        ball_radius: Radius of the ball in pixels.
        ball_color: Ball color in BGR format (default: blue).
        box_color: Box border color in BGR format (default: gray).
        bg_color: Background color in BGR format (default: black).
        box_thickness: Thickness of the box border in pixels.

    Example:
        ... engine = OpenCVRenderEngine(width=640, height=480, ball_radius=20)
        ... engine.setup_scene({})
        ... scene_state = {"objects": [{"position": [320, 240, 0]}]}
        ... image = engine.render(scene_state)
        ... image.shape
        (480, 640, 3)
    """

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        channels: int = 3,
        ball_radius: int = 20,
        ball_color: tuple[int, int, int] = (255, 0, 0),
        box_color: tuple[int, int, int] = (128, 128, 128),
        bg_color: tuple[int, int, int] = (0, 0, 0),
        box_thickness: int = 2,
        box_margin: int = 0,
        **kwargs: Any,
    ) -> None:
        """Initialize the OpenCV render engine.

        Attributes:
            width: Image width in pixels.
            height: Image height in pixels.
            channels: Number of color channels.
            ball_radius: Radius of the ball in pixels.
            ball_color: Ball color in BGR (blue).
            box_color: Box border color in BGR (gray).
            bg_color: Background color in BGR (black).
            box_thickness: Box border thickness.
            box_margin: Margin between image edge and box.
            **kwargs: Additional arguments.
        """
        super().__init__(width=width, height=height, channels=channels)

        self.ball_radius = ball_radius
        self.ball_color = ball_color
        self.box_color = box_color
        self.bg_color = bg_color
        self.box_thickness = box_thickness
        self.box_margin = box_margin
        self._last_image: np.ndarray | None = None
        self._camera_position: list[float] = [0.0, 0.0, 100.0]
        self._camera_target: list[float] = [width / 2, height / 2, 0.0]

    def render(self, scene_state: dict[str, Any]) -> np.ndarray:
        """Render the scene with a ball in a box.

        Attributes:
            scene_state: Dictionary with objects and positions.

        Returns:
            np.ndarray: Rendered image (height, width, channels).
        """
        image = np.full(
            (self.height, self.width, self.channels),
            self.bg_color,
            dtype=np.uint8,
        )

        margin = self.box_margin
        cv2.rectangle(
            image,
            (margin, margin),
            (self.width - margin - 1, self.height - margin - 1),
            self.box_color,
            self.box_thickness,
        )

        objects = scene_state.get("objects", [])
        for obj in objects:
            position = obj.get("position", [self.width / 2, self.height / 2, 0])
            x, y = int(position[0]), int(position[1])
            cv2.circle(image, (x, y), self.ball_radius, self.ball_color, -1)

        self._last_image = image
        return image

    def setup_scene(self, scene_config: dict[str, Any]) -> None:
        """Set up the scene configuration."""
        if "ball_color" in scene_config:
            self.ball_color = tuple(scene_config["ball_color"])
        if "box_color" in scene_config:
            self.box_color = tuple(scene_config["box_color"])
        if "bg_color" in scene_config:
            self.bg_color = tuple(scene_config["bg_color"])
        if "ball_radius" in scene_config:
            self.ball_radius = int(scene_config["ball_radius"])
        if "box_margin" in scene_config:
            self.box_margin = int(scene_config["box_margin"])

    def get_image(self) -> np.ndarray:
        """Get the last rendered image."""
        if self._last_image is not None:
            return self._last_image
        return np.full(
            (self.height, self.width, self.channels),
            self.bg_color,
            dtype=np.uint8,
        )

    def set_camera(
        self,
        position: list[float],
        target: list[float],
        up: list[float] | None = None,
    ) -> None:
        """Set camera position (stored but not used in 2D)."""
        self._camera_position = position
        self._camera_target = target

    def close(self) -> None:
        """Clean up resources."""
        self._last_image = None
