"""MuJoCo Render Engine implementation.

This module provides a render engine using MuJoCo's built-in rendering.
It renders the simulation from a fixed top-down camera view.
"""

from pathlib import Path
from typing import Any

import numpy as np

from rlive_sim.engine.base_render_engine import BaseRenderEngine
from rlive_sim.engine.registry import register_render_backend
from rlive_sim.config import RenderBackend


@register_render_backend(RenderBackend.MUJOCO)
class MujocoRenderEngine(BaseRenderEngine):
    """MuJoCo render engine for rendering simulation scenes.

    Renders the SpheroBolt+ simulation from a fixed top-down camera view
    using MuJoCo's built-in rendering system.

    Attributes:
        model: MuJoCo model instance.
        data: MuJoCo data instance (simulation state).
        renderer: MuJoCo renderer instance.
        width: Image width in pixels.
        height: Image height in pixels.
        channels: Number of color channels (RGB=3).
        _last_image: Last rendered image (cached).

    Methods:
        render(scene_state): Render the current scene and return an image.
        setup_scene(scene_config): Set up the scene configuration.
        get_image(): Get the last rendered image without re-rendering.
        get_resolution(): Get the render resolution as (height, width, channels).
        close(): Clean up resources.
    """

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        channels: int = 3,
        scene_path: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the MuJoCo render engine.

        Args:
            width: Image width in pixels. Defaults to 640.
            height: Image height in pixels. Defaults to 480.
            channels: Number of color channels (must be 3 for RGB). Defaults to 3.
            scene_path: Path to MJCF model file. If None, uses default spherobolt_box.xml.
            **kwargs: Additional arguments (ignored).

        Raises:
            ValueError: If channels is not 3.
        """
        super().__init__(width=width, height=height, channels=channels)

        if channels != 3:
            raise ValueError(f"MuJoCo render engine supports RGB (channels=3), got {channels}")

        # Lazy import of mujoco
        try:
            import mujoco
        except ImportError:
            raise ImportError(
                "MuJoCo is required for MujocoRenderEngine. "
                "Install with: pip install mujoco>=3.1.0"
            )

        # Determine scene path
        if scene_path is None:
            scene_path = str(
                Path(__file__).parent.parent / "resources" / "spherobolt_box.xml"
            )

        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(scene_path)
        self.data = mujoco.MjData(self.model)

        # Create MuJoCo renderer
        self.renderer = mujoco.Renderer(self.model, height=height, width=width)

        # Camera configuration
        self.camera_name = "top_camera"
        self._last_image: np.ndarray | None = None

    def render(self, scene_state: dict[str, Any]) -> np.ndarray:
        """Render the current scene and return an image.

        Args:
            scene_state: Dictionary with scene information (unused, uses internal state).

        Returns:
            np.ndarray: Rendered image as uint8 array with shape (height, width, 3).

        Examples:
            Rendering the scene:

                engine = MujocoRenderEngine(width=640, height=480)
                image = engine.render({})
                print(image.shape)  # (480, 640, 3)
                print(image.dtype)  # uint8
        """
        try:
            # Set up camera view
            self.renderer.update_scene(self.data, camera=self.camera_name)

            # Render to image
            image = self.renderer.render()

            # Ensure uint8 format
            if image.dtype != np.uint8:
                image = np.clip(image * 255, 0, 255).astype(np.uint8)

            self._last_image = image
            return image

        except Exception as e:
            # Return a blank image on error
            blank = np.zeros((self.height, self.width, self.channels), dtype=np.uint8)
            self._last_image = blank
            return blank

    def setup_scene(self, scene_config: dict[str, Any]) -> None:
        """Set up the scene configuration.

        Args:
            scene_config: Dictionary with scene configuration (currently unused).
        """
        # Scene is already set up during initialization
        pass

    def get_image(self) -> np.ndarray | None:
        """Get the last rendered image without re-rendering.

        Returns:
            np.ndarray: Last rendered image, or None if no image has been rendered yet.
        """
        return self._last_image

    def get_resolution(self) -> tuple[int, int, int]:
        """Get the render resolution.

        Returns:
            tuple[int, int, int]: (height, width, channels).
        """
        return (self.height, self.width, self.channels)

    def set_camera_pose(
        self,
        position: list[float],
        target: list[float],
        up: list[float] | None = None,
    ) -> None:
        """Set the camera position and orientation (not used in this engine).

        Args:
            position: Camera position [x, y, z].
            target: Camera target [x, y, z].
            up: Up vector [x, y, z]. Defaults to [0, 1, 0].
        """
        # MuJoCo camera is set in the XML model, not dynamically here
        pass

    def close(self) -> None:
        """Clean up resources."""
        if hasattr(self, 'renderer'):
            # MuJoCo renderer cleanup
            pass



