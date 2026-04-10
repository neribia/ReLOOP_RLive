"""Base Render Engine module.

This module provides the abstract base class for render engines that generate
visual observations from the simulation.

Supported render backends (via subclasses):
    - Mitsuba: Physically-based rendering
    - Blender: 3D rendering with Python API
    - OpenGL: Real-time rendering
    - Panda3D: Game engine rendering
"""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from rlive_sim.engine.core.base_physics_engine import SceneObject


# TODO: Create Pydantic model for SceneState like PhysicsState
# TODO: Consider if this is overkill - scene primarily changes via physics updates
class SceneState:
    """Represents the current situation of the scene.

    Attributes:
        objects: List of object positions and properties
        camera: Camera position and orientation
        lights: Lighting configuration
        time: Current simulation time
        frame: Frame number
    """
    
    objects: list[dict] | None = None



class BaseRenderEngine(ABC):
    """Abstract base class for render engines.

    This class defines the interface that all render engine implementations
    must follow. Subclasses should implement the abstract methods to integrate
    specific render backends like Mitsuba, Blender, or OpenGL.

    Attributes:
        width: Image width in pixels.
        height: Image height in pixels.
        channels: Number of color channels (e.g., 3 for RGB, 4 for RGBA).

    Methods:
        render(scene_state): Render the current scene and return an image.
        setup_scene(scene_config): Set up the scene with given configuration.
        get_image(): Get the last rendered image without re-rendering.
        set_camera_pose(position, target, up): Set the camera position and orientation.
        get_resolution(): Get the current render resolution as (height, width, channels).
        close(): Clean up resources.

    Examples:
        Using render engine with scene data:

            engine = MyRenderEngine(width=800, height=600)
            engine.setup_scene({
                "objects": [{"type": "sphere", "position": [0, 1, 0]}],
                "camera": {"position": [0, 0, 5], "target": [0, 1, 0]},
                "lights": [{"type": "sun", "intensity": 1.0}]
            })
            image = engine.render({"time": 0.0})  # Returns (600, 800, 3)
    """

    def __init__(
        self,
        width: int = 640,  # TODO: set in RenderConfig instead
        height: int = 480,  # TODO: set in RenderConfig instead
        channels: int = 3,  # TODO: set in RenderConfig instead
    ) -> None:
        """Initialize the render engine.

        Args:
            width: Image width in pixels. Defaults to 640.
            height: Image height in pixels. Defaults to 480.
            channels: Number of color channels. Defaults to 3 (RGB).
            _image: Holds the current rendered image of the scene.
        """
        self.width = width
        self.height = height
        self.channels = channels
        self._image = None

    @abstractmethod
    def render(self, objects: list[SceneObject]) -> np.ndarray:
        """Render the scene with a list of dynamic objects.

        This method should process the list of dynamic objects appropriately
        for the specific render backend (e.g., drawing shapes, updating
        scenegraph nodes) and return the resulting image.

        Args:
            objects: List of dynamic objects to render.

        Returns:
            np.ndarray: Rendered image (height, width, channels).

        Examples:
            Rendering with object and camera data:

                scene = {
                    "objects": [
                        {"id": 0, "position": [0, 1, 0], "type": "sphere"},
                        {"id": 1, "position": [2, 0, 0], "type": "box"}
                    ],
                    "camera": {
                        "position": [0, 0, 5],
                        "target": [0, 1, 0],
                        "up": [0, 1, 0]
                    },
                    "time": 0.5
                }
                image = engine.render(scene)
                # image.shape == (480, 640, 3)
        """
        pass
    
    
    # TODO: Consider creating a Pydantic model for scene_config validation
    @abstractmethod
    def setup_scene(self, scene_config: dict[str, Any]) -> None:
        """Set up the initial scene rendering context (e.g. background, static geometry).

        Args:
            scene_config: Configuration dict to set up the scene's appearance.
        """
        pass

    @abstractmethod
    def project_to_2d(self, position: list[float]) -> tuple[int, int] | None:
        """Project a 3D position in world coordinates to 2D pixel coordinates.

        Args:
            position: Position [x, y, z] in world coordinates.

        Returns:
            tuple[int, int] | None: (x, y) pixel coordinates, or None if out of bounds/unsupported.
        """
        pass

    def get_resolution(self) -> tuple[int, int, int]:
        """Get the current render resolution.

        Returns:
            tuple[int, int, int]: (height, width, channels) tuple.
                For example, (480, 640, 3) for a 640x480 RGB image.

        Examples:
            Getting resolution to validate rendering setup:

                engine = MyRenderEngine(width=800, height=600, channels=3)
                h, w, c = engine.get_resolution()
                # h == 600, w == 800, c == 3
        """
        return (self.height, self.width, self.channels)

    def close(self) -> None:
        """Clean up resources. Override in subclasses if needed."""
        pass
