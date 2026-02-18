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
    def render(self, scene_state: dict[str, Any]) -> np.ndarray:
        """Render the current scene and return an image.

        Args:
            scene_state: Dictionary containing scene information such as
                object positions, camera pose, lighting, and animation frame.
                Expected keys may include:
                - "objects": List of object positions and properties
                - "camera": Camera position and orientation
                - "lights": Lighting configuration
                - "time": Current simulation time
                - "frame": Frame number

        Returns:
            np.ndarray: Rendered image as uint8 array with shape
                (height, width, channels). For example, (480, 640, 3) for
                RGB rendering at 640x480 resolution.

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
        """Set up the scene with given configuration.

        This method initializes or reconfigures the scene with objects,
        lighting, camera, and other visual elements. Call this once before
        starting the render loop.

        Args:
            scene_config: Dictionary containing scene setup parameters.
                Expected keys include:
                - "objects": List of object definitions (type, position, size, etc.)
                - "camera": Initial camera configuration (position, target, up)
                - "lights": Lighting configuration (sun, point lights, etc.)
                - "background": Background settings (color, skybox, etc.)
                - "environment": Environmental parameters (fog, materials, etc.)

        Examples:
            Setting up a simple scene with objects and lighting:

                config = {
                    "objects": [
                        {"id": 0, "type": "sphere", "position": [0, 1, 0], "radius": 0.5},
                        {"id": 1, "type": "plane", "position": [0, 0, 0], "size": [10, 10]}
                    ],
                    "camera": {
                        "position": [0, 2, 5],
                        "target": [0, 1, 0],
                        "up": [0, 1, 0]
                    },
                    "lights": [
                        {"type": "sun", "direction": [1, 1, 1], "intensity": 1.0}
                    ],
                    "background": {"color": [0.1, 0.1, 0.1]}
                }
                engine.setup_scene(config)
        """
        pass

    @abstractmethod
    def set_camera_pose(
        self,
        position: list[float],
        target: list[float],
        up: list[float] | None = None,
    ) -> None:
        """Set the camera position and orientation.

        Updates the camera's position in world space, the point it looks at,
        and its up direction. This is typically called before render() to
        control the viewpoint.

        Args:
            position: Camera position [x, y, z] in world coordinates.
            target: Point the camera looks at [x, y, z] in world coordinates.
            up: Up vector [x, y, z] to define camera roll. Defaults to [0, 0, 1].

        Examples:
            Setting camera to view a scene from different angles:

                # Front view
                engine.set_camera(
                    position=[0, 2, 5],
                    target=[0, 1, 0],
                    up=[0, 1, 0]
                )

                # Top-down view
                engine.set_camera(
                    position=[0, 10, 0],
                    target=[0, 0, 0],
                    up=[0, 0, 1]
                )
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
