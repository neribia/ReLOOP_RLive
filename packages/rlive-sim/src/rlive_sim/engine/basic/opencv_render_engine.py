"""OpenCV Render Engine implementation.

This module provides a simple 2D render engine using OpenCV to draw
a ball in a box. The ball is blue, the box is gray, and the background
is black.
"""

from typing import Any

import math
import cv2 as cv
import numpy as np

from rlive_sim.engine.core.base_render_engine import BaseRenderEngine
from rlive_sim.engine.core.base_physics_engine import SceneObject
from rlive_sim.engine.core.registry import register_render_backend
from rlive_sim.config import RenderBackend
import rlive_sim.config.basic_config as basic_cfg


@register_render_backend(RenderBackend.OPENCV)
class OpenCVRenderEngine(BaseRenderEngine):
    """OpenCV-based 2D render engine.

    Renders a simple scene with a colored ball inside a colored box on a
    colored background. Fast and lightweight for simple 2D simulations.

    Attributes:
        fov_degrees: Field of view in degrees (default: 60.0).
        camera_position: Position of the camera [x, y, z] in meters.
        camera_rotation: Euler angles [roll, pitch, yaw] in radians.
        bg_color: Background color in BGR format (default: black).
        width: Image width in pixels (from BaseRenderEngine).
        height: Image height in pixels (from BaseRenderEngine).
        channels: Number of color channels (from BaseRenderEngine).

    Methods:
        render(objects): Render the scene with objects supplied by physics.
        setup_scene(scene_config): Configure scene colors and camera settings.
        set_camera_pose(position, target, up): Sets the camera position and recalculates rotation to face a target.
        get_image(): Get the last rendered image without re-rendering.
        get_resolution(): Get the render resolution as (height, width, channels).
        close(): Clean up resources.

    Example:
        Creating and using the OpenCV render engine:

            from rlive_sim.engine.core.base_physics_engine import SceneObject

            engine = OpenCVRenderEngine(width=640, height=480)
            objects = [
                SceneObject(id="robot", position=[0.5, 0.5, 0.0], dimensions=[0.05], rotation=[0,0,0])
            ]
            image = engine.render(objects)
            print(image.shape)  # (480, 640, 3)
    """

    def __init__(
        self,
        width: int = basic_cfg.IMAGE_WIDTH,
        height: int = basic_cfg.IMAGE_HEIGHT,
        channels: int = 3,
        fov_degrees: float = basic_cfg.FOV_DEGREES,
        bg_color: tuple[int, int, int] = (0, 0, 0),
        camera_position: list[float] | None = None,
        camera_rotation: list[float] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the OpenCV render engine.

        Args:
            width: Image width in pixels. Defaults to basic_cfg.IMAGE_WIDTH.
            height: Image height in pixels. Defaults to basic_cfg.IMAGE_HEIGHT.
            channels: Number of color channels.
            fov_degrees: Camera horizontal field of view in degrees.
            bg_color: Background color in BGR (black).
            camera_position: XYZ coordinates of the camera.
            camera_rotation: Euler angles [roll, pitch, yaw] in radians.
            **kwargs: Additional arguments.
        """
        super().__init__(width=width, height=height, channels=channels)

        self.fov_degrees = fov_degrees
        self.bg_color = bg_color

        self.camera_position = camera_position if camera_position is not None else [
            basic_cfg.CAMERA_POSITION_X,
            basic_cfg.CAMERA_POSITION_Y,
            basic_cfg.CAMERA_POSITION_Z,
        ]
        self.camera_rotation = camera_rotation if camera_rotation is not None else [
            basic_cfg.CAMERA_ROTATION_ROLL,
            basic_cfg.CAMERA_ROTATION_PITCH,
            basic_cfg.CAMERA_ROTATION_YAW,
        ]

        self._last_image: np.ndarray | None = None
        self._background_image: np.ndarray | None = None

    def _get_intrinsic_matrix(self) -> np.ndarray:
        """Calculate the camera intrinsic matrix K.

        Assumes a symmetric pinhole camera model with square pixels (fx = fy).
        """
        fov_rad = math.radians(self.fov_degrees)
        f = (self.width / 2.0) / math.tan(fov_rad / 2.0)
        cx = self.width / 2.0
        cy = self.height / 2.0
        return np.array([
            [f, 0, cx],
            [0, f, cy],
            [0, 0, 1]
        ], dtype=np.float32)

    def _get_extrinsic(self) -> tuple[np.ndarray, np.ndarray]:
        """Calculate the extrinsic rotation and translation vectors."""
        rx, ry, rz = self.camera_rotation

        # Rotation matrices for each axis
        R_x = np.array([[1, 0, 0],
                        [0, np.cos(rx), -np.sin(rx)],
                        [0, np.sin(rx), np.cos(rx)]])

        R_y = np.array([[np.cos(ry), 0, np.sin(ry)],
                        [0, 1, 0],
                        [-np.sin(ry), 0, np.cos(ry)]])

        R_z = np.array([[np.cos(rz), -np.sin(rz), 0],
                        [np.sin(rz), np.cos(rz), 0],
                        [0, 0, 1]])

        # Combined rotation matrix
        R_cam = R_z @ R_y @ R_x

        # Extrinsic matrix (world to camera mappings)
        R_ext = R_cam.T
        t_ext = -R_ext @ np.array(self.camera_position, dtype=np.float32)

        rvec, _ = cv.Rodrigues(R_ext)
        return rvec.reshape((3, 1)), t_ext.reshape((3, 1))

    def get_resolution(self) -> tuple[int, int, int]:
        """Get the render resolution as (height, width, channels)."""
        return (self.height, self.width, self.channels)

    def render(self, objects: list[SceneObject]) -> np.ndarray:
        """Render the scene with dynamic objects.

        Args:
            objects: List of dynamic scene objects (e.g. robot).

        Returns:
            np.ndarray: Rendered image (height, width, channels).
        """
        if self._background_image is not None:
            image = self._background_image.copy()
        else:
            image = np.full(
                (self.height, self.width, self.channels),
                self.bg_color,
                dtype=np.uint8,
            )

        for obj in objects:
            if obj.id == "robot":
                self._draw_robot(image, obj)
            elif obj.id == "eurobox":
                # Static objects should typically be drawn in setup_scene, but we support dynamic fallback
                self._draw_eurobox(image, obj)

        self._last_image = image
        return image

    def _draw_eurobox(self, image: np.ndarray, obj: SceneObject) -> None:
        """Draw the eurobox object on the image."""
        w, h, _ = obj.dimensions
        # Calculate corners based on position (assuming position is center)
        cx, cy, _ = obj.position
        points = [
            [cx - w/2, cy - h/2, 0.0],
            [cx + w/2, cy - h/2, 0.0],
            [cx + w/2, cy + h/2, 0.0],
            [cx - w/2, cy + h/2, 0.0],
        ]
        pts_2d = []
        for p in points:
            pt = self.project_to_2d(p)
            if pt is not None:
                pts_2d.append(pt)

        if len(pts_2d) >= 3:
            pts_2d = np.array(pts_2d, dtype=np.int32).reshape((-1, 1, 2))
            color = (128, 128, 128)  # Default visual color
            cv.fillPoly(image, [pts_2d], color)

    def _draw_robot(self, image: np.ndarray, obj: SceneObject) -> None:
        """Draw the robot object on the image."""
        position = obj.position
        radius = obj.dimensions[0] if len(obj.dimensions) > 0 else 0.05
        color = (0, 0, 255)  # Default visual color for "robot" / ball

        pt = self.project_to_2d(position)
        if pt is not None:
            # Project a point on the surface to determine proportional pixel radius based on depth
            surface_pt = [position[0] + radius, position[1], position[2] if len(position) > 2 else 0]
            pt_edge = self.project_to_2d(surface_pt)

            pixel_radius = 1
            if pt_edge is not None:
                dx = pt_edge[0] - pt[0]
                dy = pt_edge[1] - pt[1]
                pixel_radius = max(1, int(np.sqrt(dx**2 + dy**2)))

            cv.circle(image, pt, pixel_radius, color, -1)

    def setup_scene(self, scene_config: dict[str, Any]) -> None:
        """Set up the scene configuration and pre-render the static background."""
        if "bg_color" in scene_config:
            self.bg_color = tuple(scene_config["bg_color"])
        if "fov_degrees" in scene_config:
            self.fov_degrees = float(scene_config["fov_degrees"])
        if "camera_position" in scene_config:
            self.camera_position = list(scene_config["camera_position"])
        if "camera_rotation" in scene_config:
            self.camera_rotation = list(scene_config["camera_rotation"])

        self._background_image = np.full(
            (self.height, self.width, self.channels),
            self.bg_color,
            dtype=np.uint8,
        )

        static_objects = scene_config.get("static_objects", [])
        for obj in static_objects:
            if obj.id == "eurobox":
                self._draw_eurobox(self._background_image, obj)


        self._last_image = self._background_image.copy()

    def get_image(self) -> np.ndarray:
        """Get the last rendered image."""
        if self._last_image is not None:
            return self._last_image
        return np.full(
            (self.height, self.width, self.channels),
            self.bg_color,
            dtype=np.uint8,
        )

    def project_to_2d(self, position: list[float]) -> tuple[int, int] | None:
        """Project a 3D position in meters to 2D pixel coordinates.

        Args:
            position: Position [x, y, z] in world meters.

        Returns:
            tuple[int, int] | None: (x, y) pixel coordinates, or None if invalid or out of bounds.
        """
        if not position or len(position) < 2:
            return None

        pos_3d = np.array(position, dtype=np.float32)
        if len(pos_3d) == 2:
            pos_3d = np.array([pos_3d[0], pos_3d[1], 0.0], dtype=np.float32)

        K = self._get_intrinsic_matrix()
        rvec, tvec = self._get_extrinsic()

        # Convert rvec to rotation matrix to check depth (Z > 0 in camera space)
        R_ext, _ = cv.Rodrigues(rvec)
        pos_cam = R_ext @ pos_3d + tvec.reshape(3)
        if pos_cam[2] <= 0:
            return None

        pts_2d, _ = cv.projectPoints(pos_3d.reshape((1, 1, 3)), rvec, tvec, K, None)

        if pts_2d is not None:
            u, v = pts_2d[0, 0]
            if 0 <= u < self.width and 0 <= v < self.height:
                return int(u), int(v)
        return None

    def close(self) -> None:
        """Clean up resources."""
        self._last_image = None
