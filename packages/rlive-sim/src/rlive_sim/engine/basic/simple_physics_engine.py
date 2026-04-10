"""Simple Physics Engine implementation.

This module provides a basic 2D physics engine where a ball moves
in a bounded box. The ball moves in a given direction (angle) for
a given distance. If the ball hits a wall, it slides along the wall.
"""

import math
from typing import Any

import numpy as np

from rlive_sim.engine.core.base_physics_engine import BasePhysicsEngine, PhysicsState, SceneObject
from rlive_sim.engine.core.registry import register_physics_backend
from rlive_sim.config import PhysicsBackend
import rlive_sim.config.basic_config as basic_cfg
import rlive_sim.config.bolt_config as bolt_cfg


@register_physics_backend(PhysicsBackend.SIMPLE)
class SimplePhysicsEngine(BasePhysicsEngine):
    """Simple 2D physics engine with a ball in a box.

    The ball moves based on an angle, speed, and duration. When hitting walls,
    the ball slides along the wall (remaining movement is clamped to stay inside
    the box). Uses an action vector of `[heading, speed, duration]`.

    Attributes:
        box_width: Width of the bounding box in meters.
        box_height: Height of the bounding box in meters.
        ball_radius: Radius of the ball in meters.
        dt: Simulation timestep in seconds.
        gravity: Gravity vector (always [0, 0, 0] for 2D top-down).

    Methods:
        reset(initial_state): Reset the ball to the initial position.
        update(action, dt): Advance the simulation by one timestep (BasePhysicsEngine interface).
        get_state(): Get the current physics state.
        set_state(state): Directly set the physics state.
        get_bounds(): Get the valid bounds for ball position.
        close(): Clean up resources.


    Example:
        Creating and using the simple physics engine:

            engine = SimplePhysicsEngine(box_width=1.0, box_height=1.0)
            state = engine.reset()

            # Action as [heading, speed, duration]
            new_state = engine.update([45, 0.1, 1.0])  # Move 45° angle, 0.1 m/s, for 1.0 seconds
            print(new_state.position[:2])  # [x, y] position
    """

    def __init__(
        self,
        box_width: float = basic_cfg.BOX_WIDTH,
        box_height: float = basic_cfg.BOX_HEIGHT,
        ball_radius: float = bolt_cfg.RADIUS_M,
        dt: float = 0.01,
        max_speed: float = 1.0,
        **kwargs: Any,
    ) -> None:
        """Initialize the simple physics engine.

        Args:
            box_width: Width of the bounding box in meters.
            box_height: Height of the bounding box in meters.
            ball_radius: Radius of the ball in meters.
            dt: Simulation timestep (for interface compatibility).
            **kwargs: Additional arguments.
        """
        super().__init__(dt=dt, gravity=[0.0, 0.0, 0.0])  # No gravity in 2D top-down

        self.box_width = box_width
        self.box_height = box_height
        self.ball_radius = ball_radius
        self.distance = 0.05
        self.max_speed = max_speed

        self._static_objects: list[SceneObject] = []

    def setup_scene(self, scene_config: dict[str, Any]) -> None:
        if "box_width" in scene_config:
            self.box_width = float(scene_config["box_width"])
        if "box_height" in scene_config:
            self.box_height = float(scene_config["box_height"])
        if "ball_radius" in scene_config:
            self.ball_radius = float(scene_config["ball_radius"])

        w = float(self.box_width)
        h = float(self.box_height)
        self._static_objects = [
            SceneObject(
                id="eurobox",
                position=[0.0, 0.0, 0.0],
                rotation=[0.0, 0.0, 0.0],
                dimensions=[w, h, 0.0],
            )
        ]

    def reset(self, initial_state: PhysicsState | None = None) -> PhysicsState:
        """Reset the ball to initial position.

        Args:
            initial_state: Optional initial state. If None, ball starts at center.

        Returns:
            PhysicsState: Initial state after reset.
        """
        if initial_state is not None:
            # Validate and clamp position to box
            x, y, z = initial_state.position
            min_x = -float(self.box_width / 2) + float(self.ball_radius)
            max_x = float(self.box_width / 2) - float(self.ball_radius)
            min_y = -float(self.box_height / 2) + float(self.ball_radius)
            max_y = float(self.box_height / 2) - float(self.ball_radius)

            x = max(min_x, min(max_x, x))
            y = max(min_y, min(max_y, y))

            self._state = PhysicsState(
                position=[x, y, z],
                velocity=[0.0, 0.0, 0.0],
                rotation=initial_state.rotation,
                angular_velocity=[0.0, 0.0, 0.0],
            )
        else:
            # Start at center
            self._state = PhysicsState(
                position=[0.0, 0.0, 0.0],
                velocity=[0.0, 0.0, 0.0],
                rotation=[1.0, 0.0, 0.0, 0.0],
                angular_velocity=[0.0, 0.0, 0.0],
            )

        self._pending_action = None
        return self._state

    def update(self, action: np.ndarray | list[float], dt: float | None = None) -> PhysicsState:
        """Advance the physics simulation by applying the pending action.

        The ball moves in the direction specified by the angle for
        the given distance. If hitting a wall, it slides along the wall.

        Args:
            action: Movement command as vector:
                - `[heading, speed, duration]`
            dt: Not used in this simple implementation.

        Returns:
            PhysicsState: Updated state after movement.
        """

        self._apply_action(action)
        if self._pending_action is None:
            return self._state

        angle_deg, distance = self._pending_action
        self._pending_action = None

        # Convert angle to radians (0° = right, 90° = down in screen coords)
        angle_rad = math.radians(angle_deg)

        # Calculate movement delta
        dx = distance * math.cos(angle_rad)
        dy = distance * math.sin(angle_rad)

        # Current position
        x, y, z = self._state.position

        # Calculate new position
        new_x = x + dx
        new_y = y + dy

        # Clamp to box boundaries (considering ball radius)
        min_x = -float(self.box_width / 2) + float(self.ball_radius)
        max_x = float(self.box_width / 2) - float(self.ball_radius)
        min_y = -float(self.box_height / 2) + float(self.ball_radius)
        max_y = float(self.box_height / 2) - float(self.ball_radius)

        # Check if hit wall
        hit_wall = (
            new_x <= min_x or new_x >= max_x or
            new_y <= min_y or new_y >= max_y
        )

        new_x = max(min_x, min(max_x, new_x))
        new_y = max(min_y, min(max_y, new_y))

        # Update state
        self._state = PhysicsState(
            position=[new_x, new_y, z],
            velocity=[dx, dy, 0.0],  # Store last movement as velocity
            rotation=self._state.rotation,
            angular_velocity=self._state.angular_velocity,
            extra={
                "last_angle": angle_deg,
                "last_distance": distance,
                "hit_wall": hit_wall,
            },
        )

        return self._state

    def get_state(self) -> PhysicsState:
        """Get the current physics state.

        Returns:
            PhysicsState: Current ball position and velocity.
        """
        return self._state

    def get_scene_objects(self) -> list[SceneObject]:
        """Get the objects to render in the scene.

        Returns:
            list[SceneObject]: List of dynamic scene objects (the robot).
        """
        return [
            SceneObject(
                id="robot",
                position=self._state.position,
                rotation=self._state.rotation[:3] if len(self._state.rotation) == 3 else self._state.rotation, # Handle euler vs quat loosely. Or just [0,0,0]
                dimensions=[self.ball_radius],
            )
        ]

    def _apply_action(self, action: np.ndarray | list[float]) -> None:
        """Apply a movement action to the ball.

        Args:
            action: Movement command as vector:
                - `[heading, speed, duration]`

        Example:
            ... engine.apply_action([45, 0.1, 1.0])  # Move 45°, 0.1 m/s, 1.0 seconds
        """
        if isinstance(action, np.ndarray):
            action = action.tolist()

        heading = action[0]
        speed = action[1]
        duration = action[2] if len(action) > 2 else 1.0

        self._pending_action = [float(heading), float(speed) * float(duration)]

    def set_state(self, state: PhysicsState) -> None:
        """Set the physics state directly.

        Args:
            state: The state to set. Position will be clamped to box bounds.
        """
        x, y, z = state.position
        min_x = -float(self.box_width / 2) + float(self.ball_radius)
        max_x = float(self.box_width / 2) - float(self.ball_radius)
        min_y = -float(self.box_height / 2) + float(self.ball_radius)
        max_y = float(self.box_height / 2) - float(self.ball_radius)

        self._state = PhysicsState(
            position=[max(min_x, min(max_x, x)), max(min_y, min(max_y, y)), z],
            velocity=state.velocity,
            rotation=state.rotation,
            angular_velocity=state.angular_velocity,
            extra=state.extra,
        )

    def get_bounds(self) -> tuple[float, float, float, float]:
        """Get the valid bounds for ball position.

        Returns:
            tuple[float, float, float, float]: (min_x, min_y, max_x, max_y).
        """
        return (
            -float(self.box_width / 2) + float(self.ball_radius),
            -float(self.box_height / 2) + float(self.ball_radius),
            float(self.box_width / 2) - float(self.ball_radius),
            float(self.box_height / 2) - float(self.ball_radius),
        )

    def close(self) -> None:
        """Clean up resources."""
        pass
