"""Simple Physics Engine implementation.

This module provides a basic 2D physics engine where a ball moves
in a bounded box. The ball moves in a given direction (angle) for
a given distance. If the ball hits a wall, it slides along the wall.
"""

import math
from typing import Any

import numpy as np

from rlive_sim.engine.core.base_physics_engine import BasePhysicsEngine, PhysicsState
from rlive_sim.engine.core.registry import register_physics_backend
from rlive_sim.config import PhysicsBackend



@register_physics_backend(PhysicsBackend.SIMPLE)
class SimplePhysicsEngine(BasePhysicsEngine):
    """Simple 2D physics engine with a ball in a box.

    The ball moves based on an angle and optional distance. When hitting walls,
    the ball slides along the wall (remaining movement is clamped to stay inside
    the box). If only an angle is provided, uses a default distance of 20 pixels.

    Attributes:
        box_width: Width of the bounding box in pixels.
        box_height: Height of the bounding box in pixels.
        ball_radius: Radius of the ball in pixels.
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

            engine = SimplePhysicsEngine(box_width=640, box_height=480)
            state = engine.reset()

            # Action as single angle (uses default 20px distance)
            new_state = engine.update(45)  # Move 45°, distance=20px

            # Or action as [angle, distance]
            new_state = engine.update([45])  # Move 45° angle, 50 pixels
            print(new_state.position[:2])  # [x, y] position
    """

    def __init__(
        self,
        box_width: int = 640,
        box_height: int = 480,
        ball_radius: int = 20,
        dt: float = 0.01,
        **kwargs: Any,
    ) -> None:
        """Initialize the simple physics engine.

        Args:
            box_width: Width of the bounding box in pixels. Defaults to 640.
            box_height: Height of the bounding box in pixels. Defaults to 480.
            ball_radius: Radius of the ball in pixels. Defaults to 20.
            dt: Simulation timestep (for interface compatibility).
            **kwargs: Additional arguments.
        """
        super().__init__(dt=dt, gravity=[0.0, 0.0, 0.0])  # No gravity in 2D top-down

        self.box_width = box_width
        self.box_height = box_height
        self.ball_radius = ball_radius
        self.distance = 20.0

        # Pending action: [angle_degrees, distance]
        self._pending_action: list[float] | None = None

        # Initialize ball at center
        self._state = PhysicsState(
            position=[float(box_width / 2), float(box_height / 2), 0.0],
            velocity=[0.0, 0.0, 0.0],
            rotation=[1.0, 0.0, 0.0, 0.0],
            angular_velocity=[0.0, 0.0, 0.0],
        )

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
            min_x = float(self.ball_radius)
            max_x = float(self.box_width - self.ball_radius)
            min_y = float(self.ball_radius)
            max_y = float(self.box_height - self.ball_radius)

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
                position=[float(self.box_width / 2), float(self.box_height / 2), 0.0],
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
            action: Movement command as either:
                - Single angle (float/int): Direction of movement in degrees (0° = right, 90° = down)
                  Uses default distance of 20 pixels.
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
        min_x = float(self.ball_radius)
        max_x = float(self.box_width - self.ball_radius)
        min_y = float(self.ball_radius)
        max_y = float(self.box_height - self.ball_radius)

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

    def _apply_action(self, action: np.ndarray | list[float]) -> None:
        """Apply a movement action to the ball.

        Args:
            action: Movement command as either:
                - Single angle (float/int): Direction of movement in degrees (0° = right, 90° = down)
                  Uses default distance of 20 pixels.

        Example:
            ... engine.apply_action(45)              # Move 45°, distance=20px (default)
        """
        if isinstance(action, np.ndarray):
            action = action.tolist()

        self._pending_action = [float(action[0]), self.distance]

    def set_state(self, state: PhysicsState) -> None:
        """Set the physics state directly.

        Args:
            state: The state to set. Position will be clamped to box bounds.
        """
        x, y, z = state.position
        min_x = float(self.ball_radius)
        max_x = float(self.box_width - self.ball_radius)
        min_y = float(self.ball_radius)
        max_y = float(self.box_height - self.ball_radius)

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
            float(self.ball_radius),
            float(self.ball_radius),
            float(self.box_width - self.ball_radius),
            float(self.box_height - self.ball_radius),
        )

    def close(self) -> None:
        """Clean up resources."""
        pass
