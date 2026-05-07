"""Simple Physics Engine implementation.

This module provides a basic 2D physics engine where a ball moves
in a bounded box. The ball moves in a given direction (angle) for
a given distance. If the ball hits a wall it is clamped to the boundary.
"""

import math
from typing import Any

import numpy as np

from rlive_sim.engine.core.base_physics_engine import BasePhysicsEngine, PhysicsState, SceneObject
from rlive_sim.engine.core.registry import register_physics_backend
from rlive_sim.config import PhysicsBackend
import rlive_sim.config.basic_config as basic_cfg
import rlive_sim.config.bolt_config as bolt_cfg
from rlive_common.utils import get_logger

logger = get_logger(__name__)


@register_physics_backend(PhysicsBackend.SIMPLE)
class SimplePhysicsEngine(BasePhysicsEngine):
    """Simple 2D physics engine with a ball in a box.

    Models the SpheroBolt+ movement in a step-based, instant-teleport fashion.
    Each action `[heading_delta, speed, duration]` is a single macro-step:
    the robot teleports to the new position immediately, matching the real
    robot's discrete roll-command behaviour.

    Heading is **relative**: `action[0]` is an offset added to the current
    absolute heading, matching `SpheroBoltPlus.move()` semantics.

    After every step the robot is considered standing still (``velocity = 0``).
    On wall collision the ball is clamped to the boundary (stop, not bounce).

    Attributes:
        box_width: Width of the bounding box in meters.
        box_height: Height of the bounding box in meters.
        ball_radius: Radius of the ball in meters.
        dt: Simulation timestep in seconds (kept for interface compatibility).
        speed_to_distance_factor: Metres per speed-unit per second.
            ``distance_m = factor * speed_cmd * duration_s``

    Example::

        engine = SimplePhysicsEngine(box_width=1.0, box_height=1.0)
        state = engine.reset()

        # [relative_heading_delta, speed_0_255, duration_s]
        # Turn 90° clockwise (right), roll at half speed for 1 s
        new_state = engine.update([90, 128, 1.0])
        print(new_state.position[:2])  # [x, y]
    """

    def __init__(
        self,
        box_width: float = basic_cfg.BOX_WIDTH,
        box_height: float = basic_cfg.BOX_HEIGHT,
        ball_radius: float = bolt_cfg.RADIUS_M,
        dt: float = 0.01,
        speed_to_distance_factor: float = basic_cfg.SPEED_TO_DISTANCE_FACTOR,
        seed: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the simple physics engine.

        Args:
            box_width: Width of the bounding box in metres.
            box_height: Height of the bounding box in metres.
            ball_radius: Radius of the ball in metres.
            dt: Simulation timestep (kept for interface compatibility).
            speed_to_distance_factor: Metres per speed-unit per second.
                ``distance_m = factor * speed_cmd * duration_s``
            seed: Optional seed for the internal random number generator used
                to draw random start positions on reset.
            **kwargs: Additional arguments passed to the base class.
        """
        super().__init__(dt=dt, gravity=[0.0, 0.0, 0.0])  # No gravity in 2D top-down

        self.box_width = box_width
        self.box_height = box_height
        self.ball_radius = ball_radius
        self.speed_to_distance_factor = speed_to_distance_factor

        # --- Internal robot state (single source of truth) ---
        self._heading_deg: float = 0.0                   # absolute world heading [0, 360)
        self._world_position: np.ndarray = np.zeros(3)   # [x, y, z] metres
        self._rng: np.random.Generator = np.random.default_rng(seed)  # for random start positions

        self._state: PhysicsState = PhysicsState()
        self._static_objects: list[SceneObject] = []

    # ------------------------------------------------------------------ #
    #  Scene setup                                                       #
    # ------------------------------------------------------------------ #

    def setup_scene(self, scene_config: dict[str, Any]) -> None:
        if "box_width" in scene_config:
            self.box_width = float(scene_config["box_width"])
        if "box_height" in scene_config:
            self.box_height = float(scene_config["box_height"])
        if "ball_radius" in scene_config:
            self.ball_radius = float(scene_config["ball_radius"])

        w, h = float(self.box_width), float(self.box_height)
        self._static_objects = [
            SceneObject(
                id="eurobox",
                position=[0.0, 0.0, 0.0],
                rotation=[0.0, 0.0, 0.0],
                dimensions=[w, h, 0.0],
            )
        ]

    # ------------------------------------------------------------------ #
    #  Core interface                                                     #
    # ------------------------------------------------------------------ #

    def reset(self, initial_state: PhysicsState | None = None) -> PhysicsState:
        """Reset the robot to a position and heading.

        When *initial_state* is provided the robot is placed at that exact
        position (clamped to box bounds) with the given heading (``rotation[2]``).
        When *None*, a **random** position inside the box is sampled uniformly
        and the heading is set to 0°.

        Args:
            initial_state: Optional explicit start state. ``rotation[2]`` is
                used as the starting absolute heading (yaw). Position is clamped
                to box bounds.

        Returns:
            PhysicsState: Initial state after reset.
        """
        if initial_state is not None:
            if initial_state.position[2] != 0.0:
                logger.warning("The z-position should be 0.0, got %.3f", initial_state.position[2])
            self.set_state(initial_state)
        else:
            # Random start position uniformly drawn from the valid box interior
            min_x, min_y, max_x, max_y = self.get_bounds()
            x = float(self._rng.uniform(min_x, max_x))
            y = float(self._rng.uniform(min_y, max_y))
            self._heading_deg = float(self._rng.uniform(0.0, 360.0))
            self._world_position = np.array([x, y, 0.0])
            self._state = PhysicsState(
                position=self._world_position.tolist(),
                rotation=[0.0, 0.0, self._heading_deg],
            )
            logger.debug("Random start: pos=(%.3f, %.3f) heading=%.1f°", x, y, self._heading_deg)

        return self._state

    def update(self, action: np.ndarray | list[float], dt: float | None = None) -> PhysicsState:
        """Advance the simulation by one macro-step.

        Computes the new world position via vector math (instant teleport),
        clamps to box boundaries, and returns the updated state.
        Velocity is always zero (robot stands still between commands).

        Args:
            action: `[heading_delta, speed_0_255, duration_s]`.
                `heading_delta` is a **relative** offset added to the current
                absolute heading, matching `SpheroBoltPlus.move()` semantics.
            dt: Ignored (step-based engine).

        Returns:
            PhysicsState: Updated state. `rotation[2]` holds the new absolute
            heading in degrees.
        """
        action = np.asarray(action, dtype=np.float64)

        heading_delta = float(action[0])
        raw_speed     = float(action[1])
        duration      = float(action[2]) if len(action) > 2 else 1.0

        # 1. Bolt API is CW-positive; _heading_deg is CCW-positive (right-hand rule
        #    around +Z in X-right / Y-up / Z-into-camera world coords).
        #    Subtract the CW Bolt delta to accumulate in CCW convention.
        self._heading_deg = (self._heading_deg - heading_delta) % 360.0

        # 2. Speed → scalar distance for this step
        clamped_speed = float(np.clip(raw_speed, 0.0, 255.0))
        distance = clamped_speed * self.speed_to_distance_factor * duration

        # 3. Standard CCW direction vector in Y-up world coords:
        #    0° = +X (right), 90° CCW = +Y (up/forward), 180° = -X (left), 270° CCW = -Y (down)
        angle_rad = math.radians(self._heading_deg)
        direction = np.array([math.cos(angle_rad), math.sin(angle_rad)])

        # 4. Displacement and new 2-D position
        new_pos_2d = self._world_position[:2] + direction * distance

        # 5. Clamp to box bounds (wall = stop)
        min_x, min_y, max_x, max_y = self.get_bounds()
        bounds_lo = np.array([min_x, min_y])
        bounds_hi = np.array([max_x, max_y])
        new_pos_2d = np.clip(new_pos_2d, bounds_lo, bounds_hi)

        # 6. Persist world position
        self._world_position = np.array([new_pos_2d[0], new_pos_2d[1], self._world_position[2]])

        self._state = PhysicsState(
            position=self._world_position.tolist(),
            rotation=[0.0, 0.0, self._heading_deg],
        )
        logger.debug(
            "Step: heading=%.1f° distance=%.3f m → pos=(%.3f, %.3f)",
            self._heading_deg, distance, new_pos_2d[0], new_pos_2d[1],
        )
        return self._state

    def get_state(self) -> PhysicsState:
        """Return the current physics state."""
        return self._state

    def set_state(self, state: PhysicsState) -> None:
        """Set the physics state directly (e.g. for teleportation or tests).

        Keeps ``_world_position`` and ``_heading_deg`` in sync with ``_state``.

        Args:
            state: Target state. ``position`` is clamped to box bounds.
                ``rotation[2]`` becomes the new absolute heading.
        """
        x, y, z = state.position
        min_x, min_y, max_x, max_y = self.get_bounds()
        x = float(np.clip(x, min_x, max_x))
        y = float(np.clip(y, min_y, max_y))

        self._world_position = np.array([x, y, z])
        self._heading_deg = float(state.rotation[2]) % 360.0

        self._state = PhysicsState(
            position=self._world_position.tolist(),
            velocity=state.velocity,
            rotation=[
                float(state.rotation[0]),
                float(state.rotation[1]),
                self._heading_deg,
            ],
            angular_velocity=state.angular_velocity,
            extra=state.extra,
        )

    # ------------------------------------------------------------------ #
    #  Scene objects                                                      #
    # ------------------------------------------------------------------ #

    def get_scene_objects(self) -> list[SceneObject]:
        """Return the dynamic scene objects (the robot ball)."""
        return [
            SceneObject(
                id="robot",
                position=self._state.position,
                rotation=self._state.rotation,
                dimensions=[self.ball_radius],
            )
        ]

    def get_static_scene_objects(self) -> list[SceneObject]:
        """Return the static background objects (eurobox walls)."""
        return self._static_objects

    # ------------------------------------------------------------------ #
    #  Bounds                                                             #
    # ------------------------------------------------------------------ #

    def get_bounds(self) -> tuple[float, float, float, float]:
        """Get the valid bounds for ball position (inner edge of walls).

        Returns:
            tuple[float, float, float, float]: ``(min_x, min_y, max_x, max_y)``.
        """
        r = float(self.ball_radius)
        return (
            -float(self.box_width)  / 2.0 + r,
            -float(self.box_height) / 2.0 + r,
             float(self.box_width)  / 2.0 - r,
             float(self.box_height) / 2.0 - r,
        )

    def close(self) -> None:
        """Clean up resources."""
        pass
