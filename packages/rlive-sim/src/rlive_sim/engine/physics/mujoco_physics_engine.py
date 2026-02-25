"""MuJoCo Physics Engine implementation.

This module provides a physics engine using MuJoCo (Multi-Joint dynamics with Contact).
It simulates a SpheroBolt+ sphere rolling in a box with realistic physics.
"""

from pathlib import Path
from typing import Any

import numpy as np

from rlive_sim.engine.base_physics_engine import BasePhysicsEngine, PhysicsState
from rlive_sim.engine.registry import register_physics_backend
from rlive_sim.config import PhysicsBackend


@register_physics_backend(PhysicsBackend.MUJOCO)
class MujocoPhysicsEngine(BasePhysicsEngine):
    """MuJoCo physics engine for SpheroBolt+ simulation.

    Simulates a SpheroBolt+ sphere (radius 37.5mm, mass 0.12kg) rolling in an
    80x60cm box with gravity and friction.

    Action space: [relative_heading_degrees, speed_0_255, duration_seconds]
    - relative_heading: Rotation angle in degrees (0-360), accumulated each step
    - speed: Velocity magnitude (0-255 scale, normalized to m/s)
    - duration: Duration of movement in seconds

    Attributes:
        model: MuJoCo model instance.
        data: MuJoCo data instance (simulation state).
        dt: Simulation timestep in seconds.
        gravity: Gravity vector [gx, gy, gz].
        spherobolt_speed: Default speed value (0-255).
        spherobolt_duration: Default duration value (seconds).
        _absolute_heading: Accumulated heading angle (0-360°).
        _scene_path: Path to MJCF model file.

    Methods:
        reset(initial_state): Reset the simulation to initial conditions.
        update(action, dt): Advance the simulation by one timestep.
        get_state(): Get the current physics state.
        set_state(state): Directly set the physics state.
        close(): Clean up resources.
    """

    def __init__(
        self,
        dt: float = 0.01,
        gravity: list[float] | None = None,
        scene_path: str | None = None,
        spherobolt_speed: float = 50.0,
        spherobolt_duration: float = 1.0,
        **kwargs: Any,
    ) -> None:
        """Initialize the MuJoCo physics engine.

        Args:
            dt: Simulation timestep in seconds. Defaults to 0.01.
            gravity: Gravity vector [gx, gy, gz]. Defaults to [0, 0, -9.81].
            scene_path: Path to MJCF model file. If None, uses default spherobolt_box.xml.
            spherobolt_speed: Default speed value (0-255). Defaults to 50.
            spherobolt_duration: Default duration value (seconds). Defaults to 1.0.
            **kwargs: Additional arguments (ignored).
        """
        # Lazy import of mujoco
        try:
            import mujoco
        except ImportError:
            raise ImportError(
                "MuJoCo is required for MujocoPhysicsEngine. "
                "Install with: pip install mujoco>=3.1.0"
            )

        if gravity is None:
            gravity = [0.0, 0.0, -9.81]

        super().__init__(dt=dt, gravity=gravity)

        self.spherobolt_speed = spherobolt_speed
        self.spherobolt_duration = spherobolt_duration
        self._absolute_heading = 0.0  # Accumulated heading in degrees
        self._last_action_time = 0.0

        # Determine scene path
        if scene_path is None:
            scene_path = str(
                Path(__file__).parent.parent / "resources" / "spherobolt_box.xml"
            )
        self._scene_path = scene_path

        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(self._scene_path)
        self.data = mujoco.MjData(self.model)

        # Set gravity
        self.model.opt.gravity[:] = gravity

        # Initialize state
        self._state = self._extract_state()

    def reset(self, initial_state: PhysicsState | None = None) -> PhysicsState:
        """Reset the simulation to initial conditions.

        Args:
            initial_state: Optional initial state. If None, resets to default position.

        Returns:
            PhysicsState: Initial state after reset.
        """
        # Reset to default configuration
        mujoco.mj_resetData(self.model, self.data)

        # Set sphere to center of box at rest
        # Position indices for sphere body
        sphere_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "sphere")
        if sphere_body_id >= 0:
            self.data.xpos[sphere_body_id][:] = [0.0, 0.0, 0.0375]
            self.data.xvel[sphere_body_id][:] = [0.0, 0.0, 0.0]
            self.data.omega[sphere_body_id][:] = [0.0, 0.0, 0.0]

        # Apply initial state if provided
        if initial_state is not None:
            self.set_state(initial_state)

        self._absolute_heading = 0.0
        self._last_action_time = 0.0
        self._state = self._extract_state()
        return self._state

    def update(self, action: np.ndarray | list[float], dt: float | None = None) -> PhysicsState:
        """Advance the simulation by one timestep with action.

        Args:
            action: Action vector [relative_heading_deg, speed_0_255, duration_sec].
                - relative_heading_deg: Relative heading in degrees (0-360).
                - speed_0_255: Speed value (0-255, normalized to m/s).
                - duration_sec: Duration of movement in seconds.
            dt: Optional timestep override. Uses self.dt if None.

        Returns:
            PhysicsState: Updated physics state after simulation step.
        """
        if dt is None:
            dt = self.dt

        # Parse action
        if isinstance(action, (list, tuple)):
            action = np.array(action, dtype=np.float32)

        # Extract action components
        relative_heading = float(action[0]) if len(action) > 0 else 0.0
        speed = float(action[1]) if len(action) > 1 else self.spherobolt_speed
        duration = float(action[2]) if len(action) > 2 else self.spherobolt_duration

        # Update absolute heading (accumulate relative heading)
        self._absolute_heading = (self._absolute_heading + relative_heading) % 360.0

        # Convert speed from 0-255 scale to m/s (approx 0-1 m/s)
        # Normalized: speed_ms = (speed / 255) * 1.0
        speed_ms = (speed / 255.0) * 1.0

        # Calculate velocity components based on heading
        heading_rad = np.radians(self._absolute_heading)
        vx = speed_ms * np.cos(heading_rad)
        vy = speed_ms * np.sin(heading_rad)

        # Apply velocity to sphere
        sphere_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "sphere")
        if sphere_body_id >= 0:
            self.data.xvel[sphere_body_id][0] = vx
            self.data.xvel[sphere_body_id][1] = vy
            # Keep z velocity from gravity/physics

        # Simulate for the action duration
        num_steps = int(duration / self.model.opt.timestep)
        for _ in range(num_steps):
            mujoco.mj_step(self.model, self.data)

        self._state = self._extract_state()
        self._last_action_time += duration
        return self._state

    def get_state(self) -> PhysicsState:
        """Get the current physics state.

        Returns:
            PhysicsState: Current simulation state.
        """
        return self._extract_state()

    def set_state(self, state: PhysicsState) -> None:
        """Directly set the physics state.

        Args:
            state: PhysicsState object to apply to the simulation.
        """
        sphere_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "sphere")
        if sphere_body_id >= 0:
            # Set position
            self.data.xpos[sphere_body_id][:] = state.position

            # Set velocity
            self.data.xvel[sphere_body_id][:] = state.velocity

            # Set rotation (quaternion) - MuJoCo uses xquat
            # Convert quaternion to MuJoCo format if needed
            # For now, use identity if rotation is identity
            if state.rotation != [1.0, 0.0, 0.0, 0.0]:
                # TODO: Implement proper quaternion conversion
                pass

            # Set angular velocity
            self.data.omega[sphere_body_id][:] = state.angular_velocity

        self._state = state

    def _extract_state(self) -> PhysicsState:
        """Extract current state from MuJoCo data.

        Returns:
            PhysicsState: Extracted physics state.
        """
        sphere_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "sphere")

        if sphere_body_id < 0:
            # Fallback to default state
            return PhysicsState()

        # Get sphere position
        position = self.data.xpos[sphere_body_id].copy().tolist()

        # Get sphere velocity
        velocity = self.data.xvel[sphere_body_id].copy().tolist()

        # Get sphere rotation (quaternion from MuJoCo)
        rotation = self.data.xquat[sphere_body_id].copy().tolist()

        # Get sphere angular velocity
        angular_velocity = self.data.omega[sphere_body_id].copy().tolist()

        return PhysicsState(
            position=position,
            velocity=velocity,
            rotation=rotation,
            angular_velocity=angular_velocity,
            extra={
                "heading_degrees": self._absolute_heading,
                "time": self.data.time,
            },
        )

    def close(self) -> None:
        """Clean up resources."""
        # MuJoCo doesn't require explicit cleanup in Python
        pass



