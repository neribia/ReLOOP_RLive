"""MuJoCo Integrated Engine implementation.

This module provides an integrated simulation engine that combines physics and
rendering in a single MuJoCo simulation loop for maximum efficiency.
"""

from pathlib import Path
from typing import Any

import numpy as np

from rlive_sim.engine.base_integrated_engine import BaseIntegratedEngine
from rlive_sim.engine.base_physics_engine import PhysicsState
from rlive_sim.engine.registry import register_integrated_backend
from rlive_sim.config import IntegratedBackend


@register_integrated_backend(IntegratedBackend.MUJOCO)
class MujocoIntegratedEngine(BaseIntegratedEngine):
    """MuJoCo integrated engine combining physics and rendering.

    Simulates a SpheroBolt+ sphere rolling in an 80x60cm box with realistic
    physics and renders the scene from a fixed top-down camera view.

    This is a true integrated engine that combines physics simulation and
    rendering in a single loop for efficiency, unlike the separate engine
    approach.

    Action space: [relative_heading_degrees, speed_0_255, duration_seconds]
    - relative_heading: Rotation angle in degrees (0-360), accumulated each step
    - speed: Velocity magnitude (0-255 scale, normalized to m/s)
    - duration: Duration of movement in seconds

    Attributes:
        model: MuJoCo model instance.
        data: MuJoCo data instance (simulation state).
        renderer: MuJoCo renderer instance.
        dt: Simulation timestep in seconds.
        gravity: Gravity vector [gx, gy, gz].
        width: Image width in pixels.
        height: Image height in pixels.
        channels: Number of color channels.
        spherobolt_speed: Default speed value (0-255).
        spherobolt_duration: Default duration value (seconds).
        _absolute_heading: Accumulated heading angle (0-360°).
        _last_image: Last rendered image (cached).

    Methods:
        reset(initial_state): Reset the simulation to initial conditions.
        update_and_render(action, dt): Advance physics and render in one call.
        get_state(): Get the current physics state.
        set_state(state): Directly set the physics state.
        get_resolution(): Get the render resolution as (height, width, channels).
        setup_scene(scene_config): Set up the scene configuration.
        close(): Clean up resources.
    """

    def __init__(
        self,
        dt: float = 0.01,
        gravity: list[float] | None = None,
        width: int = 640,
        height: int = 480,
        channels: int = 3,
        scene_path: str | None = None,
        spherobolt_speed: float = 50.0,
        spherobolt_duration: float = 1.0,
        **kwargs: Any,
    ) -> None:
        """Initialize the MuJoCo integrated engine.

        Args:
            dt: Simulation timestep in seconds. Defaults to 0.01.
            gravity: Gravity vector [gx, gy, gz]. Defaults to [0, 0, -9.81].
            width: Image width in pixels. Defaults to 640.
            height: Image height in pixels. Defaults to 480.
            channels: Number of color channels (RGB=3). Defaults to 3.
            scene_path: Path to MJCF model file. If None, uses default spherobolt_box.xml.
            spherobolt_speed: Default speed value (0-255). Defaults to 50.
            spherobolt_duration: Default duration value (seconds). Defaults to 1.0.
            **kwargs: Additional arguments (ignored).

        Raises:
            ValueError: If channels is not 3.
        """
        if gravity is None:
            gravity = [0.0, 0.0, -9.81]

        super().__init__(
            dt=dt,
            gravity=gravity,
            width=width,
            height=height,
            channels=channels,
        )

        if channels != 3:
            raise ValueError(f"MuJoCo integrated engine supports RGB (channels=3), got {channels}")

        # FIXME: shouldnt this be in an __init__?
        # Lazy import of mujoco
        try:
            import mujoco
        except ImportError:
            raise ImportError(
                "MuJoCo is required for MujocoIntegratedEngine. "
                "Install with: pip install mujoco>=3.1.0"
            )

        self.spherobolt_speed = spherobolt_speed
        self.spherobolt_duration = spherobolt_duration
        self._absolute_heading = 0.0  # Accumulated heading in degrees
        self._last_image: np.ndarray | None = None

        # Determine scene path
        if scene_path is None:
            scene_path = str(
                Path(__file__).parent.parent / "resources" / "spherobolt_box.xml"
            )

        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(scene_path)
        self.data = mujoco.MjData(self.model)

        # Set gravity
        self.model.opt.gravity[:] = gravity

        # Create MuJoCo renderer
        self.renderer = mujoco.Renderer(self.model, height=height, width=width)

        # Camera configuration
        self.camera_name = "top_camera"

    def reset(self, initial_state: PhysicsState | None = None) -> tuple[PhysicsState, np.ndarray]:
        """Reset the simulation to initial conditions.

        Args:
            initial_state: Optional initial state. If None, resets to default position.

        Returns:
            tuple[PhysicsState, np.ndarray]: Tuple of (initial_state, rendered_image).
        """
        import mujoco

        # Reset to default configuration
        mujoco.mj_resetData(self.model, self.data)

        # Set sphere to center of box at rest
        sphere_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "sphere")
        if sphere_body_id >= 0:
            # Reset position and velocity via joints
            # We'll just let MuJoCo keep default values from reset
            pass

        # Apply initial state if provided
        if initial_state is not None:
            self.set_state(initial_state)

        self._absolute_heading = 0.0

        # Render initial state
        state = self._extract_state()
        image = self._render_frame()

        return state, image

    def update_and_render(
        self, action: np.ndarray | list[float], dt: float | None = None
    ) -> tuple[PhysicsState, np.ndarray]:
        """Advance physics and render in a single integrated call.

        Args:
            action: Action vector [relative_heading_deg, speed_0_255, duration_sec].
                - relative_heading_deg: Relative heading in degrees (0-360).
                - speed_0_255: Speed value (0-255, normalized to m/s).
                - duration_sec: Duration of movement in seconds.
            dt: Optional timestep override. Uses self.dt if None.

        Returns:
            tuple[PhysicsState, np.ndarray]: Tuple of (updated_state, rendered_image).
        """
        import mujoco

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
        speed_ms = (speed / 255.0) * 1.0

        # Calculate velocity components based on heading
        heading_rad = np.radians(self._absolute_heading)
        vx = speed_ms * np.cos(heading_rad)
        vy = speed_ms * np.sin(heading_rad)

        # Apply velocity to sphere via joint velocities (qvel)
        # Joints: 0=sphere_x, 1=sphere_y, 2=sphere_z, 3=rot_x, 4=rot_y, 5=rot_z
        if len(self.data.qvel) >= 2:
            self.data.qvel[0] = vx  # X velocity
            self.data.qvel[1] = vy  # Y velocity

        # Simulate for the action duration
        num_steps = int(duration / self.model.opt.timestep)
        for _ in range(num_steps):
            mujoco.mj_step(self.model, self.data)

        # Extract state and render
        state = self._extract_state()
        image = self._render_frame()

        return state, image

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

            # Set rotation (quaternion)
            if state.rotation != [1.0, 0.0, 0.0, 0.0]:
                # TODO: Implement proper quaternion conversion
                pass

            # Set angular velocity
            self.data.omega[sphere_body_id][:] = state.angular_velocity

    def get_resolution(self) -> tuple[int, int, int]:
        """Get the render resolution.

        Returns:
            tuple[int, int, int]: (height, width, channels).
        """
        return (self.height, self.width, self.channels)

    def setup_scene(self, scene_config: dict[str, Any]) -> None:
        """Set up the scene configuration.

        Args:
            scene_config: Dictionary with scene configuration (currently unused).
        """
        # Scene is already set up during initialization
        pass

    def close(self) -> None:
        """Clean up resources and shutdown the integrated engine."""
        # MuJoCo doesn't require explicit cleanup in Python
        pass

    def _extract_state(self) -> PhysicsState:
        """Extract current state from MuJoCo data.

        Returns:
            PhysicsState: Extracted physics state.
        """
        import mujoco

        # Get sphere position from joint positions (qpos)
        # qpos: [x, y, z, rot_x, rot_y, rot_z]
        position = [
            float(self.data.qpos[0]) if len(self.data.qpos) > 0 else 0.0,
            float(self.data.qpos[1]) if len(self.data.qpos) > 1 else 0.0,
            float(self.data.qpos[2]) if len(self.data.qpos) > 2 else 0.0375,
        ]

        # Get velocity from joint velocities (qvel)
        velocity = [
            float(self.data.qvel[0]) if len(self.data.qvel) > 0 else 0.0,
            float(self.data.qvel[1]) if len(self.data.qvel) > 1 else 0.0,
            float(self.data.qvel[2]) if len(self.data.qvel) > 2 else 0.0,
        ]

        # Rotation (simplified - default identity)
        rotation = [1.0, 0.0, 0.0, 0.0]

        # Angular velocity
        angular_velocity = [
            float(self.data.qvel[3]) if len(self.data.qvel) > 3 else 0.0,
            float(self.data.qvel[4]) if len(self.data.qvel) > 4 else 0.0,
            float(self.data.qvel[5]) if len(self.data.qvel) > 5 else 0.0,
        ]

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

    def _render_frame(self) -> np.ndarray:
        """Render a frame from the current simulation state.

        Returns:
            np.ndarray: Rendered image as uint8 array with shape (height, width, 3).
        """
        import mujoco

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
