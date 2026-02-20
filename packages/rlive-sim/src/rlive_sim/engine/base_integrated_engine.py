"""Base Integrated Engine module.

This module provides the abstract base class for integrated simulation engines
that combine both physics and rendering in a single monolithic system.

Integrated engines are ideal when physics simulation and visualization are
tightly coupled and benefit from shared resources and optimized execution.

Supported integrated backends (via subclasses):
    - Godot: Open-source game engine with Python bindings
    - Unity: Via ML-Agents or custom Python bridge
    - MuJoCo: When using built-in rendering
    - Isaac Sim: NVIDIA robotics simulation
    - Unreal Engine: Via UnrealCV or custom bridge

Contrast with separate engines:
    - PhysicsEngine: Pure physics simulation
    - RenderEngine: Pure rendering/observation generation
    - SimulationEngine: Orchestrator that combines separate engines
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np

from rlive_sim.engine.base_physics_engine import PhysicsState

if TYPE_CHECKING:
    from rlive_sim.config import IntegratedConfig


class BaseIntegratedEngine(ABC):
    """Abstract base class for integrated simulation engines.

    This class defines the interface for integrated simulation platforms where
    physics and rendering are tightly coupled and executed together. It does
    not inherit from BasePhysicsEngine or BaseRenderEngine; instead, it defines
    its own comprehensive interface that combines both domains.

    Use this for integrated simulation platforms like Godot, MuJoCo, or Isaac Sim
    where physics and rendering are naturally coupled and benefit from being
    executed together for efficiency.

    For simulations where physics and rendering are independent, prefer using
    separate PhysicsEngine and RenderEngine with SimulationEngine orchestrator.

    Interface includes:
        - Physics simulation: reset(), update(), get_state(), set_state()
        - Rendering: render(), set_camera_pose()
        - Integrated: update_and_render() for efficient combined updates
        - Scene management: setup_scene(), load_scene() (optional)

    Attributes:
        dt: Simulation timestep in seconds.
        gravity: Gravity vector [gx, gy, gz].
        width: Image width in pixels.
        height: Image height in pixels.
        channels: Number of color channels.

    Methods:
        reset(initial_state): Reset the simulation. Returns tuple[PhysicsState, np.ndarray].
        update_and_render(action, dt): Advance physics and render together. Returns tuple[PhysicsState, np.ndarray].
        get_resolution(): Get the render resolution as (height, width, channels).
        setup_scene(scene_config): Set up the scene with given configuration.
        close(): Clean up resources and shutdown the engine.

    Examples:
        Using an integrated engine with efficient combined updates:

            from rlive_sim.config import IntegratedConfig, IntegratedBackend
            from rlive_sim.engine import IntegratedEngine

            config = IntegratedConfig(backend=IntegratedBackend.MUJOCO)
            engine = IntegratedEngine(config)
            state = engine.reset()

            for step_idx in range(100):
                action = [1.0]  # Move forward
                state, image = engine.update_and_render(action)

                # Process state and image together
                reward = compute_reward(state)
                if should_visualize:
                    display_image(image)
    """

    def __init__(
        self,
        dt: float = 0.01,
        gravity: list[float] | None = None,
        width: int = 640,
        height: int = 480,
        channels: int = 3,
    ) -> None:
        """Initialize the integrated engine.

        Args:
            dt: Simulation timestep in seconds. Defaults to 0.01 (100 Hz).
            gravity: Gravity vector [gx, gy, gz]. Defaults to [0, 0, -9.81].
            width: Image width in pixels. Defaults to 640.
            height: Image height in pixels. Defaults to 480.
            channels: Number of color channels. Defaults to 3 (RGB).
        """
        self.dt = dt
        self.gravity = gravity
        self.width = width
        self.height = height
        self.channels = channels

    # FIXME: What does reset need?
    @abstractmethod
    def reset(self, initial_state: PhysicsState | None = None) -> tuple[PhysicsState, np.ndarray]:
        """Reset the simulation to initial conditions.

        Args:
            initial_state: Optional initial state. Uses engine default if None.

        Returns:
            tuple[PhysicsState, np.ndarray]: Tuple of (updated_state, rendered_image).
                The rendered_image has shape (height, width, channels) as uint8.
        """
        pass

    @abstractmethod
    def update_and_render(self, action: np.ndarray | list[float], dt: float | None = None) -> tuple[PhysicsState, np.ndarray]:
        """Advance physics and render in a single integrated call.

        This is the core method for integrated engines. It performs physics
        simulation and rendering together, which is more efficient than doing
        them separately as they can share intermediate computations.

        Args:
            action: Action vector to apply.
            dt: Optional timestep override. Uses self.dt if None.

        Returns:
            tuple[PhysicsState, np.ndarray]: Tuple of (updated_state, rendered_image).
                The rendered_image has shape (height, width, channels) as uint8.

        Examples:
            Efficient integrated engine simulation loop:

                engine = IntegratedEngine(config)
                engine.reset()

                for step_idx in range(1000):
                    engine.apply_action([forward, rotate])
                    physics_state, observation_image = engine.step_and_render()

                    # Use state and image together
                    reward = calculate_reward(physics_state)
                    terminated = check_termination(physics_state)

                    if render_mode == "human":
                        cv2.imshow("Simulation", observation_image)
        """
        pass

    @abstractmethod
    def get_resolution(self) -> tuple[int, int, int]:
        """Get the render resolution.

        Returns:
            tuple[int, int, int]: (height, width, channels).
        """
        pass

    # TODO: Rewrite docstring
    @abstractmethod
    def setup_scene(self, scene_config: dict[str, Any]) -> None:
        """Load a pre-built scene file.

        Initializes the simulation with a scene loaded from disk. Scene format
        depends on the integrated backend (Godot .tscn, MuJoCo .xml, etc.).

        Call before reset() to set up the initial scene configuration.

        Args:
            scene_config: Backend-specific scene configuration describing the
                scene to construct. Typically a mapping or configuration object
                that specifies assets, initial object placement, lighting, and
                physics settings required by the integrated backend.

        Raises:
            FileNotFoundError: If scene file does not exist.
            ValueError: If scene file format is invalid for this engine.

        Examples:
            Loading and configuring scenes:

                engine = IntegratedEngine(config)

                # Load base scene
                engine.load_scene("simulations/ball_tracking.xml")

                # Reset to scene defaults
                state = engine.reset()

                # Can spawn additional objects on top of scene
                agent = engine.spawn_object("sphere", position=[0, 1, 0])
        """
        pass

    def close(self) -> None:
        """Clean up resources and shutdown the integrated engine.

        Override in subclasses to properly teardown engine resources.
        """
        pass
