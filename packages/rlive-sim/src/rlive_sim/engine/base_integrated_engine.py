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
        - Physics simulation: step(), apply_action(), reset(), get_state(), set_state()
        - Rendering: render(), set_camera(), get_image()
        - Integrated: step_and_render() for efficient combined updates
        - Scene management: load_scene(), spawn_object(), remove_object()

    Attributes:
        dt: Simulation timestep in seconds.
        gravity: Gravity vector [gx, gy, gz].
        width: Image width in pixels.
        height: Image height in pixels.
        channels: Number of color channels.

    Examples:
        Using an integrated engine with efficient combined updates:

            from rlive_sim.config import IntegratedConfig, IntegratedBackend
            from rlive_sim.engine import IntegratedEngine

            config = IntegratedConfig(backend=IntegratedBackend.MUJOCO)
            engine = IntegratedEngine(config)
            state = engine.reset()

            for step_idx in range(100):
                action = [1.0, 0.0]  # Move forward
                engine.apply_action(action)
                state, image = engine.step_and_render()

                # Process state and image together
                reward = compute_reward(state)
                if should_visualize:
                    display_image(image)

        Loading and manipulating scenes:

            engine.load_scene("path/to/scene.xml")

            # Spawn dynamic objects
            ball_id = engine.spawn_object(
                "sphere",
                position=[0.0, 1.0, 0.0],
                radius=0.5
            )

            # Remove when done
            engine.remove_object(ball_id)
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

    @abstractmethod
    def step_and_render(self, dt: float | None = None) -> tuple[PhysicsState, np.ndarray]:
        """Advance physics and render in a single integrated call.

        This is the core method for integrated engines. It performs physics
        simulation and rendering together, which is more efficient than doing
        them separately as they can share intermediate computations.

        Args:
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
    def load_scene(self, scene_path: str) -> None:
        """Load a pre-built scene file.

        Initializes the simulation with a scene loaded from disk. Scene format
        depends on the integrated backend (Godot .tscn, MuJoCo .xml, etc.).

        Call before reset() to set up the initial scene configuration.

        Args:
            scene_path: Absolute or relative path to the scene file.
                For Godot: .tscn or .escn files
                For MuJoCo: .xml files with physics definitions
                For Isaac Sim: .usd files

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

    @abstractmethod
    def spawn_object(
        self,
        object_type: str,
        position: list[float],
        rotation: list[float] | None = None,
        **kwargs: Any,
    ) -> str:
        """Spawn a new object into the active scene.

        Dynamically creates a new physics/visual object in the simulation.
        Returns a unique ID that can be used for later reference/manipulation.

        Args:
            object_type: Type name for the object (e.g., "sphere", "box", "cylinder").
                Supported types depend on the integrated backend.
            position: Initial position [x, y, z] in world coordinates.
            rotation: Optional rotation as quaternion [w, x, y, z].
                Defaults to identity [1, 0, 0, 0] (no rotation).
            **kwargs: Additional backend-specific parameters such as:
                - size: Dimensions for box/cylinder shapes
                - radius: Radius for sphere
                - color: RGB color [r, g, b] in [0, 1] range
                - material: Physics material (friction, restitution, etc.)
                - static: Whether object is kinematic/static
                - mass: Object mass for dynamic objects

        Returns:
            str: Unique object identifier for reference in future calls.

        Raises:
            ValueError: If object_type is not supported.

        Examples:
            Spawning various object types with different properties:

                # Simple sphere
                ball = engine.spawn_object(
                    "sphere",
                    position=[0.0, 2.0, 0.0],
                    radius=0.5,
                    color=[1.0, 0.0, 0.0]
                )

                # Static ground box
                ground = engine.spawn_object(
                    "box",
                    position=[0.0, -1.0, 0.0],
                    size=[10.0, 0.5, 10.0],
                    static=True,
                    color=[0.5, 0.5, 0.5]
                )

                # Rotated cylinder
                pole = engine.spawn_object(
                    "cylinder",
                    position=[3.0, 0.0, 0.0],
                    rotation=[0.707, 0.0, 0.707, 0.0],  # 90° around Y
                    radius=0.1,
                    height=2.0
                )
        """
        pass

    @abstractmethod
    def remove_object(self, object_id: str) -> None:
        """Remove an object from the active scene.

        Deletes a previously spawned object and frees associated resources
        (physics body, visuals, memory).

        Args:
            object_id: Unique identifier returned by spawn_object().

        Raises:
            ValueError: If object_id does not exist or is invalid.

        Examples:
            Object lifecycle management:

                # Spawn temporary obstacle
                obstacle = engine.spawn_object(
                    "box",
                    position=[5.0, 0.0, 0.0],
                    size=[0.5, 2.0, 0.5]
                )

                # ... run simulation ...

                # Clean up when done
                engine.remove_object(obstacle)

                # Spawn does not return None; error if object already removed
                try:
                    engine.remove_object(obstacle)  # Will raise ValueError
                except ValueError:
                    print("Object already removed")
        """
        pass

    def close(self) -> None:
        """Clean up resources and shutdown the integrated engine.

        Override in subclasses to properly teardown engine resources.
        """
        pass
