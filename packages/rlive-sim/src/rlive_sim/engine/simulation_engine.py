"""Simulation Engine module.

This module provides the `SimulationEngine` class that orchestrates physics
and rendering, supporting both integrated engines and separate physics/render
engine combinations.

Architecture:
    - Integrated engines (Godot, MuJoCo): Physics + render tightly coupled
    - Separate engines (PyMunk + OpenCV): Independent physics and rendering
    - SimulationEngine: Unified orchestrator for both approaches
"""

from typing import Any

import numpy as np

from rlive_sim.engine.core.base_physics_engine import BasePhysicsEngine, PhysicsState
from rlive_sim.engine.core.base_render_engine import BaseRenderEngine
from rlive_sim.engine.core.base_integrated_engine import BaseIntegratedEngine
from rlive_sim.engine.core.combined_engine import CombinedEngine


# TODO: Create a class, that gathers all information created during a step??


class SimulationEngine:
    """Orchestrates physics and rendering for simulation environments.

    This class provides a unified API for working with different engine
    architectures:

    1. **Integrated engines** (Godot, MuJoCo, Isaac Sim):
       Physics and rendering are tightly coupled in a single engine.
       Use these for maximum efficiency when physics+render can share resources.

    2. **Separate engines** (PyMunk + OpenCV, etc.):
       Physics and rendering are independent. Useful for flexibility or when
       using best-of-breed solutions for each component.

    SimulationEngine automatically adapts both approaches to a common interface,
    so client code doesn't need to know which architecture is being used.

    Internally, separate engines are wrapped in CombinedEngine to expose the
    integrated engine interface, allowing uniform method handling.

    Attributes:
        _engine: Internal integrated engine (either passed directly or created by wrapping
            separate engines). Implements BaseIntegratedEngine interface.

    Methods:
        reset(initial_state): Reset the simulation to initial conditions. Returns (state, image).
        update_and_render(action, dt): Advance physics and render in one call. Returns (state, image).
        get_resolution(): Get the render resolution as (height, width, channels).
        setup_scene(scene_config): Set up the scene with given configuration.
        close(): Clean up all engine resources.

    Examples:
        Using separate engines:

            physics = SimplePhysicsEngine(box_width=640, box_height=480)
            renderer = OpenCVRenderEngine(width=640, height=480)
            sim = SimulationEngine(physics_engine=physics, render_engine=renderer)

            state = sim.reset()
            state = sim.update_and_render()

        Using an integrated engine:

            from rlive_sim.config import IntegratedConfig, IntegratedBackend
            from rlive_sim.engine import IntegratedEngine

            config = IntegratedConfig(backend=IntegratedBackend.MUJOCO)
            integrated = IntegratedEngine(config)
            sim = SimulationEngine(integrated_engine=integrated)

            state = sim.reset()
            sim.apply_action([1.0])
            state, image = sim.simulate_and_render()  # Efficient combined call
    """

    def __init__(
        self,
        physics_engine: BasePhysicsEngine | None = None,
        render_engine: BaseRenderEngine | None = None,
        integrated_engine: BaseIntegratedEngine | None = None,
    ) -> None:
        """Initialize the simulation engine.

        Provide either an `integrated_engine` OR both `physics_engine` and
        `render_engine`. Cannot mix integrated and separate engines.

        Args:
            physics_engine: Physics engine instance. Required if not using
                integrated engine.
            render_engine: Render engine instance. Required if not using
                integrated engine.
            integrated_engine: Integrated engine instance. Mutually exclusive
                with physics_engine and render_engine.

        Raises:
            ValueError: If both integrated and separate engines are provided,
                or if neither is provided, or if only one of physics/render
                is provided.

        Examples:
            Valid instantiations:

                # Integrated engine
                sim1 = SimulationEngine(integrated_engine=godot_engine)

                # Separate engines
                sim2 = SimulationEngine(
                    physics_engine=pymunk_engine,
                    render_engine=opencv_engine
                )

                # Invalid - will raise ValueError:
                # sim3 = SimulationEngine(
                #     physics_engine=pymunk,
                #     integrated_engine=godot
                # )
        """
        # Validate arguments
        has_integrated = integrated_engine is not None
        has_physics = physics_engine is not None
        has_render = render_engine is not None

        if has_integrated and (has_physics or has_render):
            raise ValueError(
                "Cannot provide both integrated_engine and physics_engine/render_engine. "
                "Use either an integrated engine OR separate physics and render engines."
            )

        if not has_integrated and not (has_physics and has_render):
            if has_physics or has_render:
                raise ValueError(
                    "Must provide both physics_engine and render_engine when not "
                    "using an integrated engine."
                )
            raise ValueError(
                "Must provide either an integrated_engine OR both physics_engine and "
                "render_engine."
            )

        # Wrap separate engines in CombinedEngine to unify interface
        if integrated_engine is not None:
            self._engine = integrated_engine
        else:
            self._engine = CombinedEngine(physics_engine, render_engine)

        # Cache for last state (used for rendering in separate mode)
        self._last_state: PhysicsState | None = None

    def reset(self, initial_state: PhysicsState | None = None) -> tuple[PhysicsState, np.ndarray]:
        """Reset the simulation to initial conditions.

        Args:
            initial_state: Optional initial state. Uses engine default if None.

        Returns:
            tuple[PhysicsState, np.ndarray]: Tuple of (initial_state, rendered_image).
        """
        state, image = self._engine.reset(initial_state)
        self._last_state = state
        return state, image

    def update_and_render(
        self, action: np.ndarray | list[float], dt: float | None = None
    ) -> tuple[PhysicsState, np.ndarray]:
        """Advance physics and render in a single call.

        This is more efficient for unified engines.

        Args:
            action: The action vector to apply.
            dt: Optional timestep override.

        Returns:
            tuple[PhysicsState, np.ndarray]: Updated state and rendered image.
        """
        state, image = self._engine.update_and_render(action, dt)
        self._last_state = state
        return state, image

    def get_resolution(self) -> tuple[int, int, int]:
        """Get the render resolution.

        Returns:
            tuple[int, int, int]: (height, width, channels).
        """
        return self._engine.get_resolution()

    def setup_scene(self, scene_config: dict[str, Any]) -> None:
        """Set up the scene with given configuration.

        Args:
            scene_config: Dictionary containing scene setup parameters.
        """
        self._engine.setup_scene(scene_config)

    def get_ball_2d_position(self) -> tuple[int, int] | None:
        """Get the 2D pixel coordinates of the ball in the current rendered image."""
        if hasattr(self._engine, "get_ball_2d_position"):
            return self._engine.get_ball_2d_position()
        return None

    def get_reachable_bounds(self) -> tuple[float, float, float, float]:
        """Get the logical physical bounds where the ball can reach.

        Returns:
            tuple[float, float, float, float]: (min_x, min_y, max_x, max_y).
        """
        return self._engine.get_reachable_bounds()


    def project_position_to_2d(self, position_3d: tuple[float, float, float]) -> tuple[int, int] | None:
        """Project a 3D physical position to 2D image coordinates.

        Args:
            position_3d: A tuple consisting of (x, y, z) in the physics world.

        Returns:
            tuple[int, int] | None: The (x, y) pixel coordinates, or None if outside view.
        """
        return self._engine.project_position_to_2d(position_3d)

    def close(self) -> None:
        """Clean up all engine resources."""
        self._engine.close()
