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

from rlive_sim.engine.base_physics_engine import BasePhysicsEngine, PhysicsState
from rlive_sim.engine.base_render_engine import BaseRenderEngine
from rlive_sim.engine.base_integrated_engine import BaseIntegratedEngine
from rlive_sim.engine.integrated.combined_engine import CombinedEngine


# TODO: Create a class, that gathers all information crated during a step??


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
        _engine: Internal integrated engine (either passed directly or created
            by wrapping separate engines).

    Examples:
        Using separate engines:

            physics = SimplePhysicsEngine(box_width=640, box_height=480)
            renderer = OpenCVRenderEngine(width=640, height=480)
            sim = SimulationEngine(physics_engine=physics, render_engine=renderer)

            state = sim.reset()
            sim.apply_action([45.0, 50.0])
            state = sim.step()
            image = sim.render()

        Using an integrated engine:

            from rlive_sim.config import IntegratedConfig, IntegratedBackend
            from rlive_sim.engine import IntegratedEngine

            config = IntegratedConfig(backend=IntegratedBackend.MUJOCO)
            integrated = IntegratedEngine(config)
            sim = SimulationEngine(integrated_engine=integrated)

            state = sim.reset()
            sim.apply_action([1.0, 0.0])
            state, image = sim.step_and_render()  # Efficient combined call
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

        # Wrap separate engines in CombinedEngine if needed
        if integrated_engine is not None:
            self._engine = integrated_engine
        else:
            self._engine = CombinedEngine(physics_engine, render_engine)

        # Cache for last state (used for rendering in separate mode)
        self._last_state: PhysicsState | None = None

    # TODO: Different name, because step normaly sets an action and returns an observation
    def step(self, dt: float | None = None) -> PhysicsState:
        """Advance the simulation by one timestep.

        Args:
            dt: Optional timestep override. Uses engine default if None.

        Returns:
            PhysicsState: The updated physics state after the step.
        """
        state = self._engine.step(dt)
        self._last_state = state
        return state

    # TODO: Same as step. Name maybe misleading
    def reset(self, initial_state: PhysicsState | None = None) -> PhysicsState:
        """Reset the simulation to initial conditions.

        Args:
            initial_state: Optional initial state. Uses engine default if None.

        Returns:
            PhysicsState: The initial physics state after reset.
        """
        state = self._engine.reset(initial_state)
        self._last_state = state
        return state

    # TODO: LLM said it's better to first set action and then take step, but is it really?
    def apply_action(self, action: np.ndarray | list[float]) -> None:
        """Apply an action to the simulation.

        Args:
            action: Action vector to apply. Format depends on engine.
        """
        self._engine.apply_action(action)

    def render(self, scene_state: dict[str, Any] | None = None) -> np.ndarray:
        """Render the current scene.

        Args:
            scene_state: Optional scene state for rendering. If None,
                uses the last physics state to build scene state.

        Returns:
            np.ndarray: Rendered image as uint8 array with shape
                (height, width, channels).
        """
        if scene_state is None:
            scene_state = self._build_scene_state()

        return self._engine.render(scene_state)

    
    # TODO: This should be step to match the gym.Env api
    def step_and_render(
        self, dt: float | None = None
    ) -> tuple[PhysicsState, np.ndarray]:
        """Advance physics and render in a single call.

        This is more efficient for unified engines.

        Args:
            dt: Optional timestep override.

        Returns:
            tuple[PhysicsState, np.ndarray]: Updated state and rendered image.
        """
        state, image = self._engine.step_and_render(dt)
        self._last_state = state
        return state, image

    def get_state(self) -> PhysicsState:
        """Get the current physics state.

        Returns:
            PhysicsState: The current state of the simulation.
        """
        return self._engine.get_state()

    def set_state(self, state: PhysicsState) -> None:
        """Set the physics state directly.

        Args:
            state: The state to set.
        """
        self._engine.set_state(state)
        self._last_state = state

    def get_observation(self) -> np.ndarray:
        """Get the current observation (rendered image).

        Convenience method that returns the last rendered image or
        renders a new one if needed.

        Returns:
            np.ndarray: Current observation image.
        """
        return self._engine.get_image()

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

    def _build_scene_state(self) -> dict[str, Any]:
        """Build scene state from the last physics state.

        Returns:
            dict[str, Any]: Scene state dictionary for rendering.
        """
        if self._last_state is None:
            return {}

        return {
            "objects": [
                {
                    "id": "main",
                    "position": self._last_state.position,
                    "rotation": self._last_state.rotation,
                    "velocity": self._last_state.velocity,
                }
            ],
            "physics_state": self._last_state.model_dump(),
        }

    def close(self) -> None:
        """Clean up all engine resources."""
        self._engine.close()
