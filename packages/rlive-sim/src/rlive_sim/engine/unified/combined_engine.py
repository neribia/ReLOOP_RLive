"""Combined Engine module.

This module provides an adapter that wraps separate physics and render
engines, exposing them through the BaseIntegratedEngine interface for
compatibility with SimulationEngine.

CombinedEngine is not itself "integrated" (it doesn't combine physics+render
into one system), but it adapts separate engines to the integrated interface.
This allows SimulationEngine to work uniformly with both:
    - True integrated engines (Godot, MuJoCo, Isaac Sim)
    - Adapted combinations of separate engines (PyMunk + OpenCV)
"""

from typing import Any

import numpy as np

from rlive_sim.engine.base_physics_engine import BasePhysicsEngine, PhysicsState
from rlive_sim.engine.base_render_engine import BaseRenderEngine
from rlive_sim.engine.base_integrated_engine import BaseIntegratedEngine


class CombinedEngine(BaseIntegratedEngine):
    """Adapter wrapping separate physics and render engines.

    This class adapts independent physics and render engines to the
    BaseIntegratedEngine interface. This provides a uniform interface for
    SimulationEngine to work with both truly integrated engines and
    carefully coordinated separate engines.

    This is NOT a true integrated engine - it's an adapter that sequences
    physics.step() then render() calls. For maximum efficiency, use
    dedicated integrated engines when available.

    Attributes:
        physics_engine: The wrapped physics engine instance.
        render_engine: The wrapped render engine instance.
        dt: Simulation timestep in seconds (from physics engine).
        gravity: Gravity vector [gx, gy, gz] (from physics engine).
        width: Image width in pixels (from render engine).
        height: Image height in pixels (from render engine).
        channels: Number of color channels (from render engine).

    Examples:
        Creating a combined engine from separate implementations:

            physics = SimplePhysicsEngine(box_width=640, box_height=480, dt=0.01)
            renderer = OpenCVRenderEngine(width=640, height=480)
            combined = CombinedEngine(physics, renderer)

            state = combined.reset()
            combined.apply_action([45.0, 50.0])  # angle, distance
            state, image = combined.step_and_render()

            # image is uint8 numpy array ready for display/processing
    """

    def __init__(
        self,
        physics_engine: BasePhysicsEngine,
        render_engine: BaseRenderEngine,
    ) -> None:
        """Initialize the combined engine adapter.

        Attributes:
            physics_engine: Physics engine instance to wrap.
            render_engine: Render engine instance to wrap.

        Raises:
            TypeError: If physics_engine is not a BasePhysicsEngine or
                render_engine is not a BaseRenderEngine.
        """
        if not isinstance(physics_engine, BasePhysicsEngine):
            raise TypeError(
                f"physics_engine must be BasePhysicsEngine, "
                f"got {type(physics_engine).__name__}"
            )
        if not isinstance(render_engine, BaseRenderEngine):
            raise TypeError(
                f"render_engine must be BaseRenderEngine, "
                f"got {type(render_engine).__name__}"
            )

        # Extract parameters from wrapped engines
        super().__init__(
            dt=physics_engine.dt,
            gravity=physics_engine.gravity,
            width=render_engine.width,
            height=render_engine.height,
            channels=render_engine.channels,
        )

        self.physics_engine = physics_engine
        self.render_engine = render_engine

    # Physics engine delegation methods
    def step(self, dt: float | None = None) -> PhysicsState:
        """Advance physics by one timestep.

        Delegates to the wrapped physics engine.

        Attributes:
            dt: Optional timestep override. Uses self.dt if None.

        Returns:
            PhysicsState: The updated physics state.
        """
        return self.physics_engine.step(dt)

    def reset(self, initial_state: PhysicsState | None = None) -> PhysicsState:
        """Reset the physics simulation to initial conditions.

        Delegates to the wrapped physics engine.

        Attributes:
            initial_state: Optional initial state. Uses engine default if None.

        Returns:
            PhysicsState: The initial physics state after reset.
        """
        return self.physics_engine.reset(initial_state)

    def get_state(self) -> PhysicsState:
        """Get the current physics state.

        Delegates to the wrapped physics engine.

        Returns:
            PhysicsState: The current simulation state.
        """
        return self.physics_engine.get_state()

    def set_state(self, state: PhysicsState) -> None:
        """Set the physics state directly.

        Delegates to the wrapped physics engine.

        Attributes:
            state: The state to set.
        """
        self.physics_engine.set_state(state)

    def apply_action(self, action: np.ndarray | list[float]) -> None:
        """Apply an action to the physics simulation.

        Delegates to the wrapped physics engine.

        Attributes:
            action: Action vector to apply.
        """
        self.physics_engine.apply_action(action)

    # Render engine delegation methods
    def render(self, scene_state: dict[str, Any]) -> np.ndarray:
        """Render the current scene.

        Delegates to the wrapped render engine.

        Attributes:
            scene_state: Dictionary containing scene information.

        Returns:
            np.ndarray: Rendered image as uint8 array with shape
                (height, width, channels).
        """
        return self.render_engine.render(scene_state)

    def get_image(self) -> np.ndarray:
        """Get the last rendered image without re-rendering.

        Delegates to the wrapped render engine.

        Returns:
            np.ndarray: Last rendered image.
        """
        return self.render_engine.get_image()

    def setup_scene(self, scene_config: dict[str, Any]) -> None:
        """Set up the scene with given configuration.

        Delegates to the wrapped render engine.

        Attributes:
            scene_config: Dictionary containing scene setup parameters.
        """
        self.render_engine.setup_scene(scene_config)

    def set_camera(
        self,
        position: list[float],
        target: list[float],
        up: list[float] | None = None,
    ) -> None:
        """Set the camera position and orientation.

        Delegates to the wrapped render engine.

        Attributes:
            position: Camera position [x, y, z] in world coordinates.
            target: Point the camera looks at [x, y, z].
            up: Up vector [x, y, z]. Defaults to [0, 0, 1] if None.
        """
        self.render_engine.set_camera(position, target, up)

    # Combined methods
    def step_and_render(self, dt: float | None = None) -> tuple[PhysicsState, np.ndarray]:
        """Advance physics and render in sequence.

        For separate engines, this calls step() then render() sequentially.
        The render uses the physics state after the step.

        Attributes:
            dt: Optional timestep override.

        Returns:
            tuple[PhysicsState, np.ndarray]: Updated state and rendered image.
        """
        state = self.step(dt)
        scene_state = self._build_scene_state(state)
        image = self.render(scene_state)
        return state, image

    # Unified engine abstract methods (not implemented for separate engines)
    def load_scene(self, scene_path: str) -> None:
        """Load a scene from file.

        Not implemented for separate engines. Use setup_scene() instead
        with a configuration dictionary.

        Raises:
            NotImplementedError: Separate engines don't support scene files.
        """
        raise NotImplementedError(
            "CombinedEngine does not support load_scene(). "
            "Use setup_scene() with a configuration dictionary instead."
        )

    def spawn_object(
        self,
        object_type: str,
        position: list[float],
        rotation: list[float] | None = None,
        **kwargs: Any,
    ) -> str:
        """Spawn an object in the scene.

        Not implemented for separate engines as object spawning is typically
        an engine-specific feature.

        Raises:
            NotImplementedError: Separate engines don't support dynamic spawning.
        """
        raise NotImplementedError(
            "CombinedEngine does not support spawn_object(). "
            "Configure objects statically in setup_scene() instead."
        )

    def remove_object(self, object_id: str) -> None:
        """Remove an object from the scene.

        Not implemented for separate engines.

        Raises:
            NotImplementedError: Separate engines don't support dynamic removal.
        """
        raise NotImplementedError(
            "CombinedEngine does not support remove_object(). "
            "Configure objects statically in setup_scene() instead."
        )

    # Utility method
    def _build_scene_state(self, state: PhysicsState) -> dict[str, Any]:
        """Build scene state from physics state for rendering.

        Internal helper to convert physics state to scene state format
        for the render engine.

        Attributes:
            state: The physics state to convert.

        Returns:
            dict[str, Any]: Scene state dictionary for rendering.
        """
        return {
            "objects": [
                {
                    "id": "main",
                    "position": state.position,
                    "rotation": state.rotation,
                    "velocity": state.velocity,
                }
            ],
            "physics_state": state.model_dump(),
        }

    def close(self) -> None:
        """Clean up resources from both wrapped engines.

        Closes both the physics and render engines in order.
        """
        if self.physics_engine:
            self.physics_engine.close()
        if self.render_engine:
            self.render_engine.close()

