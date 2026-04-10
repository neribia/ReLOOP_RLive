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

from rlive_sim.engine.core.base_physics_engine import BasePhysicsEngine, PhysicsState
from rlive_sim.engine.core.base_render_engine import BaseRenderEngine
from rlive_sim.engine.core.base_integrated_engine import BaseIntegratedEngine


class CombinedEngine(BaseIntegratedEngine):
    """Adapter wrapping separate physics and render engines.

    This class adapts independent physics and render engines to the
    BaseIntegratedEngine interface. This provides a uniform interface for
    SimulationEngine to work with both truly integrated engines and
    carefully coordinated separate engines.

    This is NOT a true integrated engine - it's an adapter that sequences
    physics.simulate() then render() calls. For maximum efficiency, use
    dedicated integrated engines when available.

    Attributes:
        physics_engine: The wrapped physics engine instance.
        render_engine: The wrapped render engine instance.
        dt: Simulation timestep in seconds (from physics engine).
        gravity: Gravity vector [gx, gy, gz] (from physics engine).
        width: Image width in pixels (from render engine).
        height: Image height in pixels (from render engine).
        channels: Number of color channels (from render engine).

    Methods:
        reset(initial_state): Reset the physics simulation. Returns (state, image).
        update_and_render(action, dt): Advance physics and render in sequence. Returns (state, image).
        get_resolution(): Get the render resolution as (height, width, channels).
        close(): Clean up resources from both wrapped engines.

    Examples:
        Creating a combined engine from separate implementations:

            physics = SimplePhysicsEngine(box_width=640, box_height=480, dt=0.01)
            renderer = OpenCVRenderEngine(width=640, height=480)
            combined = CombinedEngine(physics, renderer)

            state = combined.reset()
            state, image = combined.simulate_and_render([45.0, 50.0])# angle, distance

            # image is uint8 numpy array ready for display/processing
    """

    def __init__(
        self,
        physics_engine: BasePhysicsEngine,
        render_engine: BaseRenderEngine,
    ) -> None:
        """Initialize the combined engine adapter.

        Args:
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

        # TODO: Does this wrapper need these parameters? Maybe not, since it delegates to the wrapped engines.
        # Extract parameters from wrapped engines
        super().__init__(
            gravity=physics_engine.gravity,
            width=render_engine.width,
            height=render_engine.height,
            channels=render_engine.channels,
        )

        self.physics_engine = physics_engine
        self.render_engine = render_engine

    def reset(self, initial_state: PhysicsState | None = None) -> tuple[PhysicsState, np.ndarray]:
        """Reset the physics simulation to initial conditions.

        Delegates to the wrapped physics engine, then renders initial state.

        Args:
            initial_state: Optional initial state. Uses engine default if None.

        Returns:
            tuple[PhysicsState, np.ndarray]: Tuple of (initial_state, rendered_image).
        """
        state = self.physics_engine.reset(initial_state)
        dynamic_objects = self.physics_engine.get_scene_objects()
        image = self.render_engine.render(dynamic_objects)
        return state, image

    # Combined methods
    def update_and_render(self, action: np.ndarray | list[float], dt: float | None = None) -> tuple[PhysicsState, np.ndarray]:
        """Advance physics and render in sequence.

        For separate engines, this calls update() then render() sequentially.
        The render uses the physics state after the simulation step.

        Args:
            action: Action vector to apply.
            dt: Optional timestep override.

        Returns:
            tuple[PhysicsState, np.ndarray]: Updated state and rendered image.

        Examples:
            Update loop:

                engine = CombinedEngine(physics_engine, render_engine)

                # Inside the simulation loop:
                action = [1.0, 0.0]
                state, image = engine.update_and_render(action)
        """
        state = self.physics_engine.update(action, dt)
        dynamic_objects = self.physics_engine.get_scene_objects()
        image = self.render_engine.render(dynamic_objects)
        return state, image

    def get_resolution(self) -> tuple[int, int, int]:
        """Get the render resolution.

        Returns:
            tuple[int, int, int]: (height, width, channels).
        """
        return self.render_engine.get_resolution()

    def setup_scene(self, scene_config: dict[str, Any]) -> None:
        """Set up the scene using a configuration dictionary.

        Initializes the simulation with a scene configuration. The exact
        schema of ``scene_config`` depends on the underlying physics and
        render engines wrapped by this adapter.

        Call before reset() to set up the initial scene configuration.

        Args:
            scene_config: Dictionary describing the scene to initialize.
                This may include objects, their properties, environment
                parameters, camera settings, etc.

        Raises:
            NotImplementedError: This adapter does not implement scene setup
                by configuration. Subclasses should override this method if
                scene configuration is supported.

        Examples:
            Providing a basic scene configuration:

                engine = CombinedEngine(physics_engine, render_engine)
                scene_config = {
                    "box_width": 0.8,
                    "box_height": 0.6,
                }
                engine.setup_scene(scene_config)
        """
        self.physics_engine.setup_scene(scene_config)

        # Share static geometry discovered by physics engine to renderer
        # Pass a derived config to avoid mutating caller's dict
        derived_config = scene_config.copy()
        static_objects = self.physics_engine.get_static_scene_objects()
        if static_objects:
            # Append rather than overwrite if static objects are already provided
            existing_static = derived_config.get("static_objects", [])
            derived_config["static_objects"] = existing_static + static_objects

        self.render_engine.setup_scene(derived_config)

    def get_ball_2d_position(self) -> tuple[int, int] | None:
        """Get ball 2D position by asking the render engine.

        Returns:
            tuple[int, int] | None: The 2D pixel coordinates or None.
        """
        # For combined engine, get 3D state and project it via render engine
        state = self.physics_engine.get_state()
        pos_3d = state.position  # e.g. [x, y] or [x, y, z]
        return self.project_position_to_2d(tuple(pos_3d))

    def get_reachable_bounds(self) -> tuple[float, float, float, float]:
        """Get the logical physical bounds where the ball can reach.

        Delegates directly to the physics engine.
        """
        return self.physics_engine.get_bounds()

    def project_position_to_2d(self, position_3d: tuple[float, float, float]) -> tuple[int, int] | None:
        """Project a 3D physical position to 2D image coordinates.

        Delegates to the render engine.
        """
        return self.render_engine.project_to_2d(list(position_3d))


    def close(self) -> None:
        """Clean up resources from both wrapped engines.

        Closes both the physics and render engines in order.
        """
        if self.physics_engine:
            self.physics_engine.close()
        if self.render_engine:
            self.render_engine.close()
