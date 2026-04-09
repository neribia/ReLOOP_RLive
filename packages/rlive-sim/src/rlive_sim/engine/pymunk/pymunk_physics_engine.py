"""PyMunk Physics Engine stub implementation.

This module provides a stub implementation of `BasePhysicsEngine` using PyMunk,
a 2D physics library built on top of Chipmunk.

Alternative physics engines:
    - MuJoCo: For robotics and contact-rich simulation
    - PyBullet: For 3D physics simulation
    - Box2D: Another 2D physics option
    - Brax: JAX-based physics for GPU acceleration

Note:
    This is a stub implementation. Install pymunk with:
    `pip install pymunk` or `uv add pymunk`
"""

from typing import Any

import numpy as np

from rlive_sim.engine.core.base_physics_engine import BasePhysicsEngine, PhysicsState


class PyMunkPhysicsEngine(BasePhysicsEngine):
    """PyMunk-based 2D physics engine.

    This is a stub implementation that demonstrates the interface.
    Override methods with actual PyMunk calls for real physics simulation.

    PyMunk is ideal for:
        - 2D physics simulations
        - Simple robotics scenarios
        - Rapid prototyping
        - Educational purposes

    For 3D or more complex simulations, consider MuJoCo or PyBullet.

    Attributes:
        dt: Simulation timestep in seconds.
        gravity: Gravity vector (only x, y used for 2D).
        space: PyMunk space object (when implemented).

    Example:
        ... engine = PyMunkPhysicsEngine(dt=0.01, gravity=[0, -9.81, 0])
        ... state = engine.reset()
        ... engine.apply_action([10.0, 0.0])  # Apply force
        ... state = engine.step()
    """

    def __init__(
        self,
        dt: float = 0.01,
        gravity: list[float] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the PyMunk physics engine.

        Args:
            dt: Simulation timestep in seconds. Defaults to 0.01.
            gravity: Gravity vector [gx, gy, gz]. Only x, y used.
                Defaults to [0, -9.81, 0] for 2D.
            **kwargs: Additional PyMunk-specific configuration.
        """
        if gravity is None:
            gravity = [0.0, -9.81, 0.0]  # 2D gravity (y-down)
        super().__init__(dt=dt, gravity=gravity)

        # TODO: Initialize PyMunk space
        # import pymunk
        # self.space = pymunk.Space()
        # self.space.gravity = (gravity[0], gravity[1])
        self._space = None
        self._bodies: dict[str, Any] = {}

    def step(self, dt: float | None = None) -> PhysicsState:
        """Advance the physics simulation by one timestep.

        Args:
            dt: Optional timestep override. Uses self.dt if None.

        Returns:
            PhysicsState: The updated physics state after the step.

        Raises:
            NotImplementedError: This is a stub implementation.
        """
        raise NotImplementedError(
            "PyMunkPhysicsEngine.step() is not implemented. "
            "This is a stub. Implement with actual PyMunk calls or "
            "use an alternative like MuJoCo: `from mujoco import MjModel`"
        )

    def reset(self, initial_state: PhysicsState | None = None) -> PhysicsState:
        """Reset the physics simulation to initial conditions.

        Args:
            initial_state: Optional initial state. Uses default if None.

        Returns:
            PhysicsState: The initial physics state after reset.

        Raises:
            NotImplementedError: This is a stub implementation.
        """
        raise NotImplementedError(
            "PyMunkPhysicsEngine.reset() is not implemented. "
            "This is a stub. Implement with actual PyMunk calls or "
            "use an alternative like PyBullet: `import pybullet`"
        )

    def get_state(self) -> PhysicsState:
        """Get the current physics state.

        Returns:
            PhysicsState: The current state of the simulation.

        Raises:
            NotImplementedError: This is a stub implementation.
        """
        raise NotImplementedError(
            "PyMunkPhysicsEngine.get_state() is not implemented. "
            "This is a stub."
        )

    def apply_action(self, action: np.ndarray | list[float]) -> None:
        """Apply a force/impulse to bodies in the simulation.

        For PyMunk, actions are typically 2D forces [fx, fy].

        Args:
            action: Force vector [fx, fy] or [fx, fy, torque].

        Raises:
            NotImplementedError: This is a stub implementation.
        """
        raise NotImplementedError(
            "PyMunkPhysicsEngine.apply_action() is not implemented. "
            "This is a stub."
        )

    def set_state(self, state: PhysicsState) -> None:
        """Set the physics state directly.

        Args:
            state: The state to set.

        Raises:
            NotImplementedError: This is a stub implementation.
        """
        raise NotImplementedError(
            "PyMunkPhysicsEngine.set_state() is not implemented. "
            "This is a stub."
        )

    def close(self) -> None:
        """Clean up PyMunk resources."""
        self._space = None
        self._bodies.clear()
