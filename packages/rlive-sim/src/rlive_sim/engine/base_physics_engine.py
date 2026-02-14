"""Base Physics Engine module.

This module provides the abstract base class for physics engines and the
PhysicsState data model for representing simulation state.

Supported physics backends (via subclasses):
    - PyMunk: 2D physics engine
    - MuJoCo: Advanced robotics simulation
    - PyBullet: 3D physics simulation
    - Box2D: 2D physics engine
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Type

import numpy as np
from pydantic import BaseModel, ConfigDict


class PhysicsState(BaseModel):
    """Represents the physical state of a simulated object.

    This Pydantic model provides type-safe state representation with
    serialization support. For alternative representations, consider:
        - dict: For simple key-value access
        - np.ndarray: For vectorized operations and ML pipelines

    Attributes:
        position: 3D position vector [x, y, z] in world coordinates.
        velocity: 3D velocity vector [vx, vy, vz] in world coordinates.
        rotation: Rotation as quaternion [w, x, y, z] or Euler angles [roll, pitch, yaw].
        angular_velocity: 3D angular velocity vector [wx, wy, wz].
        extra: Additional custom state data (e.g., joint angles, contact forces).

    Examples:
        Creating a physics state with position and velocity:

            state = PhysicsState(
                position=[1.0, 2.0, 0.0],
                velocity=[0.5, 0.0, 0.0],
                rotation=[1.0, 0.0, 0.0, 0.0],
                angular_velocity=[0.0, 0.0, 0.1]
            )

        Converting state to numpy array for ML pipelines:

            state = PhysicsState(position=[1.0, 2.0, 0.0])
            arr = state.to_array()  # Returns shape (13,)
            restored = PhysicsState.from_array(arr)
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    position: list[float] = [0.0, 0.0, 0.0]
    velocity: list[float] = [0.0, 0.0, 0.0]
    rotation: list[float] = [1.0, 0.0, 0.0, 0.0]  # Quaternion [w, x, y, z]
    angular_velocity: list[float] = [0.0, 0.0, 0.0]
    extra: dict[str, Any] = {}

    # FIXME: Maybe not needed
    def to_array(self) -> np.ndarray:
        """Convert state to a flat numpy array.

        Flattens position, velocity, rotation (quaternion), and angular_velocity
        into a single 1D array for use with machine learning models.

        Returns:
            np.ndarray: Flattened state vector with shape (13,) containing
                [position (3), velocity (3), rotation (4), angular_velocity (3)].

        Examples:
            Converting state to array for neural network input:

                state = PhysicsState(
                    position=[1.0, 2.0, 0.0],
                    velocity=[0.5, 0.0, 0.0],
                    rotation=[1.0, 0.0, 0.0, 0.0],
                    angular_velocity=[0.0, 0.0, 0.1]
                )
                arr = state.to_array()
                # arr.shape == (13,)
                # arr == [1.0, 2.0, 0.0, 0.5, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1]
        """
        return np.concatenate([
            self.position,
            self.velocity,
            self.rotation,
            self.angular_velocity,
        ])

    # FIXME: Maybe not needed
    @classmethod
    def from_array(cls, arr: np.ndarray) -> "PhysicsState":
        """Create PhysicsState from a flat numpy array.

        Reconstructs a PhysicsState object from a flattened vector, typically
        used after neural network predictions or state serialization.

        Attributes:
            arr: Flattened state vector of shape (13,) with elements
                [position (3), velocity (3), rotation (4), angular_velocity (3)].

        Returns:
            PhysicsState: Reconstructed state object with all components.

        Raises:
            ValueError: If array length is not 13.

        Examples:
            Reconstructing state from neural network output:

                output = model.predict(input_state)  # Shape (13,)
                state = PhysicsState.from_array(output)
                print(state.position)  # [x, y, z]
                print(state.velocity)  # [vx, vy, vz]

            Round-trip conversion (state -> array -> state):

                original = PhysicsState(position=[1.0, 2.0, 3.0])
                arr = original.to_array()
                restored = PhysicsState.from_array(arr)
                assert restored.position == original.position
        """
        if len(arr) != 13:
            raise ValueError(f"Expected array of length 13, got {len(arr)}")
        return cls(
            position=arr[0:3].tolist(),
            velocity=arr[3:6].tolist(),
            rotation=arr[6:10].tolist(),
            angular_velocity=arr[10:13].tolist(),
        )


class BasePhysicsEngine(ABC):
    """Abstract base class for physics engines.

    This class defines the interface that all physics engine implementations
    must follow. Subclasses should implement the abstract methods to integrate
    specific physics backends like PyMunk, MuJoCo, or PyBullet.

    Attributes:
        dt: Simulation timestep in seconds. # TODO: Change from time to time_steps
        gravity: Gravity vector [gx, gy, gz].

    Examples:
        Basic usage pattern with SimulationEngine:

            engine = MyPhysicsEngine(dt=0.01, gravity=[0, 0, -9.81])
            state = engine.reset()
            engine.apply_action([1.0, 0.0])  # Apply action
            new_state = engine.step()  # Advance simulation
            current = engine.get_state()  # Get current state
    """

    def __init__(self, dt: float = 0.01, gravity: list[float] | None = None) -> None:
        """Initialize the physics engine.

        Attributes:
            dt: Simulation timestep in seconds. Defaults to 0.01 (100 Hz).
            gravity: Gravity vector [gx, gy, gz]. Defaults to [0, 0, -9.81].
        """
        self.dt = dt
        self.gravity = gravity if gravity is not None else [0.0, 0.0, -9.81]
        self._state: PhysicsState = PhysicsState()

    # TODO: Is step a good choise as a name?
    @abstractmethod
    def step(self, dt: float | None = None) -> PhysicsState:
        """Advance the physics simulation by one timestep.

        This method updates all objects in the simulation according to applied
        actions, forces, and constraints. Call apply_action() before step()
        to provide control inputs.

        Attributes:
            dt: Optional timestep override in seconds. Uses self.dt if None.

        Returns:
            PhysicsState: The updated physics state after the simulation step.

        Examples:
            Basic simulation loop with action and step:

                engine = MyPhysicsEngine()
                state = engine.reset()

                for i in range(100):
                    action = [1.0, 0.0]  # Some control input
                    engine.apply_action(action)
                    state = engine.step()  # Advances by self.dt
                    print(f"Step {i}: position={state.position}")

            Using custom timestep:

                state = engine.step(dt=0.02)  # Override to 20ms
        """
        pass

    @abstractmethod
    def reset(self, initial_state: PhysicsState | None = None) -> PhysicsState:
        """Reset the physics simulation to initial conditions.

        Clears all velocities, restores objects to default positions, and
        prepares the simulation for a new episode. Call this at the start of
        each episode or when restarting simulation.

        Attributes:
            initial_state: Optional initial state. If None, uses engine's default.

        Returns:
            PhysicsState: The initial physics state after reset.

        Examples:
            Starting a new episode:

                engine = MyPhysicsEngine()
                state = engine.reset()  # Use default initial state

            Starting with custom initial state:

                initial = PhysicsState(
                    position=[0.0, 1.0, 0.0],
                    velocity=[0.0, 0.0, 0.0]
                )
                state = engine.reset(initial_state=initial)
        """
        pass

    @abstractmethod
    def get_state(self) -> PhysicsState:
        """Get the current physics state.

        Returns a snapshot of the simulation state without modifying it.
        Useful for observation collection and debugging.

        Returns:
            PhysicsState: The current state of the simulation.

        Examples:
            Checking object position and velocity:

                state = engine.get_state()
                print(f"Position: {state.position}")  # [x, y, z]
                print(f"Velocity: {state.velocity}")  # [vx, vy, vz]
        """
        pass

    @abstractmethod
    def apply_action(self, action: np.ndarray | list[float]) -> None:
        """Apply an action (force, torque, velocity) to the simulation.

        This registers the action to be applied during the next step() call.
        The action format and interpretation depend on the specific engine.

        Attributes:
            action: Action vector to apply. Format depends on implementation.
                For example, [angle_degrees, distance_pixels] for ball-in-box,
                or [force_x, force_y, force_z] for 3D physics.

        Examples:
            Applying movement actions:

                # Ball-in-box: angle and distance
                engine.apply_action([45.0, 50.0])
                state = engine.step()

                # Multi-agent: apply multiple actions before step
                engine.apply_action([1.0, 0.0])  # Agent 1
                engine.apply_action([0.0, 1.0])  # Agent 2
                state = engine.step()  # Both applied simultaneously
        """
        pass

    @abstractmethod
    def set_state(self, state: PhysicsState) -> None:
        """Set the physics state directly.

        Allows arbitrary state injection for techniques like state cloning,
        rollback, or deterministic replay. All state components (position,
        velocity, rotation, etc.) are updated.

        Attributes:
            state: The state to set.

        Examples:
            State backup and restore:

                original_state = engine.get_state()
                # ... run simulation ...
                engine.set_state(original_state)  # Restore previous state

            Initializing with specific state:

                target_state = PhysicsState(
                    position=[1.0, 2.0, 0.0],
                    velocity=[0.0, 0.0, 0.0]
                )
                engine.set_state(target_state)
        """
        pass

    def close(self) -> None:
        """Clean up resources. Override in subclasses if needed."""
        pass
