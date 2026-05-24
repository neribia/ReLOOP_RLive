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
from typing import Any

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
        rotation: Euler angles in degrees [roll, pitch, yaw] using ZYX intrinsic
            convention (roll=X, pitch=Y, yaw=Z). Range: roll/pitch ∈ [-180, 180],
            yaw ∈ [-180, 180].
        angular_velocity: 3D angular velocity vector [wx, wy, wz].
        extra: Additional custom state data (e.g., joint angles, contact forces).

    Methods:
        to_array(): Convert state to a flat numpy array for ML models.
        from_array(arr): Create PhysicsState from a flat numpy array.

    Examples:
        Creating a physics state with position and velocity:

            state = PhysicsState(
                position=[1.0, 2.0, 0.0],
                velocity=[0.5, 0.0, 0.0],
                rotation=[0.0, 0.0, 45.0],   # facing 45° (yaw)
                angular_velocity=[0.0, 0.0, 0.1]
            )

        Converting state to numpy array for ML pipelines:

            state = PhysicsState(position=[1.0, 2.0, 0.0])
            arr = state.to_array()  # Returns shape (12,)
            restored = PhysicsState.from_array(arr)
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    position: list[float] = [0.0, 0.0, 0.0]
    velocity: list[float] = [0.0, 0.0, 0.0]
    rotation: list[float] = [0.0, 0.0, 0.0]  # Euler angles in degrees [roll, pitch, yaw]
    angular_velocity: list[float] = [0.0, 0.0, 0.0]
    extra: dict[str, Any] = {}

    # FIXME: Maybe not needed
    def to_array(self) -> np.ndarray:
        """Convert state to a flat numpy array.

        Flattens position, velocity, rotation (Euler degrees), and angular_velocity
        into a single 1D array for use with machine learning models.

        Returns:
            np.ndarray: Flattened state vector with shape (12,) containing
                [position (3), velocity (3), rotation (3), angular_velocity (3)].

        Examples:
            Converting state to array for neural network input:

                state = PhysicsState(
                    position=[1.0, 2.0, 0.0],
                    velocity=[0.5, 0.0, 0.0],
                    rotation=[0.0, 0.0, 45.0],
                    angular_velocity=[0.0, 0.0, 0.1]
                )
                arr = state.to_array()
                # arr.shape == (12,)
                # arr == [1.0, 2.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 45.0, 0.0, 0.0, 0.1]
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

        Args:
            arr: Flattened state vector of shape (12,) with elements
                [position (3), velocity (3), rotation (3), angular_velocity (3)].

        Returns:
            PhysicsState: Reconstructed state object with all components.

        Raises:
            ValueError: If array length is not 12.

        Examples:
            Reconstructing state from neural network output:

                output = model.predict(input_state)  # Shape (12,)
                state = PhysicsState.from_array(output)
                print(state.position)   # [x, y, z]
                print(state.rotation)   # [roll_deg, pitch_deg, yaw_deg]

            Round-trip conversion (state -> array -> state):

                original = PhysicsState(position=[1.0, 2.0, 3.0])
                arr = original.to_array()
                restored = PhysicsState.from_array(arr)
                assert restored.position == original.position
        """
        if len(arr) != 12:
            raise ValueError(f"Expected array of length 12, got {len(arr)}")
        return cls(
            position=arr[0:3].tolist(),
            velocity=arr[3:6].tolist(),
            rotation=arr[6:9].tolist(),
            angular_velocity=arr[9:12].tolist(),
        )


class SceneObject(BaseModel):
    """Represents a geometric object in the scene, decoupled from rendering specifics.

    Attributes:
        id: Unique identifier or type name of the object (e.g., "robot", "eurobox").
        position: 3D position vector [x, y, z] in world coordinates.
        rotation: 3D rotation Euler angles [roll, pitch, yaw] or Quaternion.
        dimensions: Dimensions of the object (e.g., [width, height, depth] or [radius]).
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str
    position: list[float]
    rotation: list[float]
    dimensions: list[float]


class BasePhysicsEngine(ABC):
    """Abstract base class for physics engines.

    This class defines the interface that all physics engine implementations
    must follow. Subclasses should implement the abstract methods to integrate
    specific physics backends like PyMunk, MuJoCo, or PyBullet.

    Attributes:
        dt: Simulation timestep in seconds. # TODO: Change from time to time_steps
        gravity: Gravity vector [gx, gy, gz].

    Methods:
        reset(initial_state): Reset the simulation to initial conditions.
        update(action, dt): Advance the simulation by one timestep (abstract).
        get_state(): Get the current physics state.
        set_state(state): Set the physics state directly.
        close(): Clean up resources.

    Examples:
        Basic usage pattern with SimulationEngine:

            engine = MyPhysicsEngine(dt=0.01, gravity=[0, 0, -9.81])
            state = engine.reset()
            new_state = engine.update([1.0])  # Advance simulation with action
            current = engine.get_state()  # Get current state
    """

    def __init__(self, dt: float = 0.01, gravity: list[float] | None = None) -> None:
        """Initialize the physics engine.

        Args:
            dt: Simulation timestep in seconds. Defaults to 0.01 (100 Hz).
            gravity: Gravity vector [gx, gy, gz]. Defaults to [0, 0, -9.81].
        """
        self.dt = dt
        self.gravity = gravity if gravity is not None else [0.0, 0.0, -9.81]
        self._state: PhysicsState = PhysicsState()

    @abstractmethod
    def reset(self, initial_state: PhysicsState | None = None) -> PhysicsState:
        """Reset the physics simulation to initial conditions.

        Clears all velocities, restores objects to default positions, and
        prepares the simulation for a new episode. Call this at the start of
        each episode or when restarting simulation.

        Args:
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
                    velocity=[0.0, 0.0, 0.0],
                    rotation=[0.0, 0.0, 90.0],  # 90° yaw
                )
                state = engine.reset(initial_state=initial)
        """
        pass

    @abstractmethod
    def update(self, action: np.ndarray | list[float], dt: float | None = None) -> PhysicsState:
        """Advance the physics simulation by one timestep.

        This method updates all objects in the simulation according to applied
        actions, forces, and constraints.

        Args:
            action: Action vector to apply. Format depends on implementation.
                For example, [angle_degrees, distance_pixels] for ball-in-box,
                or [force_x, force_y, force_z] for 3D physics.
            dt: Optional timestep override in seconds. Uses self.dt if None.

        Returns:
            PhysicsState: Updated physics state after the simulation step.

        Examples:
            Updating physics with an action:

                engine = MyPhysicsEngine()
                state = engine.reset()

                # Apply action: [45 degree angle, 50 pixel distance]
                new_state = engine.update([45, 50])

                # With custom timestep
                new_state = engine.update([45, 50], dt=0.005)
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
    def set_state(self, state: PhysicsState) -> None:
        """Set the physics state directly.

        Args:
            state: The new state to apply.
        """
        pass

    @abstractmethod
    def get_bounds(self) -> tuple[float, float, float, float]:
        """Get the valid bounds for ball position.

        Returns:
            tuple[float, float, float, float]: (min_x, min_y, max_x, max_y).
        """
        pass

    def get_scene_objects(self) -> list[SceneObject]:
        pass

    def get_static_scene_objects(self) -> list[SceneObject]:
        """Get the static objects to render in the scene (e.g. background, bounds).

        This should primarily be overridden by engines that generate static objects
        during setup_scene.

        Returns:
            list[SceneObject]: List of static scene objects.
        """
        return []

    @abstractmethod
    def setup_scene(self, scene_config: dict[str, Any]) -> None:
        """Set up the scene configuration.

        Args:
            scene_config: Configuration dictionary for the scene.
        """
        pass

    def close(self) -> None:
        """Clean up resources. Override in subclasses if needed."""
        pass
