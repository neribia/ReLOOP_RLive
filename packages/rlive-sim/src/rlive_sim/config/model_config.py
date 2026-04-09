"""Configuration module for rlive-sim.

This module provides Pydantic configuration models for:
- Physics engines (PhysicsConfig)
- Render engines (RenderConfig)
- Integrated engines (IntegratedConfig)
- Overall simulation setup (SimulationConfig)

Configuration values can be set via:
- Direct constructor injection
- Environment variables (prefix: RLIVE_SIM_)
- Config files (YAML/JSON via Pydantic)
"""

from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class PhysicsBackend(str, Enum):
    """Available physics engine backends."""

    SIMPLE = "simple"
    # PYMUNK = "pymunk"
    # MUJOCO = "mujoco"
    # PYBULLET = "pybullet"
    # BOX2D = "box2d"


class RenderBackend(str, Enum):
    """Available render engine backends."""

    OPENCV = "opencv"
    # MUJOCO = "mujoco"
    # MITSUBA = "mitsuba"
    # BLENDER = "blender"
    # OPENGL = "opengl"
    # PANDA3D = "panda3d"
    # PYRENDER = "pyrender"


class IntegratedBackend(str, Enum):
    """Available integrated engine backends.

    Integrated engines combine physics and rendering in a single monolithic system.
    """

    # GODOT = "godot"
    # UNITY = "unity"
    # MUJOCO = "mujoco"  # MuJoCo with built-in rendering
    SAPIEN = "sapien"
    # ISAAC_SIM = "isaac_sim"
    # PYBULLET = "pybullet"  # PyBullet with OpenGL rendering


class PhysicsConfig(BaseModel):
    """Configuration for physics engines.

    Attributes:
        backend: The physics backend to use.
        dt: Simulation timestep in seconds.
        gravity: Gravity vector [gx, gy, gz].
        substeps: Number of physics substeps per step.
        extra: Additional backend-specific configuration.

    Example:
        config = PhysicsConfig(backend=PhysicsBackend.SIMPLE, dt=0.01)
        config.gravity
        [0.0, 0.0, -9.81]
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    backend: PhysicsBackend = PhysicsBackend.SIMPLE
    dt: float = Field(default=0.01, gt=0, description="Timestep in seconds")
    gravity: list[float] = Field(default=[0.0, 0.0, -9.81], min_length=3, max_length=3)
    substeps: int = Field(default=1, ge=1, description="Physics substeps per step")
    extra: dict[str, Any] = Field(default_factory=dict)


class RenderConfig(BaseModel):
    """Configuration for render engines.

    Attributes:
        backend: The render backend to use.
        width: Image width in pixels.
        height: Image height in pixels.
        channels: Number of color channels (RGB=3, RGBA=4).
        spp: Samples per pixel (for ray tracing renderers).
        extra: Additional backend-specific configuration.

    Example:
        config = RenderConfig(backend=RenderBackend.OPENCV, width=640, height=480)
        config.get_resolution()
        (480, 640, 3)
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    backend: RenderBackend = RenderBackend.OPENCV
    width: int = Field(default=640, gt=0)
    height: int = Field(default=480, gt=0)
    channels: int = Field(default=3, ge=1, le=4)
    spp: int = Field(default=64, ge=1, description="Samples per pixel")
    extra: dict[str, Any] = Field(default_factory=dict)

    def get_resolution(self) -> tuple[int, int, int]:
        """Get resolution as (height, width, channels) tuple."""
        return (self.height, self.width, self.channels)


class IntegratedConfig(BaseModel):
    """Configuration for integrated engines.

    Integrated engines combine physics and rendering in a single monolithic system
    that can efficiently share resources and execute together.

    Attributes:
        backend: The integrated backend to use.
        gravity: Gravity vector [gx, gy, gz].
        width: Image width in pixels.
        height: Image height in pixels.
        channels: Number of color channels (RGB=3, RGBA=4).
        extra: Engine-specific configuration passed as kwargs to the engine.

    Example (MuJoCo backend with custom parameters):
        config = IntegratedConfig(
            backend=IntegratedBackend.MUJOCO,
            width=640,
            height=480,
            extra={
                "max_speed_ms": 0.8,
                "acceleration_time": 0.5,
                "deceleration_time": 0.3,
                "controller_kp": 0.15,
                "controller_kd": 0.05,
            }
        )
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    backend: IntegratedBackend | None = None
    gravity: list[float] = Field(default=[0.0, 0.0, -9.81], min_length=3, max_length=3)
    width: int = Field(default=640, gt=0)
    height: int = Field(default=480, gt=0)
    channels: int = Field(default=3, ge=1, le=4)
    extra: dict[str, Any] = Field(default_factory=dict)


class MujocoConfig(BaseModel):
    """DEPRECATED: Use IntegratedConfig.extra dict directly instead.

    This class is kept for backward compatibility only.
    Pass MuJoCo parameters directly via config.extra:

    Example:
        config = IntegratedConfig(
            backend=IntegratedBackend.MUJOCO,
            extra={
                "scene_path": "/path/to/scene.xml",
                "max_speed_ms": 0.5,
                "acceleration_time": 0.3,
                "deceleration_time": 0.2,
                "controller_kp": 0.1,
                "controller_kd": 0.1,
            }
        )
    """
    pass


class SimulationConfig(BaseModel):
    """Main simulation configuration.

    This is the top-level configuration that determines whether to use
    an integrated engine or separate physics/render engines.

    The configuration can be loaded from:
    - Environment variables with prefix `RLIVE_SIM_`
    - Config files (YAML/JSON)
    - Direct Python instantiation

    Attributes:
        use_integrated: Whether to use an integrated engine (monolithic).
            If False, uses separate physics and render engines.
        physics: Physics engine configuration (if not using integrated).
        render: Render engine configuration (if not using integrated).
        integrated: Integrated engine configuration (if using integrated).
        max_episode_steps: Maximum steps per episode.
        seed: Random seed for reproducibility.

    Example:
        Using environment variables:
        # Set RLIVE_SIM_USE_INTEGRATED=true
        config = SimulationConfig()
        config.use_integrated
        True

        Using separate engines:
        config = SimulationConfig(
            use_integrated=False,
            physics=PhysicsConfig(backend=PhysicsBackend.SIMPLE),
            render=RenderConfig(backend=RenderBackend.OPENCV),
        )

        Using integrated engine:
        config = SimulationConfig(
            use_integrated=True,
            integrated=IntegratedConfig(backend=IntegratedBackend.MUJOCO),
        )
    """

    model_config = ConfigDict(
        env_prefix="RLIVE_SIM_",
        extra="ignore",
    )

    use_integrated: bool = Field(
        default=False,
        description="Use integrated engine instead of separate physics/render",
    )
    physics: PhysicsConfig = Field(default_factory=PhysicsConfig)
    render: RenderConfig = Field(default_factory=RenderConfig)
    integrated: IntegratedConfig = Field(default_factory=IntegratedConfig)
    max_episode_steps: int = Field(default=100, ge=1, description="Maximum steps per episode")
    seed: int | None = Field(default=None, description="Random seed")

    def get_observation_shape(self) -> tuple[int, int, int]:
        """Get the observation shape based on render configuration.

        Returns:
            tuple[int, int, int]: (height, width, channels).
        """
        if self.use_integrated:
            return (self.integrated.height, self.integrated.width, self.integrated.channels)
        else:
            return self.render.get_resolution()


# Global default configuration instance
config = SimulationConfig()
