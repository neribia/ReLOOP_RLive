"""rlive-sim: Simulation package for ReLoop RLive.

This package provides simulation environments with flexible physics and
rendering backends for reinforcement learning.

Example:
    from rlive_sim import SimulationEngine, SimplePhysicsEngine, OpenCVRenderEngine
    engine = SimulationEngine(
        physics_engine=SimplePhysicsEngine(),
        render_engine=OpenCVRenderEngine(),
    )
    from rlive_sim import SimulationEnv
    env = SimulationEnv(engine=engine)
    obs, info = env.reset()
"""

from rlive_sim.simulation_env import SimulationEnv
from rlive_sim.engine import (
    BasePhysicsEngine,
    BaseRenderEngine,
    BaseIntegratedEngine,
    PhysicsState,
    SimulationEngine,
    # Working implementations
    SimplePhysicsEngine,
    OpenCVRenderEngine,
    # Stub implementations
    PyMunkPhysicsEngine,
    MitsubaRenderEngine,
    GodotIntegratedEngine,
)
from rlive_sim.config import (
    PhysicsBackend,
    PhysicsConfig,
    RenderBackend,
    RenderConfig,
    SimulationConfig,
    IntegratedBackend,
    IntegratedConfig,
    config,
)

__all__ = [
    # Environment
    "SimulationEnv",
    # Engine classes
    "BasePhysicsEngine",
    "BaseRenderEngine",
    "BaseIntegratedEngine",
    "PhysicsState",
    "SimulationEngine",
    # Working implementations
    "SimplePhysicsEngine",
    "OpenCVRenderEngine",
    # Stub implementations
    "PyMunkPhysicsEngine",
    "MitsubaRenderEngine",
    "GodotIntegratedEngine",
    # Config
    "PhysicsBackend",
    "PhysicsConfig",
    "RenderBackend",
    "RenderConfig",
    "SimulationConfig",
    "IntegratedBackend",
    "IntegratedConfig",
    "config",
]
