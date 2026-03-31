"""Core abstraction interfaces for all simulation engines."""

# Registries & Decorators
from rlive_sim.engine.core.registry import (
    register_physics_backend,
    register_render_backend,
    register_integrated_backend,
)

# Base classes
from rlive_sim.engine.core.base_physics_engine import BasePhysicsEngine, PhysicsState
from rlive_sim.engine.core.base_render_engine import BaseRenderEngine, SceneState
from rlive_sim.engine.core.base_integrated_engine import BaseIntegratedEngine

# Decorator/Adapter
from rlive_sim.engine.core.combined_engine import CombinedEngine

# Factories
from rlive_sim.engine.core.factories import (
    PhysicsEngine,
    RenderEngine,
    IntegratedEngine,
    SimulationEngineFactory,
)

__all__ = [
    # Registries
    "register_physics_backend",
    "register_render_backend",
    "register_integrated_backend",
    # Base Classes
    "BasePhysicsEngine",
    "PhysicsState",
    "BaseRenderEngine",
    "SceneState",
    "BaseIntegratedEngine",
    # Adapter
    "CombinedEngine",
    # Factories
    "PhysicsEngine",
    "RenderEngine",
    "IntegratedEngine",
    "SimulationEngineFactory",
]

