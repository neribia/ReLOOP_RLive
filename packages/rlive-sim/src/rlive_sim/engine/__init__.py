"""Engine module exports.

This module provides the simulation engine architecture with support for:
- Separate physics and render engines (independent, flexible)
- Integrated engines (monolithic physics + rendering)

SimulationEngine provides a unified API for both architectures.

Architecture:

1. **Separate Engines**:
   - PhysicsEngine: Pure physics simulation
   - RenderEngine: Pure rendering/observation generation
   - Use for: Flexibility, mix-and-match, independent optimization

2. **Integrated Engines**:
   - BaseIntegratedEngine: Physics + render tightly coupled
   - Use for: Efficiency, shared resources, single-world execution

3. **Orchestrator**:
   - SimulationEngine: Unified interface for both approaches
   - CombinedEngine: Adapter wrapping separate engines for compatibility

Directory Structure:
    engine/
    ├── __init__.py                     # Module exports and engine registration
    ├── simulation_engine.py            # Main Orchestrator
    ├── core/                           # Base abstractions and factories
    │   ├── base_physics_engine.py
    │   ├── base_render_engine.py
    │   ├── base_integrated_engine.py
    │   ├── combined_engine.py
    │   ├── factories.py
    │   └── registry.py
    ├── sapien/                         # SAPIEN integrated physics/render engine
    │   ├── sapien_integrated_engine.py
    │   ├── sphero_controller.py
    │   └── world_loading.py
    ├── pymunk/                         # PyMunk 2D physics engine
    │   └── pymunk_physics_engine.py
    └── basic/                          # Basic/Fallback 2D components
        ├── simple_physics_engine.py
        └── opencv_render_engine.py

Examples:

    Using separate engines:

        from rlive_sim.engine import PhysicsEngine, RenderEngine, SimulationEngine
        from rlive_sim.config import PhysicsConfig, RenderConfig

        physics = PhysicsEngine(PhysicsConfig(...))
        render = RenderEngine(RenderConfig(...))
        sim = SimulationEngine(physics_engine=physics, render_engine=render)

    Using an integrated engine:

        from rlive_sim.engine import IntegratedEngine, SimulationEngine
        from rlive_sim.config import IntegratedConfig

        integrated = IntegratedEngine(IntegratedConfig(...))
        sim = SimulationEngine(integrated_engine=integrated)
"""

# Registries & Decorators
from rlive_sim.engine.core.registry import (
    register_physics_backend,
    register_render_backend,
    register_integrated_backend,
)

# Factories
from rlive_sim.engine.core.factories import (
    PhysicsEngine,
    RenderEngine,
    IntegratedEngine,
    SimulationEngineFactory,
)

# Base classes
from rlive_sim.engine.core.base_physics_engine import BasePhysicsEngine, PhysicsState
from rlive_sim.engine.core.base_render_engine import BaseRenderEngine, SceneState
from rlive_sim.engine.core.base_integrated_engine import BaseIntegratedEngine

# Main orchestrator
from rlive_sim.engine.simulation_engine import SimulationEngine

# Physics implementations
# Import to trigger @register_physics_backend decorators
from rlive_sim.engine.basic.simple_physics_engine import SimplePhysicsEngine
from rlive_sim.engine.pymunk.pymunk_physics_engine import PyMunkPhysicsEngine

# Render implementations
# Import to trigger @register_render_backend decorators
from rlive_sim.engine.basic.opencv_render_engine import OpenCVRenderEngine

# Integrated implementations
from rlive_sim.engine.core.combined_engine import CombinedEngine
from rlive_sim.engine.sapien.sapien_integrated_engine import SapienIntegratedEngine

__all__ = [
    # Registries & Decorators
    "register_physics_backend",
    "register_render_backend",
    "register_integrated_backend",
    # Factories
    "PhysicsEngine",
    "RenderEngine",
    "IntegratedEngine",
    "SimulationEngineFactory",
    # Base classes
    "BasePhysicsEngine",
    "BaseRenderEngine",
    "BaseIntegratedEngine",
    # States
    "PhysicsState",
    "SceneState",
    # Main orchestrator
    "SimulationEngine",
    # Physics implementations
    "SimplePhysicsEngine",
    "PyMunkPhysicsEngine",
    # Render implementations
    "OpenCVRenderEngine",
    # Integrated implementations
    "CombinedEngine",
    "SapienIntegratedEngine"
]
