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
    ├── base_physics_engine.py        # Abstract physics interface
    ├── base_render_engine.py         # Abstract render interface
    ├── base_integrated_engine.py     # Abstract monolithic interface
    ├── registry.py                   # @register_* decorators & registries
    ├── factories.py                  # PhysicsEngine, RenderEngine, IntegratedEngine factories
    ├── simulation_engine.py          # Orchestrator
    ├── physics/                      # Physics implementations
    │   ├── simple_physics_engine.py  # 2D ball-in-box physics
    │   └── ...
    ├── render/                       # Render implementations
    │   ├── opencv_render_engine.py   # OpenCV rendering
    │   └── ...
    └── integrated/                   # Integrated implementations
        ├── combined_engine.py        # Adapter for separate engines
        └── ...

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
from rlive_sim.engine.registry import (
    register_physics_backend,
    register_render_backend,
    register_integrated_backend,
)

# Factories
from rlive_sim.engine.factories import PhysicsEngine, RenderEngine, IntegratedEngine

# Base classes
from rlive_sim.engine.base_physics_engine import BasePhysicsEngine, PhysicsState
from rlive_sim.engine.base_render_engine import BaseRenderEngine, SceneState
from rlive_sim.engine.base_integrated_engine import BaseIntegratedEngine

# Main orchestrator
from rlive_sim.engine.simulation_engine import SimulationEngine

# Physics implementations
# Import to trigger @register_physics_backend decorators
from rlive_sim.engine.physics.simple_physics_engine import SimplePhysicsEngine
from rlive_sim.engine.physics.pymunk_physics_engine import PyMunkPhysicsEngine

# Render implementations
# Import to trigger @register_render_backend decorators
from rlive_sim.engine.render.opencv_render_engine import OpenCVRenderEngine
from rlive_sim.engine.render.mitsuba_render_engine import MitsubaRenderEngine

# Integrated implementations
from rlive_sim.engine.unified.combined_engine import CombinedEngine
from rlive_sim.engine.unified.godot_integrated_engine import GodotIntegratedEngine

__all__ = [
    # Registries & Decorators
    "register_physics_backend",
    "register_render_backend",
    "register_integrated_backend",
    # Factories
    "PhysicsEngine",
    "RenderEngine",
    "IntegratedEngine",
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
    "MitsubaRenderEngine",
    # Integrated implementations
    "CombinedEngine",
    "GodotUnifiedEngine",
]
