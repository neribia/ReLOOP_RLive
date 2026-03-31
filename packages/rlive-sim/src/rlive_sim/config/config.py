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

Resources are located at:
- RESOURCES_DIR: Root resources directory
- RESOURCES_DIR / "mujoco": MuJoCo model files
"""
from pathlib import Path


__all__ = [
    "RESOURCES_DIR",
]


# Resources directory reference
RESOURCES_DIR = Path(__file__).parents[3] / "resources"
"""Root resources directory for rlive-sim package.

Location: packages/rlive-sim/resources/

Contains subdirectories:
- mujoco/: MuJoCo model files (.xml, .urdf)
- [other engine-specific resources]
"""
