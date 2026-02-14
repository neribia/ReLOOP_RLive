"""
Registry for physics, render, and integrated engines.

This module provides:
- Registries for mapping backends to engine classes
- Decorators for auto-registering engines (@register_*_backend)
- Factory integration for engine creation from config

Registries are populated when engine modules are imported (via decorators).
This allows engines to be added without modifying the registry code.
"""

from typing import Any, Dict, Type

from rlive_sim.config import PhysicsBackend, RenderBackend, IntegratedBackend
from rlive_sim.engine.base_physics_engine import BasePhysicsEngine
from rlive_sim.engine.base_render_engine import BaseRenderEngine
from rlive_sim.engine.base_integrated_engine import BaseIntegratedEngine


# ==========================================================
# Physics Engine Registry
# ==========================================================

_PHYSICS_REGISTRY: dict[Any, Type["BasePhysicsEngine"]] = {}


def register_physics_backend(backend: "PhysicsBackend"):
    """Decorator to register a physics engine for a backend."""
    def decorator(cls: Type["BasePhysicsEngine"]):
        _PHYSICS_REGISTRY[backend] = cls
        return cls
    return decorator


# ==========================================================
# Render Engine Registry
# ==========================================================

_RENDER_REGISTRY: dict[Any, Type["BaseRenderEngine"]] = {}


def register_render_backend(backend: "RenderBackend"):
    """Decorator to register a render engine for a backend."""
    def decorator(cls: Type["BaseRenderEngine"]):
        _RENDER_REGISTRY[backend] = cls
        return cls
    return decorator


# ==========================================================
# Integrated Engine Registry
# ==========================================================

_INTEGRATED_REGISTRY: dict[Any, Type["BaseIntegratedEngine"]] = {}


def register_integrated_backend(backend: "IntegratedBackend"):
    """Decorator to register an integrated engine for a backend."""
    def decorator(cls: Type["BaseIntegratedEngine"]):
        _INTEGRATED_REGISTRY[backend] = cls
        return cls
    return decorator

