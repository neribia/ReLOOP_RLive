"""Factory module for creating engines from configuration.

Provides factory classes that dispatch config to appropriate engine implementations.
Factories use registries populated by @register_*_backend decorators on concrete engines.
"""
from typing import Any

from rlive_sim.config import PhysicsConfig, RenderConfig, IntegratedConfig
from rlive_sim.engine.registry import _PHYSICS_REGISTRY, _RENDER_REGISTRY, _INTEGRATED_REGISTRY



# ==========================================================
# Public Factory Class / Dispatcher
# ==========================================================

class PhysicsEngine:
    """Factory class for creating physics engines from config.

    Usage:
        config = PhysicsConfig(backend=PhysicsBackend.PYMUNK, dt=0.01)
        engine = PhysicsEngine(config)  # Returns appropriate engine instance
    """

    def __new__(cls, config: "PhysicsConfig", **kwargs: Any):
        """Create physics engine from config.

        Args:
            config: PhysicsConfig instance with backend and parameters.
            **kwargs: Additional engine-specific parameters.

        Returns:
            BasePhysicsEngine: Instantiated engine of appropriate type.

        Raises:
            ValueError: If backend is not registered.
        """
        engine_cls = _PHYSICS_REGISTRY.get(config.backend)

        if engine_cls is None:
            raise ValueError(
                f"No physics engine registered for backend {config.backend}. "
                f"Available: {list(_PHYSICS_REGISTRY.keys())}"
            )

        # Create engine instance with config parameters
        return engine_cls(
            dt=config.dt,
            gravity=config.gravity,
            **config.extra,
            **kwargs,
        )


# ==========================================================
# Public Factory Class / Dispatcher
# ==========================================================

class RenderEngine:
    """Factory class for creating render engines from config.

    Usage:
        config = RenderConfig(backend=RenderBackend.OPENCV, width=640, height=480)
        engine = RenderEngine(config)  # Returns appropriate engine instance
    """

    def __new__(cls, config: "RenderConfig", **kwargs: Any):
        """Create render engine from config.

        Args:
            config: RenderConfig instance with backend and parameters.
            **kwargs: Additional engine-specific parameters.

        Returns:
            BaseRenderEngine: Instantiated engine of appropriate type.

        Raises:
            ValueError: If backend is not registered.
        """
        engine_cls = _RENDER_REGISTRY.get(config.backend)

        if engine_cls is None:
            raise ValueError(
                f"No render engine registered for backend {config.backend}. "
                f"Available: {list(_RENDER_REGISTRY.keys())}"
            )

        # Create engine instance with config parameters
        return engine_cls(
            width=config.width,
            height=config.height,
            channels=config.channels,
            **config.extra,
            **kwargs,
        )


# ==========================================================
# Integrated Engine Factory Class / Dispatcher
# ==========================================================

class IntegratedEngine:
    """Factory class for creating integrated engines from config.

    Usage:
        config = IntegratedConfig(backend=IntegratedBackend.MUJOCO)
        engine = IntegratedEngine(config)  # Returns appropriate engine instance
    """

    def __new__(cls, config: "IntegratedConfig", **kwargs: Any):
        """Create integrated engine from config.

        Args:
            config: IntegratedConfig instance with backend and parameters.
            **kwargs: Additional engine-specific parameters.

        Returns:
            BaseIntegratedEngine: Instantiated engine of appropriate type.

        Raises:
            ValueError: If backend is not registered.
        """
        engine_cls = _INTEGRATED_REGISTRY.get(config.backend)

        if engine_cls is None:
            raise ValueError(
                f"No integrated engine registered for backend {config.backend}. "
                f"Available: {list(_INTEGRATED_REGISTRY.keys())}"
            )

        # Create engine instance with config parameters
        return engine_cls(
            dt=config.dt,
            gravity=config.gravity,
            width=config.width,
            height=config.height,
            channels=config.channels,
            **config.extra,
            **kwargs,
        )

