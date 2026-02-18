"""Tests for configuration models."""

import pytest
from rlive_sim.config import (
    SimulationConfig,
    PhysicsConfig,
    RenderConfig,
    IntegratedConfig,
    PhysicsBackend,
    RenderBackend,
    IntegratedBackend,
)


class TestPhysicsConfig:
    """Test PhysicsConfig validation."""

    def test_default_values(self):
        """Test default configuration values."""
        config = PhysicsConfig()
        assert config.backend == PhysicsBackend.SIMPLE
        assert config.dt == 0.01
        assert config.gravity == [0.0, 0.0, -9.81]

    def test_dt_validation(self):
        """Test that dt must be positive."""
        with pytest.raises(ValueError):
            PhysicsConfig(dt=-0.01)

        with pytest.raises(ValueError):
            PhysicsConfig(dt=0)

    def test_gravity_validation(self):
        """Test that gravity must have 3 components."""
        with pytest.raises(ValueError):
            PhysicsConfig(gravity=[0.0, 0.0])

        with pytest.raises(ValueError):
            PhysicsConfig(gravity=[0.0, 0.0, 0.0, -9.81])

    def test_custom_values(self):
        """Test setting custom values."""
        config = PhysicsConfig(dt=0.001, gravity=[0.0, 0.0, -5.0])
        assert config.dt == 0.001
        assert config.gravity == [0.0, 0.0, -5.0]


class TestRenderConfig:
    """Test RenderConfig validation."""

    def test_default_values(self):
        """Test default configuration values."""
        config = RenderConfig()
        assert config.backend == RenderBackend.OPENCV
        assert config.width == 640
        assert config.height == 480
        assert config.channels == 3

    def test_resolution_method(self):
        """Test get_resolution helper."""
        config = RenderConfig(width=800, height=600)
        assert config.get_resolution() == (600, 800, 3)

    def test_dimension_validation(self):
        """Test that dimensions must be positive."""
        with pytest.raises(ValueError):
            RenderConfig(width=0)

        with pytest.raises(ValueError):
            RenderConfig(height=-1)

    def test_channels_validation(self):
        """Test channel count must be valid."""
        with pytest.raises(ValueError):
            RenderConfig(channels=0)

        with pytest.raises(ValueError):
            RenderConfig(channels=5)


class TestSimulationConfig:
    """Test SimulationConfig."""

    def test_default_values(self):
        """Test default configuration."""
        config = SimulationConfig()
        assert config.use_integrated is False
        assert config.max_episode_steps == 100

    def test_observation_shape_separate(self):
        """Test observation shape for separate engines."""
        config = SimulationConfig(
            use_integrated=False,
            render=RenderConfig(width=800, height=600),
        )
        assert config.get_observation_shape() == (600, 800, 3)

    def test_observation_shape_integrated(self):
        """Test observation shape for integrated engine."""
        config = SimulationConfig(
            use_integrated=True,
            integrated=IntegratedConfig(width=1024, height=768),
        )
        assert config.get_observation_shape() == (768, 1024, 3)

    def test_max_episode_steps_validation(self):
        """Test that max_episode_steps must be positive."""
        with pytest.raises(ValueError):
            SimulationConfig(max_episode_steps=0)

        with pytest.raises(ValueError):
            SimulationConfig(max_episode_steps=-10)


class TestIntegratedConfig:
    """Test IntegratedConfig."""

    def test_default_values(self):
        """Test default integrated config."""
        config = IntegratedConfig()
        assert config.dt == 0.01
        assert config.gravity == [0.0, 0.0, -9.81]
        assert config.width == 640
        assert config.height == 480
        assert config.channels == 3

    def test_backend_required(self):
        """Test that backend can be optional."""
        config = IntegratedConfig(backend=IntegratedBackend.GODOT)
        assert config.backend == IntegratedBackend.GODOT

