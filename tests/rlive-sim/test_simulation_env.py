"""Tests for SimulationEnv."""

import pytest
import numpy as np
from rlive_sim import SimulationEnv
from rlive_sim.config import SimulationConfig, PhysicsConfig, RenderConfig


class TestSimulationEnv:
    """Test SimulationEnv Gymnasium compatibility."""

    def test_initialization(self):
        """Test default environment initialization."""
        env = SimulationEnv()
        assert env is not None
        assert env.observation_space is not None
        assert env.action_space is not None

    def test_reset(self):
        """Test environment reset."""
        env = SimulationEnv()
        obs, info = env.reset()

        assert isinstance(obs, np.ndarray)
        assert obs.shape == env.observation_space.shape
        assert obs.dtype == np.uint8
        assert isinstance(info, dict)
        assert "episode" in info

    def test_step(self):
        """Test environment step."""
        env = SimulationEnv()
        env.reset()

        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        assert isinstance(obs, np.ndarray)
        assert obs.shape == env.observation_space.shape
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_max_episode_steps(self):
        """Test that episode truncates at max_episode_steps."""
        env = SimulationEnv(max_episode_steps=5)
        env.reset()

        truncated = False
        for _ in range(10):
            _, _, terminated, truncated, _ = env.step(env.action_space.sample())
            if terminated or truncated:
                break

        assert truncated

    def test_close(self):
        """Test environment cleanup."""
        env = SimulationEnv()
        env.reset()
        env.close()  # Should not raise

    def test_render_mode(self):
        """Test render mode."""
        env = SimulationEnv(render_mode="opencv")
        env.reset()
        image = env.render()
        assert image is not None


class TestSimulationEnvWithCustomConfig:
    """Test SimulationEnv with custom configuration."""

    def test_custom_resolution(self):
        """Test custom render resolution."""
        config = SimulationConfig(
            render=RenderConfig(width=800, height=600)
        )
        env = SimulationEnv(config=config)
        obs, _ = env.reset()

        assert obs.shape == (600, 800, 3)  # height, width, channels

    def test_custom_physics_params(self):
        """Test physics engine with custom parameters."""
        config = SimulationConfig(
            physics=PhysicsConfig(
                dt=0.01,
                extra={
                    "box_width": 800,
                    "box_height": 600,
                    "ball_radius": 15,
                }
            ),
            render=RenderConfig(width=800, height=600)
        )
        env = SimulationEnv(config=config)
        obs, _ = env.reset()
        assert obs.shape == (600, 800, 3)

    def test_reset_reproducibility(self):
        """Test that reset produces valid initial state."""
        env = SimulationEnv()
        obs1, info1 = env.reset(seed=42)
        env.close()

        env2 = SimulationEnv()
        obs2, info2 = env2.reset(seed=42)
        env2.close()

        # With same seed, should get reproducible results
        assert info1["episode"] > 0
        assert info2["episode"] > 0

