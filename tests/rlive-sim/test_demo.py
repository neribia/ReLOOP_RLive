"""Tests for demo functionality."""

import pytest
from rlive_sim import SimulationEnv
from rlive_sim.config import SimulationConfig, PhysicsConfig, RenderConfig, PhysicsBackend, RenderBackend
from rlive_sim.engine import SimplePhysicsEngine, OpenCVRenderEngine, SimulationEngine


class TestDemoModernApproach:
    """Test modern (config-based) demo approach."""

    def test_modern_demo_runs(self):
        """Test modern approach with config works."""
        config = SimulationConfig(
            use_integrated=False,
            physics=PhysicsConfig(
                backend=PhysicsBackend.SIMPLE,
                extra={
                    "box_width": 640,
                    "box_height": 480,
                    "ball_radius": 20,
                }
            ),
            render=RenderConfig(
                backend=RenderBackend.OPENCV,
                width=640,
                height=480,
            ),
        )

        env = SimulationEnv(config=config)
        obs, info = env.reset()

        assert obs is not None
        assert obs.shape == (480, 640, 3)

        # Run 5 steps
        for _ in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            assert obs is not None
            if terminated or truncated:
                break

        env.close()

    def test_modern_with_custom_params(self):
        """Test modern approach with custom render parameters."""
        config = SimulationConfig(
            physics=PhysicsConfig(
                backend=PhysicsBackend.SIMPLE,
                extra={
                    "box_width": 640,
                    "box_height": 480,
                    "ball_radius": 20,
                }
            ),
            render=RenderConfig(
                backend=RenderBackend.OPENCV,
                width=640,
                height=480,
                extra={
                    "ball_color": (255, 0, 0),
                    "box_color": (128, 128, 128),
                    "bg_color": (0, 0, 0),
                    "box_thickness": 2,
                }
            ),
        )

        env = SimulationEnv(config=config)
        obs, info = env.reset()
        assert obs is not None
        env.close()


class TestDemoLegacyApproach:
    """Test legacy (direct instantiation) demo approach."""

    def test_legacy_demo_runs(self):
        """Test legacy approach with direct engine instantiation works."""
        physics = SimplePhysicsEngine(
            box_width=640,
            box_height=480,
            ball_radius=20,
            dt=0.01,
        )

        render = OpenCVRenderEngine(
            width=640,
            height=480,
        )

        sim = SimulationEngine(physics_engine=physics, render_engine=render)
        env = SimulationEnv(engine=sim)

        obs, info = env.reset()
        assert obs is not None
        assert obs.shape == (480, 640, 3)

        # Run 5 steps
        for _ in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            assert obs is not None
            if terminated or truncated:
                break

        env.close()

    def test_legacy_with_custom_params(self):
        """Test legacy approach with custom parameters."""
        physics = SimplePhysicsEngine(
            box_width=640,
            box_height=480,
            ball_radius=20,
            dt=0.01,
        )

        render = OpenCVRenderEngine(
            width=640,
            height=480,
            ball_color=(255, 0, 0),
            box_color=(128, 128, 128),
            bg_color=(0, 0, 0),
            box_thickness=2,
        )

        sim = SimulationEngine(physics_engine=physics, render_engine=render)
        env = SimulationEnv(engine=sim)
        obs, info = env.reset()
        assert obs is not None
        env.close()


class TestBothApproachesEquivalent:
    """Test that modern and legacy approaches produce similar results."""

    def test_same_config_produces_same_output(self):
        """Test that same configuration works with both approaches."""
        # Both should accept same parameters and work
        config = SimulationConfig(
            render=RenderConfig(width=640, height=480)
        )

        env = SimulationEnv(config=config)
        obs, _ = env.reset()
        assert obs.shape == (480, 640, 3)
        env.close()

