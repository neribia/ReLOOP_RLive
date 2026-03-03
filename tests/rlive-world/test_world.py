"""Tests for the World class."""

import unittest
from unittest.mock import MagicMock

import numpy as np

from rlive_common.core.response import BaseResponse, AttachHardwareResponse, DetachHardwareResponse
from rlive_common.core.request import StepRequest, ResetRequest, AttachHardwareRequest, DetachHardwareRequest
from rlive_common.core.hardware_config import CameraConfig, BoltConfig
from rlive_world.world import World


class ToyWorld(World):
    """Test subclass of World with mocked observation."""

    def _make_observation(self) -> np.ndarray:
        """Return a zero array for testing."""
        return np.zeros((480, 640, 3), dtype=np.uint8)


class TestWorld(unittest.TestCase):
    """Test suite for World class."""

    def test_attach_hardware(self):
        """Test that attach_hardware returns success response."""
        world = ToyWorld()
        request = AttachHardwareRequest(bolt_config=BoltConfig(use_dummy=True))

        result = world.attach_hardware(request)

        self.assertIsInstance(result, AttachHardwareResponse)
        self.assertIn("status", result.info)
        self.assertIn("msg", result.info)
        self.assertTrue(result.success)
        self.assertTrue(world._hardware_attached)

    def test_attach_hardware_idempotent(self):
        """Test that attach_hardware can be called multiple times safely."""
        world = ToyWorld()
        request = AttachHardwareRequest(bolt_config=BoltConfig(use_dummy=True))

        # First attach
        result1 = world.attach_hardware(request)
        self.assertTrue(result1.success)
        self.assertEqual(result1.info["status"], "ok")

        # Second attach should return already_attached
        result2 = world.attach_hardware(request)
        self.assertTrue(result2.success)
        self.assertEqual(result2.info["status"], "already_attached")

    def test_detach_hardware(self):
        """Test that detach_hardware returns success response."""
        world = ToyWorld()

        # First attach
        world.attach_hardware(AttachHardwareRequest(bolt_config=BoltConfig(use_dummy=True)))

        # Then detach
        request = DetachHardwareRequest()
        result = world.detach_hardware(request)

        self.assertIsInstance(result, DetachHardwareResponse)
        self.assertIn("status", result.info)
        self.assertIn("msg", result.info)
        self.assertTrue(result.success)
        self.assertFalse(world._hardware_attached)

    def test_detach_hardware_when_not_attached(self):
        """Test that detach_hardware when not attached is safe."""
        world = ToyWorld()
        request = DetachHardwareRequest()

        result = world.detach_hardware(request)

        self.assertTrue(result.success)
        self.assertEqual(result.info["status"], "not_attached")

    def test_make_observation_requires_hardware(self):
        """Test that _make_observation raises error when hardware not attached."""
        world = World()

        # Should raise RuntimeError
        with self.assertRaises(RuntimeError) as context:
            world._make_observation()

        self.assertIn("Hardware not attached", str(context.exception))

    def test_reset_requires_hardware(self):
        """Test that reset raises error when hardware not attached."""
        world = ToyWorld()
        request = ResetRequest()

        # Should raise RuntimeError
        with self.assertRaises(RuntimeError) as context:
            world.reset(request)

        self.assertIn("Hardware not attached", str(context.exception))

    def test_step_requires_hardware(self):
        """Test that step raises error when hardware not attached."""
        world = ToyWorld()
        request = StepRequest(action=np.array([90, 50, 1]))

        # Should raise RuntimeError
        with self.assertRaises(RuntimeError) as context:
            world.step(request)

        self.assertIn("Hardware not attached", str(context.exception))

    def test_make_observation(self):
        """Test _make_observation returns correct array shape and type."""
        world = World()

        # Attach hardware to initialize camera and robot
        world.attach_hardware(AttachHardwareRequest(bolt_config=BoltConfig(use_dummy=True)))

        obs = world._make_observation()

        self.assertIsInstance(obs, np.ndarray)
        self.assertEqual(obs.shape, (480, 640, 3))
        self.assertEqual(obs.dtype, np.uint8)

        # Clean up
        world.detach_hardware(DetachHardwareRequest())

    def test_reset(self):
        """Test that reset returns a valid response."""
        world = ToyWorld()

        # Attach hardware first
        world.attach_hardware(AttachHardwareRequest(bolt_config=BoltConfig(use_dummy=True)))

        request = ResetRequest()
        result = world.reset(request)

        self.assertIsInstance(result, BaseResponse)

        # Clean up
        world.detach_hardware(DetachHardwareRequest())

    def test_step(self):
        """Test that step returns a valid response with correct observation."""
        world = ToyWorld()

        # Attach hardware first
        world.attach_hardware(AttachHardwareRequest(bolt_config=BoltConfig(use_dummy=True)))

        # Setup mock robot to avoid NoneType error
        world.robot = MagicMock()
        world.robot.move = MagicMock()

        action = np.array([90, 50, 1])
        request = StepRequest(action=action)
        result = world.step(request)

        self.assertIsInstance(result, BaseResponse)
        self.assertEqual(result.observation.shape, (480, 640, 3))
        self.assertEqual(result.observation.dtype, np.uint8)
        world.robot.move.assert_called_once_with(heading=90, speed=50, duration=1.0)

        # Clean up
        world.detach_hardware(DetachHardwareRequest())
