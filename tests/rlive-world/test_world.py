"""Tests for the World class."""

import unittest
import numpy as np
from unittest.mock import MagicMock, patch

from rlive_common.core.response import BaseResponse, AttachHardwareResponse, DetachHardwareResponse
from rlive_common.core.request import StepRequest, ResetRequest, AttachHardwareRequest, DetachHardwareRequest
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
        request = AttachHardwareRequest()

        result = world.attach_hardware(request)

        self.assertIsInstance(result, AttachHardwareResponse)
        self.assertIn("status", result.info)
        self.assertIn("msg", result.info)
        self.assertTrue(result.success)

    def test_detach_hardware(self):
        """Test that detach_hardware returns success response."""
        world = ToyWorld()
        request = DetachHardwareRequest()

        result = world.detach_hardware(request)

        self.assertIsInstance(result, DetachHardwareResponse)
        self.assertIn("status", result.info)
        self.assertIn("msg", result.info)
        self.assertTrue(result.success)

    def test_make_observation(self):
        """Test _make_observation returns correct array shape and type."""
        world = ToyWorld()
        # Use the mocked version from ToyWorld
        obs = world._make_observation()

        self.assertIsInstance(obs, np.ndarray)
        self.assertEqual(obs.shape, (480, 640, 3))
        self.assertEqual(obs.dtype, np.uint8)

    def test_reset(self):
        """Test that reset returns a valid response."""
        world = ToyWorld()
        request = ResetRequest()
        result = world.reset(request)

        self.assertIsInstance(result, BaseResponse)

    def test_step(self):
        """Test that step returns a valid response with correct observation."""
        world = ToyWorld()
        # Setup mock robot to avoid NoneType error
        world.robot = MagicMock()
        world.robot.move = MagicMock()

        request = StepRequest(action=5)
        result = world.step(request)

        self.assertIsInstance(result, BaseResponse)
        self.assertEqual(result.observation.shape, (480, 640, 3))
        self.assertEqual(result.observation.dtype, np.uint8)
        world.robot.move.assert_called_once_with(5)
