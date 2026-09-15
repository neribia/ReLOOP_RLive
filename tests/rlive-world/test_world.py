"""Tests for the World class."""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from rlive_common.core.response import BaseResponse, AttachHardwareResponse, DetachHardwareResponse
from rlive_common.core.request import StepRequest, ResetRequest, AttachHardwareRequest, DetachHardwareRequest
from rlive_common.core.hardware_config import WorldConfig
from rlive_world.camera import CameraService
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
        request = AttachHardwareRequest(world_config=WorldConfig(bolt_use_dummy=True))

        result = world.attach_hardware(request)

        self.assertIsInstance(result, AttachHardwareResponse)
        self.assertIn("status", result.info)
        self.assertIn("msg", result.info)
        self.assertTrue(result.success)
        self.assertTrue(world._hardware_attached)

    def test_attach_hardware_idempotent(self):
        """Test that attach_hardware can be called multiple times safely."""
        world = ToyWorld()
        request = AttachHardwareRequest(world_config=WorldConfig(bolt_use_dummy=True))

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
        world.attach_hardware(AttachHardwareRequest(world_config=WorldConfig(bolt_use_dummy=True)))

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
        world.attach_hardware(AttachHardwareRequest(world_config=WorldConfig(bolt_use_dummy=True)))

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
        world.attach_hardware(AttachHardwareRequest(world_config=WorldConfig(bolt_use_dummy=True)))

        request = ResetRequest()
        result = world.reset(request)

        self.assertIsInstance(result, BaseResponse)

        # Clean up
        world.detach_hardware(DetachHardwareRequest())

    def test_step(self):
        """Test that step returns a valid response with correct observation."""
        world = ToyWorld()

        # Attach hardware first
        world.attach_hardware(AttachHardwareRequest(world_config=WorldConfig(bolt_use_dummy=True)))

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


class TestWorldHardwareCleanup(unittest.TestCase):
    """Regression tests for hardware rollback on failed attach and on shutdown.

    Covers the bug where a peripheral that failed mid-sequence left the other one
    connected while `_hardware_attached` stayed False, which in turn made
    `detach_hardware()` and the shutdown hook skip the cleanup entirely.
    """

    @staticmethod
    def _dummy_request() -> AttachHardwareRequest:
        """Attach request that uses no real hardware at all."""
        return AttachHardwareRequest(
            world_config=WorldConfig(bolt_use_dummy=True, camera_type="dummy")
        )

    def test_attach_rolls_back_camera_when_bolt_fails(self):
        """A bolt failure must release the camera that was already set up."""
        world = ToyWorld()

        with (
            patch.object(CameraService, "release", autospec=True) as mock_release,
            patch("rlive_world.world.SpheroBoltPlus") as mock_bolt,
        ):
            mock_bolt.return_value.connect.side_effect = RuntimeError("Sphero 'DummyBolt' not found")

            with self.assertRaises(RuntimeError):
                world.attach_hardware(self._dummy_request())

            mock_release.assert_called_once()

        self.assertIsNone(world.camera)
        self.assertIsNone(world.robot)
        self.assertFalse(world._hardware_attached)

    def test_attach_rolls_back_when_camera_fails(self):
        """A camera failure must leave no bolt connected behind it."""
        world = ToyWorld()

        camera_error = RuntimeError("Camera 0 could not be opened!")
        with (
            patch.object(CameraService, "setup", side_effect=camera_error),
            patch("rlive_world.world.SpheroBoltPlus") as mock_bolt,
        ):
            with self.assertRaises(RuntimeError):
                world.attach_hardware(self._dummy_request())

            # The camera is set up first, so the bolt must never even be contacted.
            mock_bolt.return_value.connect.assert_not_called()

        self.assertIsNone(world.camera)
        self.assertIsNone(world.robot)
        self.assertFalse(world._hardware_attached)

    def test_detach_releases_orphaned_hardware(self):
        """detach_hardware must clean up hardware left live by a failed attach.

        This state used to be unrecoverable without restarting the server: hardware
        connected, `_hardware_attached` False, so detach reported "not_attached" and
        released nothing.
        """
        world = ToyWorld()
        world.camera = MagicMock()
        world.robot = MagicMock()
        world._hardware_attached = False
        camera, robot = world.camera, world.robot

        result = world.detach_hardware(DetachHardwareRequest())

        camera.release.assert_called_once()
        robot.disconnect.assert_called_once()
        self.assertIsNone(world.camera)
        self.assertIsNone(world.robot)
        self.assertTrue(result.success)
        self.assertEqual(result.info["status"], "not_attached")

    def test_disconnect_all_hardware_is_idempotent(self):
        """Calling the cleanup twice must be safe."""
        world = ToyWorld()
        world.attach_hardware(self._dummy_request())

        world.disconnect_all_hardware()
        world.disconnect_all_hardware()

        self.assertIsNone(world.camera)
        self.assertIsNone(world.robot)
        self.assertFalse(world._hardware_attached)

    def test_disconnect_all_hardware_continues_after_camera_error(self):
        """One peripheral failing to release must not strand the other."""
        world = ToyWorld()
        world.camera = MagicMock()
        world.camera.release.side_effect = RuntimeError("camera wedged")
        world.robot = MagicMock()
        robot = world.robot

        world.disconnect_all_hardware()

        robot.disconnect.assert_called_once()
        self.assertIsNone(world.camera)
        self.assertIsNone(world.robot)
        self.assertFalse(world._hardware_attached)

    def test_close_releases_hardware(self):
        """World.close() is the shutdown hook, so it must actually disconnect."""
        world = ToyWorld()
        world.attach_hardware(self._dummy_request())
        robot = world.robot

        world.close()

        self.assertIsNone(world.camera)
        self.assertIsNone(world.robot)
        self.assertFalse(world._hardware_attached)
        self.assertIsNone(robot.api)  # the dummy BLE session was really exited

    def test_dummy_hardware_does_not_open_a_real_camera(self):
        """bolt_use_dummy with camera_type='dummy' must not touch a webcam.

        The dummy branch used to hardcode type="webcam", so "dummy mode" still
        opened a real capture device and failed on machines without one.
        """
        world = ToyWorld()

        with patch("rlive_world.camera.camera_factory.CameraFactory._build_webcam") as mock_webcam:
            world.attach_hardware(self._dummy_request())

        mock_webcam.assert_not_called()
        self.assertTrue(world._hardware_attached)

        world.detach_hardware(DetachHardwareRequest())
