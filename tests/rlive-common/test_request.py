"""Tests for request models."""

import unittest

import numpy as np

from rlive_common.core.request import (
    StepRequest,
    ResetRequest,
    AttachHardwareRequest,
    DetachHardwareRequest,
)
from rlive_common.core.hardware_config import CameraConfig, BoltConfig
from rlive_common.core.enums import CameraType


class TestRequest(unittest.TestCase):

    """Test suite for request models."""

    def test_step_request_creation(self):
        """Test that StepRequest can be created with action."""
        action = np.array([90, 50, 100])
        request = StepRequest(action=action)
        np.testing.assert_array_equal(request.action, action)

    def test_reset_request_creation(self):
        """Test that ResetRequest can be created."""
        request = ResetRequest()
        self.assertIsNotNone(request)

    def test_attach_hardware_request_creation(self):
        """Test that AttachHardwareRequest can be created."""
        request = AttachHardwareRequest()
        self.assertIsNotNone(request)
        self.assertIsNotNone(request.camera_config)
        self.assertIsNotNone(request.bolt_config)

    def test_attach_hardware_request_with_camera_config(self):
        """Test AttachHardwareRequest with custom CameraConfig."""
        camera_cfg = CameraConfig(type="webcam", id=1, width=1280, height=720)
        bolt_cfg = BoltConfig(name="TestBolt", use_dummy=True)
        request = AttachHardwareRequest(
            camera_config=camera_cfg,
            bolt_config=bolt_cfg
        )
        self.assertEqual(request.camera_config.type, "webcam")
        self.assertEqual(request.camera_config.id, 1)
        self.assertEqual(request.camera_config.width, 1280)
        self.assertEqual(request.camera_config.height, 720)
        self.assertEqual(request.bolt_config.name, "TestBolt")
        self.assertTrue(request.bolt_config.use_dummy)

    def test_attach_hardware_request_with_none_values(self):
        """Test AttachHardwareRequest with None values (uses rlive-world defaults)."""
        camera_cfg = CameraConfig()
        bolt_cfg = BoltConfig()
        request = AttachHardwareRequest(
            camera_config=camera_cfg,
            bolt_config=bolt_cfg
        )
        # All values should be None, rlive-world will apply defaults
        self.assertIsNone(request.camera_config.type)
        self.assertIsNone(request.camera_config.id)
        self.assertIsNone(request.camera_config.width)
        self.assertIsNone(request.camera_config.height)
        self.assertIsNone(request.bolt_config.name)
        self.assertFalse(request.bolt_config.use_dummy)

    def test_camera_config_with_enum(self):
        """Test CameraConfig accepts CameraType enum."""
        camera_cfg = CameraConfig(type=CameraType.DUMMY)
        self.assertEqual(camera_cfg.type, "dummy")

    def test_camera_config_with_string(self):
        """Test CameraConfig accepts string camera type."""
        camera_cfg = CameraConfig(type="picam")
        self.assertEqual(camera_cfg.type, "picam")

    def test_camera_config_invalid_type(self):
        """Test CameraConfig rejects invalid camera type."""
        with self.assertRaises(ValueError):
            CameraConfig(type="invalid_camera")

    def test_bolt_config_color_constraints(self):
        """Test BoltConfig color values are constrained to 0-255."""
        # Valid colors
        bolt_cfg = BoltConfig(color_r=255, color_g=128, color_b=0)
        self.assertEqual(bolt_cfg.color_r, 255)

        # Invalid colors should raise validation error
        with self.assertRaises(ValueError):
            BoltConfig(color_r=256)

        with self.assertRaises(ValueError):
            BoltConfig(color_g=-1)

    def test_detach_hardware_request_creation(self):
        """Test that DetachHardwareRequest can be created."""
        request = DetachHardwareRequest()
        self.assertIsNotNone(request)