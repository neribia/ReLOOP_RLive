"""Tests for request models."""

import unittest

import numpy as np

from rlive_common.core.request import (
    StepRequest,
    ResetRequest,
    AttachHardwareRequest,
    DetachHardwareRequest,
)
from rlive_common.core.hardware_config import WorldConfig
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
        self.assertIsNotNone(request.world_config)

    def test_attach_hardware_request_with_world_config(self):
        """Test AttachHardwareRequest with custom WorldConfig."""
        world_cfg = WorldConfig(
            camera_type=CameraType.WEBCAM,
            camera_id=1,
            camera_resolution=(1280, 720),
            bolt_name="TestBolt",
            bolt_use_dummy=True
        )
        request = AttachHardwareRequest(world_config=world_cfg)
        self.assertEqual(request.world_config.camera_type, CameraType.WEBCAM)
        self.assertEqual(request.world_config.camera_id, 1)
        self.assertEqual(request.world_config.camera_resolution, (1280, 720))
        self.assertEqual(request.world_config.bolt_name, "TestBolt")
        self.assertTrue(request.world_config.bolt_use_dummy)

    def test_attach_hardware_request_with_none_values(self):
        """Test AttachHardwareRequest with None values (uses rlive-world defaults)."""
        world_cfg = WorldConfig()
        request = AttachHardwareRequest(world_config=world_cfg)
        # All values should be None, rlive-world will apply defaults
        self.assertIsNone(request.world_config.camera_type)
        self.assertIsNone(request.world_config.camera_id)
        self.assertIsNone(request.world_config.camera_resolution)
        self.assertIsNone(request.world_config.bolt_name)
        self.assertFalse(request.world_config.bolt_use_dummy)


    def test_world_config_with_camera_type_enum(self):
        """Test WorldConfig accepts CameraType enum."""
        world_cfg = WorldConfig(camera_type=CameraType.DUMMY)
        self.assertEqual(world_cfg.camera_type, CameraType.DUMMY)

    def test_world_config_with_camera_type_string(self):
        """Test WorldConfig accepts string camera type."""
        world_cfg = WorldConfig(camera_type="picam")
        self.assertEqual(world_cfg.camera_type, CameraType.PICAM)

    def test_world_config_invalid_camera_type(self):
        """Test WorldConfig rejects invalid camera type."""
        with self.assertRaises(ValueError):
            WorldConfig(camera_type="invalid_camera")

    def test_world_config_bolt_color_constraints(self):
        """Test WorldConfig bolt color values are constrained to 0-255."""
        # Valid colors
        world_cfg = WorldConfig(bolt_color_r=255, bolt_color_g=128, bolt_color_b=0)
        self.assertEqual(world_cfg.bolt_color_r, 255)

        # Invalid colors should raise validation error
        with self.assertRaises(ValueError):
            WorldConfig(bolt_color_r=256)

        with self.assertRaises(ValueError):
            WorldConfig(bolt_color_g=-1)

    def test_detach_hardware_request_creation(self):
        """Test that DetachHardwareRequest can be created."""
        request = DetachHardwareRequest()
        self.assertIsNotNone(request)