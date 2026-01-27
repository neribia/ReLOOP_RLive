"""Tests for request models."""

import unittest

from rlive_common.core.request import (
    StepRequest,
    ResetRequest,
    AttachHardwareRequest,
    DetachHardwareRequest,
)


class TestRequest(unittest.TestCase):

    """Test suite for request models."""

    def test_step_request_creation(self):
        """Test that StepRequest can be created with action."""
        request = StepRequest(action=5)
        self.assertEqual(request.action, 5)

    def test_reset_request_creation(self):
        """Test that ResetRequest can be created."""
        request = ResetRequest()
        self.assertIsNotNone(request)

    def test_attach_hardware_request_creation(self):
        """Test that AttachHardwareRequest can be created."""
        request = AttachHardwareRequest()
        self.assertIsNotNone(request)

    def test_attach_hardware_request_with_params(self):
        """Test AttachHardwareRequest with custom parameters."""
        request = AttachHardwareRequest(
            camera_type="webcam",
            robot_name="TestBolt",
            use_dummy=True
        )
        self.assertEqual(request.camera_type, "webcam")
        self.assertEqual(request.robot_name, "TestBolt")
        self.assertTrue(request.use_dummy)

    def test_attach_hardware_request_defaults(self):
        """Test AttachHardwareRequest default values."""
        request = AttachHardwareRequest()
        self.assertIsNone(request.camera_type)
        self.assertIsNone(request.robot_name)
        self.assertFalse(request.use_dummy)

    def test_detach_hardware_request_creation(self):
        """Test that DetachHardwareRequest can be created."""
        request = DetachHardwareRequest()
        self.assertIsNotNone(request)