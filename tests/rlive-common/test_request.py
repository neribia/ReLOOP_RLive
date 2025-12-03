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

    def test_detach_hardware_request_creation(self):
        """Test that DetachHardwareRequest can be created."""
        request = DetachHardwareRequest()
        self.assertIsNotNone(request)