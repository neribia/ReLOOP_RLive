"""Tests for the WorldInterface client."""

import unittest
import json

import numpy as np
import httpx
from httpx import Response

from rlive_env.world_client import WorldInterface, ApiError
from rlive_common.core.request import (
    StepRequest,
    ResetRequest,
    AttachHardwareRequest,
    DetachHardwareRequest,
)
from rlive_common.core.response import (
    StepResponseMultipart,
    StepResponseJSON,
    ResetResponse,
    AttachHardwareResponse,
    DetachHardwareResponse,
)


class TestWorldInterface(unittest.TestCase):
    """Test suite for WorldInterface HTTP client."""

    def setUp(self):
        """Set up test fixtures with mock transport."""
        self.base_url = "http://test-world"
        self.zero_image = np.zeros((480, 640, 3), dtype=np.uint8)

        # Helper: Create interface using transport handler
        def make_iface(handler):
            transport = httpx.MockTransport(handler)
            client = httpx.Client(base_url=self.base_url, transport=transport)
            return WorldInterface(base_url=self.base_url, client=client)

        self.make_iface = make_iface

        def make_json_handler(path, expected_request, response_model):
            def handler(req):
                self.assertEqual(req.method, "POST")
                self.assertEqual(req.url.path, path)
                self.assert_json_body(req, expected_request)
                return Response(200, json=response_model.model_dump())

            return handler

        self.make_handler = make_json_handler

        def assert_json_body(req, expected_model):
            self.assertEqual(json.loads(req.content), expected_model.model_dump())

        self.assert_json_body = assert_json_body

        def assert_numpy_image(image, expected_image):
            np.testing.assert_array_equal(image, expected_image)
            self.assertEqual(image.dtype, np.uint8)
            self.assertEqual(image.shape, expected_image.shape)

        self.assert_numpy_image = assert_numpy_image

    # -------------------------------------------------------------
    # /reset
    # -------------------------------------------------------------
    def test_reset_returns_resetresponse(self):
        """Test that reset returns a valid ResetResponse."""
        expected = ResetResponse(
            observation=self.zero_image,
            truncated=False,
            info={"debug": True},
        )

        def handler(req: httpx.Request) -> Response:
            self.assertEqual(req.method, "POST")
            self.assertEqual(req.url.path, "/reset")

            self.assert_json_body(req, ResetRequest())

            return Response(200, json=expected.model_dump())

        iface = self.make_iface(handler)
        result = iface.reset()

        self.assertIsInstance(result, ResetResponse)
        self.assert_numpy_image(result.observation, expected.observation)
        self.assertEqual(result.info, expected.info)

    def test_reset_raises_on_http_error(self):
        """Test that reset raises ApiError on HTTP 500 error."""
        def handler(req):
            return Response(500, json={"error": "server"})

        iface = self.make_iface(handler)
        with self.assertRaises(ApiError):
            iface.reset()

    # -------------------------------------------------------------
    # /step_json
    # -------------------------------------------------------------
    def test_step_json_returns_stepresponsejson(self):
        """Test that step_json returns a valid StepResponseJSON."""
        expected = StepResponseJSON(
            observation=self.zero_image,
            truncated=False,
            info={"foo": "bar"},
        )

        def handler(req: httpx.Request) -> Response:
            self.assertEqual(req.method, "POST")
            self.assertEqual(req.url.path, "/step_json")

            self.assert_json_body(req, StepRequest(action=7))

            return Response(200, json=expected.model_dump())

        iface = self.make_iface(handler)
        result = iface.step_json(action=7)

        self.assertIsInstance(result, StepResponseJSON)
        self.assert_numpy_image(result.observation, expected.observation)
        self.assertEqual(result.info, expected.info)

    # -------------------------------------------------------------
    # /step_multipart
    # -------------------------------------------------------------
    def test_step_multipart_returns_stepresponsemultipart(self):
        """Test that step_multipart returns a valid StepResponseMultipart."""
        expected = StepResponseMultipart(
            observation=self.zero_image,
            truncated=False,
            info={},
        )

        body, content_type = expected.encode()

        def handler(req: httpx.Request) -> Response:
            self.assertEqual(req.method, "POST")
            self.assertEqual(req.url.path, "/step_multipart")

            self.assert_json_body(req, StepRequest(action=5))

            return Response(
                status_code=200,
                content=body,
                headers={"Content-Type": content_type},
            )

        iface = self.make_iface(handler)
        result = iface.step_multipart(5)

        self.assertIsInstance(result, StepResponseMultipart)
        self.assert_numpy_image(result.observation, expected.observation)
        self.assertEqual(result.info, expected.info)

    def test_step_multipart_invalid_content_type(self):
        """Test that step_multipart raises ValueError on invalid content type."""
        def handler(req):
            return Response(200, content=b"junk", headers={"Content-Type": "text/plain"})

        iface = self.make_iface(handler)
        with self.assertRaises(ValueError):
            iface.step_multipart(5)

    # -------------------------------------------------------------
    # /attach_hardware
    # -------------------------------------------------------------
    def test_attach_hardware_returns_attachhardwareresponse(self):
        """Test that attach_hardware returns a valid AttachHardwareResponse."""
        expected = AttachHardwareResponse(success=True, info={"status": "ok"})

        def handler(req: httpx.Request) -> Response:
            self.assertEqual(req.method, "POST")
            self.assertEqual(req.url.path, "/attach_hardware")

            self.assert_json_body(req, AttachHardwareRequest())

            return Response(200, json=expected.model_dump())

        iface = self.make_iface(handler)
        result = iface.attach_hardware()

        self.assertIsInstance(result, AttachHardwareResponse)
        self.assertTrue(result.success)

    # -------------------------------------------------------------
    # /detach_hardware
    # -------------------------------------------------------------
    def test_detach_hardware_returns_detachhardwareresponse(self):
        """Test that detach_hardware returns a valid DetachHardwareResponse."""
        expected = DetachHardwareResponse(success=True, info={"status": "ok"})

        def handler(req: httpx.Request) -> Response:
            self.assertEqual(req.method, "POST")
            self.assertEqual(req.url.path, "/detach_hardware")

            self.assert_json_body(req, DetachHardwareRequest())

            return Response(200, json=expected.model_dump())

        iface = self.make_iface(handler)
        result = iface.detach_hardware()

        self.assertIsInstance(result, DetachHardwareResponse)
        self.assertTrue(result.success)

    # -------------------------------------------------------------
    # /health
    # -------------------------------------------------------------
    def test_health_check_returns_status(self):
        """Test that health_check returns health status."""
        expected = {"status": "healthy", "hardware_attached": True}

        def handler(req: httpx.Request) -> Response:
            self.assertEqual(req.method, "GET")
            self.assertEqual(req.url.path, "/health")
            return Response(200, json=expected)

        iface = self.make_iface(handler)
        result = iface.health_check()

        self.assertEqual(result, expected)
        self.assertEqual(result["status"], "healthy")

    def test_health_check_handles_error(self):
        """Test that health_check returns unhealthy on error."""
        def handler(req: httpx.Request) -> Response:
            return Response(500, json={"error": "server down"})

        iface = self.make_iface(handler)
        result = iface.health_check()

        self.assertEqual(result["status"], "unhealthy")
        self.assertIn("error", result)

    # -------------------------------------------------------------
    # /status
    # -------------------------------------------------------------
    def test_get_status_returns_detailed_status(self):
        """Test that get_status returns detailed server status."""
        expected = {
            "hardware_attached": True,
            "camera_active": True,
            "robot_connected": False
        }

        def handler(req: httpx.Request) -> Response:
            self.assertEqual(req.method, "GET")
            self.assertEqual(req.url.path, "/status")
            return Response(200, json=expected)

        iface = self.make_iface(handler)
        result = iface.get_status()

        self.assertEqual(result, expected)
        self.assertTrue(result["hardware_attached"])
        self.assertTrue(result["camera_active"])
        self.assertFalse(result["robot_connected"])

    def test_get_status_handles_error(self):
        """Test that get_status returns error dict on failure."""
        def handler(req: httpx.Request) -> Response:
            return Response(500, json={"error": "internal error"})

        iface = self.make_iface(handler)
        result = iface.get_status()

        self.assertIn("error", result)

