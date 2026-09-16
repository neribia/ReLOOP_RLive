"""Tests for the World API server endpoints."""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from fastapi.testclient import TestClient

from rlive_world.camera import CameraService
from rlive_world.errors import PermanentHardwareError, TransientHardwareError
from rlive_world.world_server import app, resources  # adjust import to where your app lives


def fake_reset(req):
    """Fake reset handler for testing."""
    from rlive_common.core.response import ResetResponse
    return ResetResponse(observation=np.zeros((2, 2, 3), dtype=np.uint8) , truncated=False, info={})


def fake_step(req):
    """Fake step handler for testing."""
    from rlive_common.core.response import BaseResponse
    return BaseResponse(observation=np.zeros((2, 2, 3), dtype=np.uint8), truncated=False, info={})

def fake_attach(req):
    """Fake attach hardware handler for testing."""
    from rlive_common.core.response import AttachHardwareResponse
    return AttachHardwareResponse(success=True, info={"message": "Detached OK"})

def fake_detach(req):
    """Fake detach hardware handler for testing."""
    from rlive_common.core.response import DetachHardwareResponse
    return DetachHardwareResponse(success=True, info={"message": "Detached OK"})


class TestWorldAPI(unittest.TestCase):
    """Test suite for World API endpoints."""

    def test_attach_hardware_endpoint(self):
        """Test POST /attach_hardware returns success."""
        with TestClient(app) as client:
            resources.world.attach_hardware = fake_attach
            resp = client.post("/attach_hardware", json={})
            self.assertEqual(resp.status_code, 200)

            data = resp.json()
            self.assertIn("success", data)
            self.assertIn("info", data)
            self.assertTrue(data["success"])

    def test_detach_hardware_endpoint(self):
        """Test POST /detach_hardware returns success."""
        with TestClient(app) as client:
            resources.world.detach_hardware = fake_detach
            resp = client.post("/detach_hardware", json={})
            self.assertEqual(resp.status_code, 200)

            data = resp.json()
            self.assertIn("success", data)
            self.assertIn("info", data)
            self.assertTrue(data["success"])


    def test_reset_endpoint(self):
        """Test POST /reset returns observation."""
        with TestClient(app) as client:
            resources.world.reset = fake_reset
            resp = client.post("/reset", json={})
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertIn("observation", data)

    def test_step_json_endpoint(self):
        """Test POST /step_json returns correctly encoded observation."""
        with TestClient(app) as client:
            resources.world.step = fake_step
            action = np.array([90, 50, 1])
            resp = client.post("/step_json", json={"action": action.tolist()})
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertIn("observation", data)

            # ImageArray is serialized as a dict with shape, dtype, and base64 data
            obs = data["observation"]
            self.assertIsInstance(obs, dict)
            self.assertIn("shape", obs)
            self.assertIn("dtype", obs)
            self.assertIn("data", obs)
            self.assertEqual(obs["shape"], [2, 2, 3])
            self.assertEqual(obs["dtype"], "uint8")

    def test_step_multipart_endpoint(self):
        """Test POST /step_multipart returns multipart response."""
        with TestClient(app) as client:
            resources.world.step = fake_step
            action = np.array([90, 50, 1])
            resp = client.post("/step_multipart", json={"action": action.tolist()})
            self.assertEqual(resp.status_code, 200)
            self.assertTrue(resp.headers["content-type"].startswith("multipart/form-data"))
            self.assertIn(b"application/json", resp.content)

    def test_health_endpoint(self):
        """Test GET /health returns health status."""
        with TestClient(app) as client:
            resp = client.get("/health")
            self.assertEqual(resp.status_code, 200)

            data = resp.json()
            self.assertIn("status", data)
            self.assertEqual(data["status"], "healthy")
            self.assertIn("hardware_attached", data)
            self.assertIsInstance(data["hardware_attached"], bool)

    def test_status_endpoint(self):
        """Test GET /status returns detailed server status."""
        with TestClient(app) as client:
            resp = client.get("/status")
            self.assertEqual(resp.status_code, 200)

            data = resp.json()
            self.assertIn("hardware_attached", data)
            self.assertIn("camera_active", data)
            self.assertIn("robot_connected", data)

            # All should be boolean values
            self.assertIsInstance(data["hardware_attached"], bool)
            self.assertIsInstance(data["camera_active"], bool)
            self.assertIsInstance(data["robot_connected"], bool)


DUMMY_ATTACH_BODY = {"world_config": {"bolt_use_dummy": True, "camera_type": "dummy"}}


class TestWorldServerHardwareCleanup(unittest.TestCase):
    """Regression tests for hardware cleanup on server shutdown."""

    def test_shutdown_detaches_attached_hardware(self):
        """Leaving the lifespan must release hardware that is attached."""
        with TestClient(app) as client:
            world = resources.world
            resp = client.post("/attach_hardware", json=DUMMY_ATTACH_BODY)

            self.assertEqual(resp.status_code, 200)
            self.assertTrue(world._hardware_attached)

        # Exiting the TestClient context runs the lifespan shutdown.
        self.assertFalse(world._hardware_attached)
        self.assertIsNone(world.robot)
        self.assertIsNone(world.camera)

    def test_shutdown_releases_orphaned_hardware(self):
        """Hardware left live by a failed attach must still be released on shutdown.

        The shutdown hook used to be gated on `_hardware_attached`, which a failed
        attach never sets, so orphaned connections survived until the process died.
        """
        with TestClient(app):
            world = resources.world
            world.robot = MagicMock()
            world.camera = MagicMock()
            world._hardware_attached = False  # the state a failed attach leaves behind
            robot, camera = world.robot, world.camera

        robot.disconnect.assert_called_once()
        camera.release.assert_called_once()
        self.assertIsNone(world.robot)
        self.assertIsNone(world.camera)

    def test_failed_attach_reports_503_and_clean_status(self):
        """A failed attach must surface as 503 and leave nothing connected."""
        with TestClient(app) as client:
            with patch.object(CameraService, "setup", side_effect=RuntimeError("Camera 0 could not be opened!")):
                resp = client.post("/attach_hardware", json=DUMMY_ATTACH_BODY)

            self.assertEqual(resp.status_code, 503)

            status = client.get("/status").json()
            self.assertFalse(status["hardware_attached"])
            self.assertFalse(status["camera_active"])
            self.assertFalse(status["robot_connected"])

    def test_detach_after_failed_attach_is_recoverable(self):
        """A failed attach must not wedge the server: detach then attach must work."""
        with TestClient(app) as client:
            world = resources.world

            with patch.object(CameraService, "setup", side_effect=RuntimeError("Camera 0 could not be opened!")):
                client.post("/attach_hardware", json=DUMMY_ATTACH_BODY)

            detach = client.post("/detach_hardware", json={})
            self.assertEqual(detach.status_code, 200)
            self.assertTrue(detach.json()["success"])

            retry = client.post("/attach_hardware", json=DUMMY_ATTACH_BODY)
            self.assertEqual(retry.status_code, 200)
            self.assertTrue(world._hardware_attached)

    def test_timeout_error_reports_503_not_500(self):
        """A hardware timeout must read as a retryable 503, not an opaque 500.

        TimeoutError subclasses OSError, not RuntimeError, so it used to miss the
        hardware handler entirely -- costing a 500 plus two extra tracebacks (the
        catch-all handler, then Starlette re-raising into uvicorn).
        """
        with TestClient(app) as client:
            with patch.object(CameraService, "setup", side_effect=TimeoutError("BLE connect timed out")):
                resp = client.post("/attach_hardware", json=DUMMY_ATTACH_BODY)

            self.assertEqual(resp.status_code, 503)

            body = resp.json()
            self.assertEqual(body["error"], "HardwareError")
            self.assertTrue(body["recoverable"])
            self.assertIn("BLE connect timed out", body["message"])

    def test_runtime_error_still_reports_503(self):
        """The original RuntimeError path must keep its 503 contract."""
        with TestClient(app) as client:
            with patch.object(CameraService, "setup", side_effect=RuntimeError("Camera 2 could not be opened!")):
                resp = client.post("/attach_hardware", json=DUMMY_ATTACH_BODY)

            self.assertEqual(resp.status_code, 503)
            self.assertEqual(resp.json()["error"], "HardwareError")


class TestHardwareErrorRetryPolicy(unittest.TestCase):
    """The status code carries the retry policy: 503 retries, 422 does not.

    The client only retries 502/503/504, so a permanent failure must not answer 503
    or every missing camera costs four Bolt scans and connects before failing.
    """

    def _attach_with(self, error: Exception):
        with TestClient(app) as client:
            with patch.object(CameraService, "setup", side_effect=error):
                return client.post("/attach_hardware", json=DUMMY_ATTACH_BODY)

    def test_permanent_error_is_not_retryable(self):
        """A missing camera answers 422 so the client gives up immediately."""
        resp = self._attach_with(PermanentHardwareError("Camera 2 could not be opened (available indices: [0, 1])"))

        self.assertEqual(resp.status_code, 422)
        body = resp.json()
        self.assertFalse(body["retryable"])
        self.assertIn("available indices", body["message"])

    def test_transient_error_is_retryable(self):
        """A sleeping Bolt answers 503 so the client tries again."""
        resp = self._attach_with(TransientHardwareError("Sphero 'BP-D217' not found, available Toys: []"))

        self.assertEqual(resp.status_code, 503)
        self.assertTrue(resp.json()["retryable"])

    def test_unclassified_runtime_error_still_retries(self):
        """Anything not yet classified keeps the old retryable 503 behaviour."""
        resp = self._attach_with(RuntimeError("something unclassified"))

        self.assertEqual(resp.status_code, 503)
        self.assertTrue(resp.json()["retryable"])

    def test_client_retry_policy_agrees_with_the_status_codes(self):
        """Guard the contract across the package boundary."""
        from rlive_env.world_client import WorldInterface

        self.assertTrue(WorldInterface.is_retryable_status(503))
        self.assertFalse(WorldInterface.is_retryable_status(422))

