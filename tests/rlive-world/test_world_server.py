"""Tests for the World API server endpoints."""

import unittest
import numpy as np
from fastapi.testclient import TestClient

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

