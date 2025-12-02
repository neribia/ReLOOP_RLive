"""Tests for the World API server endpoints."""

import unittest
import numpy as np
from fastapi.testclient import TestClient

from rlive_world.world_server import app, resources
from rlive_common.core.response import (
    ResetResponse,
    BaseResponse,
    AttachHardwareResponse,
    DetachHardwareResponse,
)
from rlive_common.core.types import _decode_image


def fake_reset(req):
    """Fake reset handler for testing."""
    return ResetResponse(observation=np.zeros((2, 2, 3), dtype=np.uint8), truncated=False, info={})


def fake_step(req):
    """Fake step handler for testing."""
    return BaseResponse(observation=np.zeros((2, 2, 3), dtype=np.uint8), truncated=False, info={})


def fake_attach(req):
    """Fake attach hardware handler for testing."""
    return AttachHardwareResponse(success=True, info={"message": "Attached OK"})


def fake_detach(req):
    """Fake detach hardware handler for testing."""
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
            resp = client.post("/step_json", json={"action": 5})
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertIn("observation", data)
            # Decode the observation from the encoded format
            obs = _decode_image(data["observation"])
            self.assertEqual(np.uint8, obs.dtype)
            self.assertEqual((2, 2, 3), obs.shape)

    def test_step_multipart_endpoint(self):
        """Test POST /step_multipart returns multipart response."""
        with TestClient(app) as client:
            resources.world.step = fake_step
            resp = client.post("/step_multipart", json={"action": 5})
            self.assertEqual(resp.status_code, 200)
            self.assertTrue(resp.headers["content-type"].startswith("multipart/form-data"))
            self.assertIn(b"application/json", resp.content)
