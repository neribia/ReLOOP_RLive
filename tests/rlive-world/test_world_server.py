import unittest
import numpy as np
from fastapi.testclient import TestClient

from rlive_world.world_server import app, resources  # adjust import to where your app lives


def fake_reset(req):
    from rlive_common.core.response import ResetResponse
    return ResetResponse(observation=np.zeros((2, 2, 3), dtype=np.uint8) , truncated=False, info={})


def fake_step(req):
    from rlive_common.core.response import BaseResponse
    return BaseResponse(observation=np.zeros((2, 2, 3), dtype=np.uint8), truncated=False, info={})

def fake_attach(req):
    from rlive_common.core.response import AttachHardwareResponse
    return AttachHardwareResponse(success=True, info={"message": "Detached OK"})

def fake_detach(req):
    from rlive_common.core.response import DetachHardwareResponse
    return DetachHardwareResponse(success=True, info={"message": "Detached OK"})


class TestWorldAPI(unittest.TestCase):
    def test_attach_hardware_endpoint(self):
        with TestClient(app) as client:
            resources.world.attach_hardware = fake_attach
            resp = client.post("/attach_hardware", json={})
            self.assertEqual(resp.status_code, 200)

            data = resp.json()
            self.assertIn("success", data)
            self.assertIn("info", data)
            self.assertTrue(data["success"])

    def test_detach_hardware_endpoint(self):
        with TestClient(app) as client:
            resources.world.detach_hardware = fake_detach
            resp = client.post("/detach_hardware", json={})
            self.assertEqual(resp.status_code, 200)

            data = resp.json()
            self.assertIn("success", data)
            self.assertIn("info", data)
            self.assertTrue(data["success"])


    def test_reset_endpoint(self):
        with TestClient(app) as client:
            resources.world.reset = fake_reset
            resp = client.post("/reset", json={})
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertIn("observation", data)

    def test_step_json_endpoint(self):
        with TestClient(app) as client:
            resources.world.step = fake_step
            resp = client.post("/step_json", json={"action": 5})
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertIn("observation", data)
            self.assertEqual(np.uint8, np.array(data["observation"]).dtype)
            self.assertEqual((2, 2, 3), np.array(data["observation"]).shape)

    def test_step_multipart_endpoint(self):
        with TestClient(app) as client:
            resources.world.step = fake_step
            resp = client.post("/step_multipart", json={"action": 5})
            self.assertEqual(resp.status_code, 200)
            self.assertTrue(resp.headers["content-type"].startswith("multipart/form-data"))
            self.assertIn(b"application/json", resp.content)
