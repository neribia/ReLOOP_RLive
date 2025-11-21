import unittest
import numpy as np
from fastapi.testclient import TestClient

from rlive_world.world_server import app, resources  # adjust import to where your app lives


def fake_reset(req):
    from rlive_common.core.response import ResetResponse
    return ResetResponse(observation=np.array([1, 2, 3]), truncated=False, info={})


def fake_step(req):
    from rlive_common.core.response import Response
    return Response(observation=np.array([1, 2, 3]), truncated=False, info={}), np.zeros((2, 2, 3), dtype=np.uint8)


class TestWorldAPI(unittest.TestCase):
    def test_reset_endpoint(self):
        resources.world.reset = fake_reset
        with TestClient(app) as client:

            resp = client.post("/reset", json={})
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertIn("observation", data)

    def test_step_json_endpoint(self):
        resources.world.step = fake_step
        with TestClient(app) as client:
            resp = client.post("/step_json", json={"action": 5})
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertIn("image", data)
            self.assertIn("observation", data)

    def test_step_multipart_endpoint(self):
        resources.world.step = fake_step
        with TestClient(app) as client:
            resp = client.post("/step_multipart", json={"action": 5})
            self.assertEqual(resp.status_code, 200)
            self.assertTrue(resp.headers["content-type"].startswith("multipart/form-data"))
            self.assertIn(b"application/json", resp.content)
