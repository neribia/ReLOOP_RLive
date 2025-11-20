import unittest
import numpy as np
from fastapi.testclient import TestClient

from rlive_world.world_server import app, world  # adjust import to where your app lives


async def fake_reset(req):
    from rlive_common.core.response import ResetResponse
    return ResetResponse(observation=np.array([1, 2, 3]), truncated=False, info={})


async def fake_step(req):
    from rlive_common.core.response import Response
    return Response(observation=np.array([1, 2, 3]), truncated=False, info={}), np.zeros((2, 2, 3), dtype=np.uint8)


class TestWorldAPI(unittest.TestCase):

    def setUp(self):
        # Create a test client for the FastAPI app
        self.client = TestClient(app)

    def test_reset_endpoint(self):
        # Patch world.reset to return a ResetResponse
        world.reset = fake_reset

        resp = self.client.post("/reset", json={})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("observation", data)

    def test_step_json_endpoint(self):
        world.step = fake_step

        resp = self.client.post("/step_json", json={"action": 5})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("image", data)
        self.assertIn("observation", data)

    def test_step_multipart_endpoint(self):
        world.step = fake_step

        resp = self.client.post("/step_multipart", json={"action": 5})
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.headers["content-type"].startswith("multipart/mixed"))
        self.assertIn(b"application/json", resp.content)
