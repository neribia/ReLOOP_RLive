import unittest
import numpy as np
from rlive_common.core.response import Response, AttachHardwareResponse, DetachHardwareResponse
from rlive_common.core.request import StepRequest, ResetRequest, AttachHardwareRequest, DetachHardwareRequest
from rlive_world.world import World


class ToyWorld(World):
    def _make_observation(self) -> np.ndarray:
        return np.zeros((100, 100), dtype=np.uint32)  # (480, 640))


class TestWorld(unittest.TestCase):
    def test_attach_hardware(self):
        world = ToyWorld()
        request = AttachHardwareRequest()

        result = world.attach_hardware(request)

        self.assertIsInstance(result, AttachHardwareResponse)
        self.assertIn("status", result.info)
        self.assertIn("msg", result.info)
        self.assertTrue(result.success)

    def test_detach_hardware(self):
        world = ToyWorld()
        request = DetachHardwareRequest()

        result = world.detach_hardware(request)

        self.assertIsInstance(result, DetachHardwareResponse)
        self.assertIn("status", result.info)
        self.assertIn("msg", result.info)
        self.assertTrue(result.success)

    def test_make_observation(self):
        world = World()
        obs = world._make_observation()

        self.assertIsInstance(obs, np.ndarray)
        self.assertEqual(obs.dtype, np.float32)
        self.assertIn(obs.ndim, (1, 2, 3))  # FIXME: remove 1 when camera is implemented in this function

    def test_reset(self):
        world = ToyWorld()
        request = ResetRequest()
        result = world.reset(request)

        self.assertIsInstance(result, Response)

    def test_step(self):
        world = ToyWorld()
        request = StepRequest(action=5)
        result, img = world.step(request)

        self.assertIsInstance(result, Response)
        self.assertIsInstance(img, np.ndarray)
        self.assertEqual(img.dtype, np.uint8)
        self.assertIn(img.ndim, (2, 3))
