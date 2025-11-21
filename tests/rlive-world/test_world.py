import unittest
import numpy as np
import asyncio

from rlive_common.core.response import Response
from rlive_common.core.request import StepRequest, ResetRequest
from rlive_world.world import World


class ToyWorld(World):
     def _make_observation(self) -> np.ndarray:
        return np.zeros((100, 100), dtype=np.uint32)  # (480, 640))


class TestWorld(unittest.TestCase):
    def test_make_observation(self):
        world = World()
        obs = world._make_observation()

        self.assertIsInstance(obs, np.ndarray)
        self.assertTrue(obs.dtype == np.float32)
        self.assertTrue(obs.ndim in (1, 2, 3)) # FIXME: remove 1 when camera is implemented in this function

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
        self.assertTrue(img.dtype == np.uint8)
        self.assertTrue(img.ndim in (2, 3))







