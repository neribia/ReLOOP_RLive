import unittest
import numpy as np
import asyncio

from rlive_common.core.response import Response, StepResponseJSON
from rlive_common.core.request import StepRequest
from rlive_world.world import World


class ToyWorld(World):

    async def _make_observation(self) -> np.ndarray:
        await asyncio.sleep(0)  # yield control; stands in for real async IO
        return np.zeros((100, 100), dtype=np.float32)  # (480, 640))



class TestCommunications(unittest.TestCase):

    def test_step(self):
        world = ToyWorld()
        request = StepRequest(action=5)
        result, img = asyncio.run(world.step(request))

        self.assertIsInstance(result, Response)
        self.assertTupleEqual(img.shape, (2, 4, 3))

    def test_receive_message(self):
        # Placeholder for testing receive_message function
        self.assertTrue(True)


class TestRequests(unittest.TestCase):

    def test_encoding_and_decoding(self):

        img = np.random.randint(0, 255, size=(2, 4, 3), dtype=np.uint8)

        model = StepResponseJSON(image=img)

        data = model.encode()
