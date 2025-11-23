import unittest
import numpy as np
from rlive_common.core.response import StepResponseJSON, StepResponseMultipart


class TestRequests(unittest.TestCase):
    def test_encoding_and_decoding_with_json(self):
        img = np.random.randint(0, 255, size=(2, 4, 3), dtype=np.uint8)

        response = StepResponseJSON(observation=img, truncated=False, info={"meta": "json"})

        dumped = response.model_dump_json()
        loaded = StepResponseJSON.model_validate_json(dumped)

        self.assertTrue(np.array_equal(img, loaded.observation))

    def test_encoding_and_decoding_with_multipart(self):
        # RGB
        img_rgb = np.random.randint(0, 255, size=(2, 4, 3), dtype=np.uint8)

        response = StepResponseMultipart(observation=img_rgb, truncated=False, info={"meta": "multipart"})
        body, content_type = response.encode()
        loaded = StepResponseMultipart.decode(body, content_type)

        self.assertEqual(loaded.info["meta"], "multipart")
        self.assertTrue(np.array_equal(img_rgb, loaded.observation))

        # Gray
        img_gray = np.random.randint(0, 255, size=(4, 3), dtype=np.uint8)

        response = StepResponseMultipart(observation=img_gray, truncated=False, info={"meta": "multipart"})
        body, content_type = response.encode()
        loaded = StepResponseMultipart.decode(body, content_type)

        self.assertEqual(loaded.info["meta"], "multipart")
        self.assertTrue(np.array_equal(img_gray, loaded.observation))
