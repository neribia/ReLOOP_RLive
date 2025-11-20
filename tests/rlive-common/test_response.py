import unittest
import pytest
import numpy as np
import cv2 as cv

from rlive_common.core.response import Response, ResetResponse, StepResponseJSON, StepResponseMultipart


class TestRequests(unittest.TestCase):
    def test_encoding_and_decoding_with_json(self):
        obs = np.array([1, 2, 3])
        img = np.random.randint(0, 255, size=(2, 4, 3), dtype=np.uint8)

        response = StepResponseJSON(observation=obs, truncated=False, info={"meta": "json"}, image=img)

        dumped = response.model_dump_json()
        loaded = StepResponseJSON.model_validate_json(dumped)

        self.assertTrue(np.array_equal(obs, loaded.observation))
        self.assertTrue(np.array_equal(img, loaded.image))

    def test_encoding_and_decoding_with_json_no_image(self):
        obs = np.array([1, 2, 3])

        response = StepResponseJSON(observation=obs, truncated=False, info={"meta": "json"})
        dumped = response.model_dump_json()
        loaded = StepResponseJSON.model_validate_json(dumped)

        self.assertIsNone(loaded.image)

    def test_encoding_and_decoding_with_multipart(self):
        obs = np.array([1, 2, 3])
        img = np.random.randint(0, 255, size=(2, 4, 3), dtype=np.uint8)

        response = StepResponseMultipart(observation=obs, truncated=False, info={"meta": "multipart"}, image=img)
        body, content_type = response.encode()
        loaded = StepResponseMultipart.decode(body, content_type)

        self.assertTrue(np.array_equal(obs, loaded.observation))
        self.assertEqual(loaded.info["meta"], "multipart")
        self.assertTrue(np.array_equal(img, loaded.image))

    def test_encoding_and_decoding_with_multipart_no_image(self):
        obs = np.array([1, 2, 3])

        response = StepResponseMultipart(observation=obs, truncated=False, info={"meta": "multipart"})
        body, content_type = response.encode()
        loaded = StepResponseMultipart.decode(body, content_type)

        self.assertTrue(np.array_equal(obs, loaded.observation))
        self.assertEqual(loaded.info["meta"], "multipart")
        self.assertIsNone(loaded.image)
