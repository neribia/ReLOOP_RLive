import unittest
from rlive_common.core.pydantic_types import (
    _encode_image,
    _decode_image,
    _ndarray_serializer,
    _ndarray_before_validator,
)
import numpy as np


class TestImageEncoding(unittest.TestCase):
    def test_image_encoding_decoding(self):
        original_array = np.random.randint(0, 255, size=(2, 4, 3), dtype=np.uint8)
        encoded = _encode_image(original_array)
        decoded_array = _decode_image(encoded)
        self.assertTrue(np.array_equal(original_array, decoded_array))

    def test_numpy_encoding_decoding(self):
        original_array = np.random.randint(0, 255, size=(2, 4, 3), dtype=np.uint8)
        encoded = _ndarray_serializer(original_array)
        decoded_array = _ndarray_before_validator(encoded)
        self.assertTrue(np.array_equal(original_array, decoded_array))
