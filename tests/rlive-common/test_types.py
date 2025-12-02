import unittest
import base64
import json
import numpy as np

from rlive_common.core.types import (
    _encode_image,
    _decode_image,
    _ndarray_serializer,
    _ndarray_before_validator,
)


class TestImageEncodingDecoding(unittest.TestCase):
    def test_encode_image_returns_dict(self):
        array = np.zeros((10, 10, 3), dtype=np.uint8)
        result = _encode_image(array)

        self.assertIsInstance(result, dict)
        self.assertIn("shape", result)
        self.assertIn("dtype", result)
        self.assertIn("data", result)

    def test_encode_image_shape(self):
        array = np.zeros((480, 640, 3), dtype=np.uint8)
        result = _encode_image(array)

        self.assertEqual(result["shape"], (480, 640, 3))

    def test_encode_image_dtype(self):
        array = np.zeros((10, 10), dtype=np.uint8)
        result = _encode_image(array)

        self.assertEqual(result["dtype"], "uint8")

    def test_encode_image_data_is_base64(self):
        array = np.ones((2, 2), dtype=np.uint8)
        result = _encode_image(array)

        # Should be valid base64 string
        decoded = base64.b64decode(result["data"])
        self.assertEqual(len(decoded), 4)  # 2x2 bytes

    def test_decode_image_from_dict(self):
        original = np.array([[1, 2], [3, 4]], dtype=np.uint8)
        encoded = _encode_image(original)
        decoded = _decode_image(encoded)

        np.testing.assert_array_equal(original, decoded)

    def test_decode_image_from_ndarray(self):
        original = np.array([1, 2, 3], dtype=np.uint8)
        result = _decode_image(original)

        self.assertIs(result, original)  # Should return same object

    def test_decode_image_invalid_type(self):
        with self.assertRaises(TypeError):
            _decode_image("not a valid payload")

    def test_decode_image_invalid_dict(self):
        with self.assertRaises(TypeError):
            _decode_image({"incomplete": "dict"})

    def test_roundtrip_rgb_image(self):
        original = np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8)
        encoded = _encode_image(original)
        decoded = _decode_image(encoded)

        np.testing.assert_array_equal(original, decoded)

    def test_roundtrip_grayscale_image(self):
        original = np.random.randint(0, 255, size=(50, 50), dtype=np.uint8)
        encoded = _encode_image(original)
        decoded = _decode_image(encoded)

        np.testing.assert_array_equal(original, decoded)


class TestNdarrayValidatorSerializer(unittest.TestCase):
    def test_serializer_returns_list(self):
        array = np.array([1, 2, 3])
        result = _ndarray_serializer(array)

        self.assertIsInstance(result, list)
        self.assertEqual(result, [1, 2, 3])

    def test_serializer_2d_array(self):
        array = np.array([[1, 2], [3, 4]])
        result = _ndarray_serializer(array)

        self.assertEqual(result, [[1, 2], [3, 4]])

    def test_validator_from_ndarray(self):
        array = np.array([1, 2, 3])
        result = _ndarray_before_validator(array)

        self.assertIs(result, array)

    def test_validator_from_list(self):
        data = [1, 2, 3]
        result = _ndarray_before_validator(data)

        self.assertIsInstance(result, np.ndarray)
        np.testing.assert_array_equal(result, np.array([1, 2, 3]))

    def test_validator_from_tuple(self):
        data = (1, 2, 3)
        result = _ndarray_before_validator(data)

        self.assertIsInstance(result, np.ndarray)
        np.testing.assert_array_equal(result, np.array([1, 2, 3]))

    def test_validator_from_json_string(self):
        data = json.dumps([1, 2, 3])
        result = _ndarray_before_validator(data)

        self.assertIsInstance(result, np.ndarray)
        np.testing.assert_array_equal(result, np.array([1, 2, 3]))

    def test_validator_invalid_type(self):
        with self.assertRaises(TypeError):
            _ndarray_before_validator(12345)  # Not a valid type

    def test_validator_nested_list(self):
        data = [[1, 2], [3, 4]]
        result = _ndarray_before_validator(data)

        self.assertIsInstance(result, np.ndarray)
        np.testing.assert_array_equal(result, np.array([[1, 2], [3, 4]]))


if __name__ == "__main__":
    unittest.main()
