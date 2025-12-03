import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from rlive_world.camera.base_camera import BaseCamera
from rlive_world.camera.dummy_camera import DummyCamera
from rlive_world.camera.camera_factory import CameraFactory, CameraConfig
from rlive_world.camera.camera_service import CameraService


class TestDummyCamera(unittest.TestCase):
    def test_init_default_dimensions(self):
        cam = DummyCamera()
        self.assertEqual(cam._width, 640)
        self.assertEqual(cam._height, 480)
        self.assertEqual(cam._channels, 3)

    def test_init_custom_dimensions(self):
        cam = DummyCamera(width=320, height=240, channels=1)
        self.assertEqual(cam._width, 320)
        self.assertEqual(cam._height, 240)
        self.assertEqual(cam._channels, 1)

    def test_setup_does_nothing(self):
        cam = DummyCamera()
        cam.setup()  # Should not raise

    def test_release_does_nothing(self):
        cam = DummyCamera()
        cam.release()  # Should not raise

    def test_get_image_returns_ndarray(self):
        cam = DummyCamera(width=640, height=480, channels=3)
        img = cam.get_image()

        self.assertIsInstance(img, np.ndarray)
        self.assertEqual(img.shape, (480, 640, 3))
        self.assertEqual(img.dtype, np.uint8)

    def test_get_image_grayscale(self):
        cam = DummyCamera(width=320, height=240, channels=1)
        img = cam.get_image()

        self.assertEqual(img.shape, (240, 320, 1))
        self.assertEqual(img.dtype, np.uint8)

    def test_get_image_random_values(self):
        cam = DummyCamera(width=100, height=100)
        img1 = cam.get_image()
        img2 = cam.get_image()

        # Random images should (almost certainly) be different
        self.assertFalse(np.array_equal(img1, img2))


class TestCameraConfig(unittest.TestCase):
    def test_default_values(self):
        cfg = CameraConfig()
        self.assertEqual(cfg.type, "dummy")
        self.assertEqual(cfg.id, 0)
        self.assertEqual(cfg.width, 640)
        self.assertEqual(cfg.height, 480)

    def test_custom_values(self):
        cfg = CameraConfig(type="webcam", id=1, width=1280, height=720)
        self.assertEqual(cfg.type, "webcam")
        self.assertEqual(cfg.id, 1)
        self.assertEqual(cfg.width, 1280)
        self.assertEqual(cfg.height, 720)

    def test_frozen_config(self):
        cfg = CameraConfig()
        with self.assertRaises(Exception):  # FrozenInstanceError
            cfg.type = "webcam"


class TestCameraFactory(unittest.TestCase):
    def test_build_dummy_camera(self):
        factory = CameraFactory()
        cfg = CameraConfig(type="dummy", width=320, height=240)
        cam = factory.build(cfg)

        self.assertIsInstance(cam, DummyCamera)
        self.assertEqual(cam._width, 320)
        self.assertEqual(cam._height, 240)

    def test_build_unknown_type_falls_back_to_dummy(self):
        factory = CameraFactory()
        cfg = CameraConfig(type="unknown_camera", width=640, height=480)
        cam = factory.build(cfg)

        self.assertIsInstance(cam, DummyCamera)

    def test_build_with_whitespace_type(self):
        factory = CameraFactory()
        cfg = CameraConfig(type="  dummy  ", width=640, height=480)
        cam = factory.build(cfg)

        self.assertIsInstance(cam, DummyCamera)

    def test_build_with_uppercase_type(self):
        factory = CameraFactory()
        cfg = CameraConfig(type="DUMMY", width=640, height=480)
        cam = factory.build(cfg)

        self.assertIsInstance(cam, DummyCamera)

    @patch("rlive_world.camera.webcam.Webcam")
    def test_build_webcam_fails_falls_back_to_dummy(self, MockWebcam):
        MockWebcam.side_effect = RuntimeError("No webcam available")

        factory = CameraFactory()
        cfg = CameraConfig(type="webcam", id=0, width=640, height=480)
        cam = factory.build(cfg)

        # Should fall back to dummy camera
        self.assertIsInstance(cam, DummyCamera)

    def test_build_picam_falls_back_to_dummy(self):
        # PICamera raises NotImplementedError
        factory = CameraFactory()
        cfg = CameraConfig(type="picam", width=640, height=480)
        cam = factory.build(cfg)

        # Should fall back to dummy camera
        self.assertIsInstance(cam, DummyCamera)


class TestCameraService(unittest.TestCase):
    def setUp(self):
        self.mock_factory = MagicMock(spec=CameraFactory)
        self.mock_camera = MagicMock(spec=BaseCamera)
        self.mock_factory.build.return_value = self.mock_camera
        self.cfg = CameraConfig(type="dummy", width=640, height=480)

    def test_init(self):
        service = CameraService(config=self.cfg, factory=self.mock_factory)
        self.assertEqual(service._cfg, self.cfg)
        self.assertIsNone(service._cam)

    def test_setup_builds_and_starts_camera(self):
        service = CameraService(config=self.cfg, factory=self.mock_factory)
        service.setup()

        self.mock_factory.build.assert_called_once_with(self.cfg)
        self.mock_camera.setup.assert_called_once()
        self.assertEqual(service._cam, self.mock_camera)

    def test_setup_idempotent(self):
        service = CameraService(config=self.cfg, factory=self.mock_factory)
        service.setup()
        service.setup()  # Second call should be ignored

        self.mock_factory.build.assert_called_once()

    def test_get_frame_returns_image(self):
        service = CameraService(config=self.cfg, factory=self.mock_factory)
        service.setup()

        expected_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.mock_camera.get_image.return_value = expected_frame

        frame = service.get_image()

        self.mock_camera.get_image.assert_called_once()
        np.testing.assert_array_equal(frame, expected_frame)

    def test_get_frame_without_setup_returns_none(self):
        service = CameraService(config=self.cfg, factory=self.mock_factory)
        frame = service.get_image()

        self.assertIsNone(frame)

    def test_release_stops_camera(self):
        service = CameraService(config=self.cfg, factory=self.mock_factory)
        service.setup()
        service.release()

        self.mock_camera.release.assert_called_once()
        self.assertIsNone(service._cam)

    def test_release_without_setup_does_nothing(self):
        service = CameraService(config=self.cfg, factory=self.mock_factory)
        service.release()  # Should not raise

    def test_release_handles_exception(self):
        service = CameraService(config=self.cfg, factory=self.mock_factory)
        service.setup()
        self.mock_camera.release.side_effect = RuntimeError("Camera error")

        service.release()  # Should not raise

        self.assertIsNone(service._cam)
