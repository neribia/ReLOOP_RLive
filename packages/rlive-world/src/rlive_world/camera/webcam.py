import cv2 as cv
import numpy as np

from rlive_world.camera.base_camera import BaseCamera


class Webcam(BaseCamera):
    def __init__(self, cam_index: int = 0, width: int = 640, height: int = 480):
        super().__init__(width, height)
        self.cam_index = cam_index
        self.cam: cv.VideoCapture | None = None

    def setup(self) -> None:
        """Initialisiert die Webcam."""
        self.cam = cv.VideoCapture(self.cam_index)
        if not self.cam.isOpened():
            raise RuntimeError(f"Camera {self.cam_index} could not be opened!")

        self.cam.set(cv.CAP_PROP_FRAME_WIDTH, self._width)
        self.cam.set(cv.CAP_PROP_FRAME_HEIGHT, self._height)

    def release(self) -> None:
        """Releases the camera."""
        if self.cam:
            self.cam.release()
            self.cam = None

    def get_image(self) -> np.ndarray:
        """Returns the image of the camera."""
        ret, frame = self.cam.read()
        if not ret:
            raise RuntimeError("Could not get image from camera.")
        return cv.cvtColor(frame, cv.COLOR_BGR2RGB)
