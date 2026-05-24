import cv2 as cv
import numpy as np
import platform
import time

from rlive_world.camera.base_camera import BaseCamera


class Webcam(BaseCamera):
    def __init__(self, cam_index: int = 0, width: int = 640, height: int = 480):
        super().__init__(width, height)
        self.cam_index = cam_index
        self.cam: cv.VideoCapture | None = None

    def setup(self) -> None:
        """Initialisiert die Webcam."""
        # Detect platform and use appropriate backend
        system = platform.system()
        if system == "Windows":
            backend = cv.CAP_DSHOW
        elif system == "Linux":
            backend = cv.CAP_V4L2
        else:
            backend = None

        self.cam = cv.VideoCapture(self.cam_index, backend) if backend else cv.VideoCapture(self.cam_index)
        if not self.cam.isOpened():
            raise RuntimeError(f"Camera {self.cam_index} could not be opened!")

        self.cam.set(cv.CAP_PROP_FRAME_WIDTH, self._width)
        self.cam.set(cv.CAP_PROP_FRAME_HEIGHT, self._height)

        # Enable autofocus
        self.cam.set(cv.CAP_PROP_AUTOFOCUS, 1)

        # Warm up: let autofocus converge BEFORE locking it.
        # 2 s + 30 frames is enough for most USB webcams to find focus.
        time.sleep(2.0)
        for _ in range(30):  # drain stale frames while autofocus runs
            self.cam.grab()

        # Disable autofocus
        self.cam.set(cv.CAP_PROP_AUTOFOCUS, 0)
        # Optionally set a fixed focus value (0.0 = infinity, adjust as needed)
        #self.cam.set(cv.CAP_PROP_FOCUS, 0)

    def release(self) -> None:
        """Releases the camera."""
        if self.cam:
            self.cam.release()
            self.cam = None

    def get_image(self) -> np.ndarray:
        """Returns the most recent frame from the camera.

        OpenCV's VideoCapture maintains an internal buffer (typically 4-5 frames on
        Windows/CAP_DSHOW). After the robot has moved we must flush that buffer so
        we capture the *current* scene, not a stale frame queued before the action
        completed. Calling grab() repeatedly drains the queue without decoding
        the frames, which is cheap.
        """
        # Flush the internal frame buffer
        flush_count = int(max(self.cam.get(cv.CAP_PROP_BUFFERSIZE), 1))  # at least 1 grab even if property returns 0
        for _ in range(flush_count):
            self.cam.read()

        ret, frame = self.cam.read()
        if not ret:
            raise RuntimeError("Could not get image from camera.")
        return cv.cvtColor(frame, cv.COLOR_BGR2RGB)
