import cv2 as cv
import numpy as np
import platform
import time

from rlive_world.camera.base_camera import BaseCamera
from rlive_world.errors import PermanentHardwareError


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
            self.cam.release()
            self.cam = None
            available = self._available_indices(backend)
            detail = f"available indices: {available}" if available else "no capture devices found"
            raise PermanentHardwareError(f"Camera {self.cam_index} could not be opened ({detail})")

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

    @staticmethod
    def _available_indices(backend: int | None, max_index: int = 5) -> list[int]:
        """Probe which capture indices can actually be opened.

        Only runs on the failure path, to turn "Camera 2 could not be opened" into
        something the caller can act on. OpenCV emits its own warning for each index
        that is not there; that noise is left alone rather than silenced, because the
        log-level API moved between OpenCV versions and this must never be the reason
        an attach fails. Any error here degrades to "no list available".
        """
        found: list[int] = []
        try:
            for index in range(max_index):
                probe = cv.VideoCapture(index, backend) if backend else cv.VideoCapture(index)
                try:
                    if probe.isOpened():
                        found.append(index)
                finally:
                    probe.release()
        except Exception:
            return found
        return found

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
