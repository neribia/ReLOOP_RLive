import cv2 as cv

from rlive_world.camera import BaseCamera


class Webcam(BaseCamera):
    def __init__(self, cam_index: int = 0,  width: int = 640, height: int = 480):
        self.cam_index = cam_index
        self.cam: cv.VideoCapture | None = None
        self.width = width
        self.height = height

    def setup(self):
        """
        Initialisiert die Webcam.
        """
        self.cam = cv.VideoCapture(self.cam_index)
        if not self.cam.isOpened():
            raise RuntimeError(f"Camera {self.cam_index} could not be opened!")

        self.cam.set(cv.CAP_PROP_FRAME_WIDTH, self.width)
        self.cam.set(cv.CAP_PROP_FRAME_HEIGHT, self.height)

    def release(self) -> None:
        """
        Releases the camera.
        :return: None
        """
        if self.cam:
            self.cam.release()
            self.cam = None

    def get_image(self):
        ret, frame = self.cam.read()
        if not ret:
            return None
        return cv.cvtColor(frame, cv.COLOR_BGR2RGB)
