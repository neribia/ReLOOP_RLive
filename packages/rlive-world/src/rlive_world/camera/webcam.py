import cv2 as cv

from rlive_world.camera import BaseCamera


class Webcam(BaseCamera):
    def __init__(self, cam_index: int = 0):
        self.cam_index = cam_index
        self.cam = None

    def setup(self):
        self.cam = cv.VideoCapture(self.cam_index)
        if not self.cam.isOpened():
            raise RuntimeError(f"Kamera {self.cam_index} konnte nicht geöffnet werden")

    def release(self):
        if self.cam:
            self.cam.release()
            self.cam = None

    def get_image(self):
        ret, frame = self.cam.read()
        if not ret:
            return {"info": "Kein Frame erhalten"}
        # OpenCV liefert BGR → GUI erwartet RGB/PIL-kompatibel
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        return frame_rgb
