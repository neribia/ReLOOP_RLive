import numpy as np

from rlive_world.camera.base_camera import BaseCamera


class DummyCamera(BaseCamera):
    def __init__(self, width: int = 640, height: int = 480, channels: int = 3):
        super().__init__(width=width, height=height)
        self._channels = channels

    def setup(self):
        pass

    def release(self):
        pass

    def get_image(self):
        return np.random.randint(
            0, 256, (self._height, self._width, self._channels), dtype=np.uint8
        )
