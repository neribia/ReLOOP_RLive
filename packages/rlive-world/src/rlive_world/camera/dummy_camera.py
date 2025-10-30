import numpy as np

from rlive_world.camera import BaseCamera


class DummyCamera(BaseCamera):
    def __init__(self, width: int = 640, height: int = 480, channels: int = 3):
        self.width = width
        self.height = height
        self.channels = channels

    def setup(self):
        pass

    def release(self):
        pass

    def get_image(self):
        return (np.random.rand(self.height, self.width, self.channels) * 255).astype(np.uint8)
