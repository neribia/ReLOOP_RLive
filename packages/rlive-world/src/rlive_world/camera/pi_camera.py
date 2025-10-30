from rlive_world.camera import BaseCamera


class PICamera(BaseCamera):
    def __init__(self):
        raise NotImplementedError

    def setup(self):
        raise NotImplementedError

    def release(self):
        raise NotImplementedError

    def get_image(self):
        raise NotImplementedError
