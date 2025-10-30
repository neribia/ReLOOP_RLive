from abc import ABC, abstractmethod
import numpy as np


class BaseCamera(ABC):
    @abstractmethod
    def __init__(self, width, height):
        ...

    @abstractmethod
    def setup(self):
        ...

    @abstractmethod
    def get_image(self) -> np.ndarray:
        ...

    @abstractmethod
    def release(self):
        ...
