from abc import ABC, abstractmethod
import numpy as np


class BaseCamera(ABC):
    def __init__(self, width, height):
        self._width = width
        self._height = height

    @abstractmethod
    def setup(self) -> None: ...

    @abstractmethod
    def get_image(self) -> np.ndarray: ...

    @abstractmethod
    def release(self): ...
