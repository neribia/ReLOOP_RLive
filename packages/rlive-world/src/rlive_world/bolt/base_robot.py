from abc import ABC, abstractmethod


class BaseRobot(ABC):
    @abstractmethod
    def connect(self):
        pass

    @abstractmethod
    def disconnect(self):
        pass

    @abstractmethod
    def move(self):
        pass
