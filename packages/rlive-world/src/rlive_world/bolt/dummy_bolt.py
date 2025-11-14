from rlive_world.bolt.base_robot import BaseRobot

from rlive_common.utils import get_logger

logger = get_logger(__name__)


class DummyBolt(BaseRobot):
    def __init__(self):
        logger.debug("Initializing DummyBolt class")

    def connect(self):
        logger.debug("Connect DummyBolt class")

    def disconnect(self):
        logger.debug("Disconnect DummyBolt class")

    def move(self, *args, **kwargs):
        logger.debug("Move DummyBolt class")
