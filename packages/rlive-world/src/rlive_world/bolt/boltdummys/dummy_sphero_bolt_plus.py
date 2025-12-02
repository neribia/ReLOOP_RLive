"""Dummy implementation of SpheroBoltPlus robot for testing."""

from rlive_world.bolt.base_robot import BaseRobot
from rlive_common.utils import get_logger

logger = get_logger(__name__)


class DummySpheroBoltPlus(BaseRobot):
    """Dummy SpheroBoltPlus robot for testing purposes."""

    def __init__(self) -> None:
        """Initialize the dummy robot."""
        self.name = "DummyBoltPlus"
        self.connected = False
        self.heading = 0

    def connect(self, bolt_name: str = "DummyBoltPlus") -> None:
        """Connect to the dummy robot.

        Args:
            bolt_name: The name of the robot to connect to.
        """
        logger.debug(f"[DUMMY] Connecting to {bolt_name}")
        self.name = bolt_name
        self.connected = True

    def disconnect(self) -> None:
        """Disconnect from the dummy robot."""
        logger.debug("[DUMMY] Disconnecting")
        self.connected = False

    def move(self, heading: int = 0, speed: int = 100, duration: float = 1.0) -> None:
        """Move the dummy robot.

        Args:
            heading: Direction to move (0-360 degrees).
            speed: Speed of movement (-255 to 255).
            duration: Duration of movement in seconds.
        """
        logger.debug(f"[DUMMY] Move: heading={heading}, speed={speed}, duration={duration}")
        self.heading = (self.heading + heading + 360) % 360
