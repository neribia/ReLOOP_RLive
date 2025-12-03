import atexit
import signal
import weakref
import time
from typing import Any

from sphero_unsw.sphero_edu import SpheroEduAPI
from sphero_unsw.toy.boltplus import BOLTPLUS

from rlive_world.bolt.base_robot import BaseRobot
from rlive_world.bolt.sphero_finder import SpheroFinder
from rlive_world.config import config as cfg
from rlive_common.utils import get_logger

logger = get_logger(__name__)


class SpheroBoltPlus(BaseRobot):
    """Simple and testable robot class.
    Optional arguments allow injecting mocks during tests without factories.
    """

    def __init__(
            self,
            scanner: SpheroFinder | None = SpheroFinder, # FIXME: get class not instance
            api_class: SpheroEduAPI = SpheroEduAPI,
            register_handlers=False,
    ):
        self.scanner: SpheroFinder = scanner or SpheroFinder() # FIXME: get class not instance
        self.api_class: SpheroEduAPI | None = api_class
        self.api: SpheroEduAPI | None = None
        self.toy: BOLTPLUS | None = None
        self.name: str | None = None

        self.heading = 0

        if register_handlers:
            atexit.register(self._cleanup)
            signal.signal(signal.SIGINT, self._signal_cleanup)
            signal.signal(signal.SIGTERM, self._signal_cleanup)
            weakref.finalize(self, self._cleanup)

    # -----------------------------------------------------

    def connect(self, bolt_name: str = cfg.SPHEROBOLTPLUS_NAME):
        logger.info("Scanning for Sphero BOLT...")
        toys = self.scanner.scan_toys()

        self.toy = self.scanner.select_toy(bolt_name)
        if not self.toy:
            raise RuntimeError(f"Sphero '{bolt_name}' not found, avalible Toys: {[toy.name for toy in toys]}")

        self.name = str(self.toy.name)
        # If api was not injected, create it now
        self.api = self.api_class(self.toy)
        self.api.__enter__()

        logger.info(f"Connected to {self.name}")

    # -----------------------------------------------------

    def disconnect(self):
        self._cleanup()

    def _cleanup(self):
        if self.api:
            try:
                self.api.__exit__(None, None, None)
                logger.info("Disconnected from Sphero BOLT.")
            finally:
                self.api = None

    def _signal_cleanup(self, signum, frame):
        logger.debug(f"Signal {signum} received. Cleaning up...")
        self._cleanup()
        raise SystemExit(0)

    # -----------------------------------------------------
    def __exit__(self, exc_type, exc_val, exc_tb):
        self._cleanup()

    def _require_connection(self):
        if not self.api:
            raise RuntimeError("Robot is not connected.")

    def move(self, heading: int, speed=cfg.SPHEROBOLTPLUS_SPEED, duration=cfg.SPHEROBOLTPLUS_DURATION):
        """Move the Sphero in a relative direction.

        Arttributes:
            - heading: Moving direktion (0-360°)
            - speed: Moving speed (-255 - 255)
            - duration: Moving duration (seconds)

        Return: None
        """
        self._require_connection()

        self.heading = (self.heading + heading + 360) % 360
        logger.info(f"Moving: heading={heading}, speed={speed}, duration={duration}")
        self.api.roll(self.heading, 0, duration)
        self.api.roll(self.heading, speed, duration)

    def get_sensor_data(self) -> dict[str, Any]:
        """Returns a snapshot of all sensor readings.

        Returns:
            dict[str, Any]: A dictionary containing:
                - ambient_light (float): Measured ambient light level.
                - orientation (dict[str, float]): {"pitch", "roll", "yaw"} in degrees.
                - velocity (dict[str, float]): {"x", "y"} velocity in m/s.
                - location (dict[str, float]): {"x", "y"} position relative to start.
                - gyroscope (dict[str, float]): {"x", "y", "z"} angular velocity in deg/s.
                - acceleration ([str, float]): {"x", "y", "z"} linear acceleration.
                - travel_distance (float): Total distance traveled in meters.
                - heading (float): Heading angle in degrees (0–359).
        """
        self._require_connection()

        return {
            "ambient_light": self.api.get_luminosity()['ambient_light'],
            "orientation": self.api.get_orientation(),
            "velocity": self.api.get_velocity(),
            "location": self.api.get_location(),
            "gyroscope": self.api.get_gyroscope(),
            "acceleration": self.api.get_acceleration(),
            "travel_distance": self.api.get_distance(),
            "heading": self.api.get_heading(),
        }


if __name__ == "__main__":
    from rlive_world.bolt.boltdummys import DummySpheroEduAPI, DummyFinder

    robot = SpheroBoltPlus(api_class=DummySpheroEduAPI, scanner=DummyFinder())
    robot.connect("DummyBolt")
    robot.move(heading=0)
    robot.move(heading=90)
    robot.move(heading=0)
    # data = robot.get_sensor_data()
    # for key, value in data.items():
    #     logger.info(f"{key.replace('_', ' ').title()}: {value}")
    robot.disconnect()
