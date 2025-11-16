import atexit
import signal
import weakref

import time
from sphero_unsw.sphero_edu import SpheroEduAPI

from rlive_world.bolt.base_robot import BaseRobot
from rlive_world.bolt.bolt_finder import SpheroFinder
from rlive_world.config import config as cfg
from rlive_common.utils import get_logger

logger = get_logger(__name__)


def requires_connection(func):
    def wrapper(self, *args, **kwargs):
        if not self.api:
            logger.error("Not connected. Use _connect() first.")
            # FIXME: Maybe raise Exception
            return None
        return func(self, *args, **kwargs)

    return wrapper


class SpheroBoltPlus(BaseRobot):
    def __init__(self):
        """Initialize the SpheroBoltPlus instance."""
        self.scanner = SpheroFinder()
        self.toy = None
        self.api = None
        self.name = None

        # Ensure cleanup at interpreter exit
        atexit.register(self._cleanup)

        # Handle Ctrl-C and kill gracefully
        signal.signal(signal.SIGINT, self._signal_cleanup)
        signal.signal(signal.SIGTERM, self._signal_cleanup)

        # Fallback finalizer if object is GC'd
        weakref.finalize(self, self._cleanup)

    def connect(self, bolt_name: str = cfg.SPHEROBOLTPLUS_NAME) -> None:
        """Scan for and connect to a nearby Sphero BOLT."""
        self.scanner.scan_toys()
        self.toy = self.scanner.select_toy(name=bolt_name)
        self.name = self.toy.name if self.toy else None
        self.api = SpheroEduAPI(self.toy)
        self.api.__enter__()  # proper connection method
        logger.info(f"Connected to {self.toy.name}")

    def disconnect(self) -> None:
        self._cleanup()

    def _cleanup(self) -> None:
        """Disconnect from the robot safely."""
        if self.api:
            try:
                self.api.__exit__(None, None, None)  # proper release method
                logger.info("Disconnected from Sphero BOLT.")
            except Exception as e:
                logger.error(f"Error during disconnect: {e}")
            finally:
                self.api = None

    def _signal_cleanup(self, signum, frame) -> None:
        logger.debug(f"Received signal {signum}, cleaning up...")
        self._cleanup()
        raise SystemExit(0)

    @requires_connection
    def move(
        self, heading: int = 0, speed: int = cfg.SPHEROBOLTPLUS_SPEED, duration: float = cfg.SPHEROBOLTPLUS_DURATION
    ) -> None:
        """Move the Sphero in a given direction.

        Arttributes:
            - heading: Moving direktion (0-360°)
            - speed: Moving speed (-255 - 255)
            - duration: Moving duration (seconds)

        Retrun: None
        """
        logger.info(f"Moving at heading {heading}°, speed {speed}")
        self.api.roll(heading, speed, duration)
        time.sleep(duration)


if __name__ == "__main__":
    bot = SpheroBoltPlus()
    bot.connect()
    bot.move(0, 100, 1)
    bot.disconnect()
