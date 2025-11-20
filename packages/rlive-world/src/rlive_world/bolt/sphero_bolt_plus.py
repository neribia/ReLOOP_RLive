import atexit
import signal
import weakref
import time
from typing import Optional, Callable

from sphero_unsw.sphero_edu import SpheroEduAPI
from sphero_unsw.toy.boltplus import BOLTPLUS

from rlive_world.bolt.base_robot import BaseRobot
from rlive_world.bolt.sphero_finder import SpheroFinder
from rlive_world.config import config as cfg
from rlive_common.utils import get_logger

logger = get_logger(__name__)


class SpheroBoltPlus(BaseRobot):
    """
    Simple and testable robot class.
    Optional arguments allow injecting mocks during tests without factories.
    """

    def __init__(
        self,
        scanner: Optional[SpheroFinder] = None,
        api: Optional[SpheroEduAPI] = None,
        sleep_fn=time.sleep,
        register_handlers=True,
    ):
        self.scanner: SpheroFinder = scanner or SpheroFinder()
        self.api: SpheroEduAPI | None = api  # set after connect() normally
        self.sleep: Callable[[float], None] = sleep_fn
        self.toy: BOLTPLUS | None = None
        self.name: str | None = None

        if register_handlers:
            atexit.register(self._cleanup)
            signal.signal(signal.SIGINT, self._signal_cleanup)
            signal.signal(signal.SIGTERM, self._signal_cleanup)
            weakref.finalize(self, self._cleanup)

    # -----------------------------------------------------

    def connect(self, bolt_name: str = cfg.SPHEROBOLTPLUS_NAME):
        logger.info("Scanning for Sphero BOLT...")
        self.scanner.scan_toys()

        self.toy = self.scanner.select_toy(bolt_name)
        if not self.toy:
            raise RuntimeError(f"Sphero '{bolt_name}' not found")

        self.name = str(self.toy.name)

        # If api was not injected, create it now
        self.api = self.api or SpheroEduAPI(self.toy)
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

    def _require_connection(self):
        if not self.api:
            raise RuntimeError("Robot is not connected.")

    def move(self, heading: int, speed=cfg.SPHEROBOLTPLUS_SPEED, duration=cfg.SPHEROBOLTPLUS_DURATION):
        self._require_connection()
        logger.info(f"Moving: heading={heading}, speed={speed}, duration={duration}")
        self.api.roll(heading, speed, duration)
        self.sleep(duration)


if __name__ == "__main__":
    robot = SpheroBoltPlus()
    robot.connect()
    robot.move(heading=0)
    robot.move(heading=90)
    robot.disconnect()
