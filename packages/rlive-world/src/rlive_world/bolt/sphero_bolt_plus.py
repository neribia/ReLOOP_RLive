import atexit
import signal
import weakref
import time
from typing import Any, Literal

import numpy as np

from sphero_unsw.sphero_edu import SpheroEduAPI
from sphero_unsw.toy.boltplus import BOLTPLUS
from sphero_unsw.types import Color

from rlive_world.bolt.base_robot import BaseRobot
from rlive_world.bolt.sphero_finder import SpheroFinder
from rlive_world.bolt.bitmaps import Bitmap, ARROW_BITMAPS
from rlive_world.config import config as cfg
from rlive_common.utils import get_logger

logger = get_logger(__name__)


class SpheroBoltPlus(BaseRobot):
    """Simple and testable robot class.
    Optional arguments allow injecting mocks during tests without factories.
    """

    def __init__(
            self,
            scanner_class: type[SpheroFinder] = SpheroFinder,
            api_class: type[SpheroEduAPI] = SpheroEduAPI,
            register_handlers=False,
    ):
        self.scanner: SpheroFinder = scanner_class()
        self.api_class: SpheroEduAPI = api_class
        self.api: SpheroEduAPI | None = None
        self.toy: BOLTPLUS | None = None
        self.name: str | None = None

        self._animation_index = 0

        self.heading = 0

        if register_handlers:
            atexit.register(self._cleanup)
            signal.signal(signal.SIGINT, self._signal_cleanup)
            signal.signal(signal.SIGTERM, self._signal_cleanup)
            weakref.finalize(self, self._cleanup)

    # -----------------------------------------------------

    def connect(self, bolt_name: str, **kwargs):
        """Connect to the Sphero BOLT robot by name.

        Args:
            - bolt_name: str: Name of the Sphero BOLT to connect to.
            - timeout: float: Time to scan for the robot (seconds).

        Returns: None
        """
        logger.info("Scanning for Sphero BOLT...")
        toys = self.scanner.scan_toys(**kwargs)

        self.toy = self.scanner.select_toy(bolt_name)
        if not self.toy:
            raise RuntimeError(f"Sphero '{bolt_name}' not found, available Toys: {[toy.name for toy in toys]}")

        self.name = str(self.toy.name)
        # If api was not injected, create it now
        self.api = self.api_class(self.toy)
        self.api.__enter__()

        logger.info(f"Connected to {self.name}")

    # -----------------------------------------------------

    def disconnect(self):
        self._cleanup()

    def _cleanup(self):
        self._animation_index = 0
        
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

    def move(self, heading: int, speed: int, duration: float):
        """Move the Sphero in a relative direction.

        Attributes:
            - heading (int): Moving direktion (0-360°)
            - speed (int): Moving speed (-255 - 255)
            - duration (float): Moving duration (seconds)

        Return: None
        """
        self._require_connection()

        self.heading = (self.heading + heading) % 360
        logger.info(f"Moving: heading={heading}, speed={speed}, duration={duration}")
        self.api.roll(self.heading, 0, duration)
        time.sleep(duration/2)
        self.api.roll(self.heading, speed, duration)
        time.sleep(duration/2)

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

    # ---------------------------------------------------------
    # LED Matrix Display Methods
    # ---------------------------------------------------------

    def _get_default_color(self) -> Color:
        """Get the default display color from config."""
        return Color(
            cfg.SPHEROBOLTPLUS_DISPLAY_COLOR_R,
            cfg.SPHEROBOLTPLUS_DISPLAY_COLOR_G,
            cfg.SPHEROBOLTPLUS_DISPLAY_COLOR_B,
        )

    def display_bitmap(self, bitmap: Bitmap, color: Color | None = None) -> None:
        """Display an 8x8 bitmap pattern on the LED matrix.

        Args:
            bitmap: 8x8 boolean grid where True = pixel on, False = pixel off.
            color: RGB color for lit pixels. Uses default config color if None.
        """
        self._require_connection()

        if color is None:
            color = self._get_default_color()

        # Validate bitmap shape and type before building the frame.
        # Expect exactly an 8x8 grid of boolean values.
        try:
            bitmap_array = np.asarray(bitmap)
        except Exception as exc:
            raise TypeError("bitmap must be an array-like 8x8 grid of booleans") from exc

        if bitmap_array.shape != (8, 8):
            raise ValueError(
                f"bitmap must be 8x8, got shape {bitmap_array.shape!r}"
            )

        if bitmap_array.dtype != np.bool_:
            # Allow values that can be sensibly interpreted as booleans (e.g. 0/1).
            try:
                bitmap_array = bitmap_array.astype(bool)
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    "bitmap values must be boolean or boolean-convertible"
                ) from exc

        # Build frame using numpy - convert boolean bitmap to palette indices (0 or 1)
        # Frame format: 8x8 grid of integers (palette indices)
        # Palette: index 0 = black (off), index 1 = color (on)
        frame = bitmap_array.astype(np.uint8)
        frame = np.fliplr(frame)  # Flip horizontally to fix left/right swap
        frame = np.rot90(frame, k=1)
        frame_list = frame.tolist()

        # Create palette: index 0 = black, index 1 = the specified color
        palette = [Color(0, 0, 0), color]

        # NOTE: register_matrix_animation() does NOT return the animation index,
        # and registered animations persist until disconnect (clear_matrix() only stops playback).
        # We must track _animation_index manually.
        self.api.register_matrix_animation(
            frames=[frame_list],
            palette=palette,
            fps=1,
            transition=False
        )
        self.api.play_matrix_animation(animation_id=self._animation_index, loop=True)

        self._animation_index += 1
        logger.debug("Displayed bitmap using matrix animation")

    def display_arrow(
        self,
        direction: Literal["up", "down", "left", "right"],
        color: Color | None = None,
    ) -> None:
        """Display an arrow pointing in the specified direction.

        Args:
            direction: Arrow direction ("up", "down", "left", "right").
            color: RGB color for the arrow. Uses default config color if None.

        Raises:
            ValueError: If direction is not valid.
        """
        if direction not in ARROW_BITMAPS:
            raise ValueError(f"Invalid direction '{direction}'. Must be one of: {list(ARROW_BITMAPS.keys())}")

        bitmap = ARROW_BITMAPS[direction]
        self.display_bitmap(bitmap, color)
        logger.info(f"Displayed arrow: {direction}")

    def display_character(self, char: str, color: Color | None = None) -> None:
        """Display a single character on the LED matrix.

        Args:
            char: Single character to display.
            color: RGB color for the character. Uses default config color if None.
        """
        self._require_connection()

        if color is None:
            color = self._get_default_color()

        self.api.set_matrix_character(char, color)
        logger.info(f"Displayed character: '{char}'")

    def scroll_text(
        self,
        text: str,
        color: Color | None = None,
        fps: int = 5,
        wait: bool = True,
    ) -> None:
        """Scroll text across the LED matrix.

        Args:
            text: Text string to scroll.
            color: RGB color for the text. Uses default config color if None.
            fps: Frames per second for scrolling animation.
            wait: If True, block until scrolling completes.
        """
        self._require_connection()

        if color is None:
            color = self._get_default_color()

        self.api.scroll_matrix_text(text, color, fps=fps, wait=wait)  # Forward wait to underlying API; see its docs for blocking behavior.
        logger.info(f"Scrolled text: '{text}'")

    def clear_display(self) -> None:
        """Clear the LED matrix (turn off all pixels)."""
        self._require_connection()

        self.api.clear_matrix()

        logger.debug("Cleared LED matrix display")
