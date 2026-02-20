from rlive_common.utils import get_logger

logger = get_logger(__name__)


# ============================================================
#   DUMMY API (replacement for SpheroEduAPI)
# ============================================================

class DummySpheroEduAPI:
    NOTIFY_UUID = "DUMMY-NOTIFY-UUID"
    COMMAND_UUID = "DUMMY-CMD-UUID"

    def __init__(self, toy):
        self.toy = toy
        self.sensors = {
            "pitch": 0.0,
            "roll": 0.0,
            "yaw": 0.0,
            "acc_x": 0.0,
            "acc_y": 0.0,
            "acc_z": 9.8,
        }

    def __enter__(self):
        logger.debug("[DUMMY] API __enter__")

        # connect fake notifications
        self.toy.adapter.set_callback(
            self.NOTIFY_UUID,
            self.handle_notification
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        logger.debug("[DUMMY] API __exit__")

    # BLE write simulation
    def roll(self, heading, speed, duration):
        logger.debug(f"[DUMMY] roll: heading={heading}, speed={speed}, duration={duration}")
        packet = f"HEADING={heading},SPEED={speed},DURATION={duration}".encode()
        self.toy.adapter.write(self.COMMAND_UUID, packet)

    # BLE notification handler
    def handle_notification(self, uuid, data): # FIXME: should this be here or should it be on a device
        self.sensors["yaw"] = data[0]
        self.sensors["pitch"] = data[1]
        self.sensors["roll"] = data[2]
        self.sensors["acc_x"] = (data[3] - 128) / 10
        self.sensors["acc_y"] = (data[4] - 128) / 10
        self.sensors["acc_z"] = (data[5] - 128) / 10

    # getters
    def get_orientation(self):
        return {
            "pitch": self.sensors["pitch"],
            "roll": self.sensors["roll"],
            "yaw": self.sensors["yaw"],
        }

    def get_acceleration(self):
        return {
            "x": self.sensors["acc_x"],
            "y": self.sensors["acc_y"],
            "z": self.sensors["acc_z"],
        }

    # ---------------------------------------------------------
    # LED Matrix display methods (8x8 matrix)
    # ---------------------------------------------------------

    def set_matrix_pixel(self, x: int, y: int, color):
        """Set a single pixel on the 8x8 LED matrix."""
        logger.debug(f"[DUMMY] set_matrix_pixel: x={x}, y={y}, color={color}")

    def set_matrix_character(self, char: str, color):
        """Display a character on the 8x8 LED matrix."""
        logger.debug(f"[DUMMY] set_matrix_character: char='{char}', color={color}")

    def set_matrix_fill(self, x1: int, y1: int, x2: int, y2: int, color):
        """Fill a rectangular region on the 8x8 LED matrix."""
        logger.debug(f"[DUMMY] set_matrix_fill: ({x1},{y1}) to ({x2},{y2}), color={color}")

    def scroll_matrix_text(self, text: str, color, fps: int = 5, wait: bool = True):
        """Scroll text across the 8x8 LED matrix."""
        logger.debug(f"[DUMMY] scroll_matrix_text: text='{text}', color={color}, fps={fps}, wait={wait}")

    def register_matrix_animation(self, frames, palette, fps: int, transition: bool):
        """Register a matrix animation."""
        palette_len = len(palette) if palette is not None else 0
        logger.debug(f"[DUMMY] register_matrix_animation: frames={len(frames)}, palette={palette_len}, fps={fps}, transition={transition}")

    def play_matrix_animation(self, animation_id: int, loop: bool = True):
        """Play a matrix animation."""
        logger.debug(f"[DUMMY] play_matrix_animation: animation_id={animation_id}, loop={loop}")

    def clear_matrix(self):
        """Clear the matrix animation."""
        logger.debug("[DUMMY] clear_matrix")
