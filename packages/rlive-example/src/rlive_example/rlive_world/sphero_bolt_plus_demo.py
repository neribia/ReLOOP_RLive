"""
Sphero BOLT+ Demo Script

This script demonstrates all the features of the SpheroBoltPlus class:
- LED Matrix Display (arrows, characters, scrolling text, custom bitmaps)
- LED Control (main, front, back LEDs)
- Movement control
- Sensor data reading

Usage:
    uv run python examples/sphero_bolt_plus_demo.py

Make sure your Sphero BOLT+ is turned on and in range before running.
Update the BOLT_NAME variable to match your robot's Bluetooth name.
"""

import time

from typing import Literal

from rlive_world.bolt import SpheroBoltPlus, Color, ARROW_UP
from rlive_common.utils import get_logger

logger = get_logger(__name__)

# Configuration - Update this to match your Sphero BOLT+ name
BOLT_NAME = "BP-D217"
SCAN_TIMEOUT = 3


def demo_display_arrows(robot: SpheroBoltPlus) -> None:
    """Demonstrate arrow display on the LED matrix."""
    logger.info("=== Arrow Display Demo ===")

    directions: list[tuple[Literal["up", "down", "left", "right"], Color]] = [
        ("up", Color(0, 255, 0)),      # Green
        ("right", Color(255, 255, 0)), # Yellow
        ("down", Color(255, 0, 0)),    # Red
        ("left", Color(0, 0, 255)),    # Blue
    ]

    for direction, color in directions:
        logger.info(f"Displaying {direction} arrow...")
        robot.display_arrow(direction, color)
        time.sleep(1)

    robot.clear_display()
    logger.info("Arrow demo complete.\n")


def demo_display_characters(robot: SpheroBoltPlus) -> None:
    """Demonstrate character display on the LED matrix."""
    logger.info("=== Character Display Demo ===")

    characters = [
        ("H", Color(255, 0, 0)),    # Red H
        ("I", Color(0, 255, 0)),    # Green I
        ("!", Color(0, 0, 255)),    # Blue !
    ]

    for char, color in characters:
        logger.info(f"Displaying character: '{char}'")
        robot.display_character(char, color)
        time.sleep(0.8)

    robot.clear_display()
    logger.info("Character demo complete.\n")


def demo_scroll_text(robot: SpheroBoltPlus) -> None:
    """Demonstrate scrolling text on the LED matrix."""
    logger.info("=== Scroll Text Demo ===")

    logger.info("Scrolling 'HELLO'...")
    robot.scroll_text("HELLO", Color(0, 255, 255), fps=5, wait=True)
    time.sleep(0.5)

    logger.info("Scrolling 'BOLT+'...")
    robot.scroll_text("BOLT+", Color(255, 0, 255), fps=8, wait=True)
    time.sleep(0.5)

    robot.clear_display()
    logger.info("Scroll text demo complete.\n")


def demo_custom_bitmap(robot: SpheroBoltPlus) -> None:
    """Demonstrate custom bitmap display on the LED matrix."""
    logger.info("=== Custom Bitmap Demo ===")

    # Create a simple smiley face bitmap
    # fmt: off
    smiley = [
        [False, False, True,  True,  True,  True,  False, False],
        [False, True,  False, False, False, False, True,  False],
        [True,  False, True,  False, False, True,  False, True ],
        [True,  False, False, False, False, False, False, True ],
        [True,  False, True,  False, False, True,  False, True ],
        [True,  False, False, True,  True,  False, False, True ],
        [False, True,  False, False, False, False, True,  False],
        [False, False, True,  True,  True,  True,  False, False],
    ]
    # fmt: on

    logger.info("Displaying custom smiley bitmap...")
    robot.display_bitmap(smiley, Color(255, 255, 0))  # Yellow smiley
    time.sleep(2)

    # Use the predefined arrow bitmaps
    logger.info("Displaying predefined ARROW_UP bitmap...")
    robot.display_bitmap(ARROW_UP, Color(0, 255, 0))
    time.sleep(1)

    robot.clear_display()
    logger.info("Custom bitmap demo complete.\n")


def demo_movement(robot: SpheroBoltPlus) -> None:
    """Demonstrate movement control."""
    logger.info("=== Movement Demo ===")

    logger.info("Moving forward...")
    robot.move(heading=0, speed=50, duration=1)

    logger.info("Moving backward...")
    robot.move(heading=90, speed=50, duration=1)

    logger.info("Turning right...")
    robot.move(heading=90, speed=30, duration=0.5)

    logger.info("Movement demo complete.\n")


def demo_sensors(robot: SpheroBoltPlus) -> None:
    """Demonstrate sensor data reading."""
    logger.info("=== Sensor Data Demo ===")

    data = robot.get_sensor_data()

    logger.info("Current sensor readings:")
    for key, value in data.items():
        logger.info(f"  {key.replace('_', ' ').title()}: {value}")

    logger.info("Sensor demo complete.\n")


def main():
    """Main demo function."""
    logger.info("=" * 50)
    logger.info("Sphero BOLT+ Demo")
    logger.info("=" * 50)

    robot = SpheroBoltPlus(register_handlers=True)

    try:
        # Connect to the robot
        logger.info(f"Connecting to {BOLT_NAME}...")
        robot.connect(bolt_name=BOLT_NAME, timeout=SCAN_TIMEOUT)
        logger.info(f"Connected to {robot.name}!\n")

        # Run demos
        #demo_display_arrows(robot)
        #demo_display_characters(robot)
        #demo_scroll_text(robot)
        #demo_custom_bitmap(robot)
        demo_movement(robot)  # Uncomment to enable movement
        #demo_sensors(robot)

        logger.info("=" * 50)
        logger.info("All demos complete!")
        logger.info("=" * 50)

    except Exception as e:
        logger.exception(f"An error occurred: {e}")

    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
