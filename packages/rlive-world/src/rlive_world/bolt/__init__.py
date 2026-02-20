"""
Sphero BOLT+ robot control module.

This module provides classes for controlling Sphero BOLT+ robots,
including movement, sensor reading, and LED matrix display.

Example:
    from rlive_world.bolt import SpheroBoltPlus, Color

    robot = SpheroBoltPlus(register_handlers=True)
    robot.connect("BP-D217", timeout=3)

    # Display an arrow
    robot.display_arrow("up", Color(0, 255, 0))

    # Move the robot
    robot.move(heading=90, speed=50, duration=1)

    robot.disconnect()
"""

# Main robot class
from rlive_world.bolt.sphero_bolt_plus import SpheroBoltPlus

# Bitmap types and patterns for LED matrix
from rlive_world.bolt.bitmaps import (
    Bitmap,
    ARROW_UP,
    ARROW_DOWN,
    ARROW_LEFT,
    ARROW_RIGHT,
    ARROW_BITMAPS,
)

# Re-export Color from sphero_unsw for convenience
from sphero_unsw.types import Color

__all__ = [
    # Robot class
    "SpheroBoltPlus",
    # Color type
    "Color",
    # Bitmap type and patterns
    "Bitmap",
    "ARROW_UP",
    "ARROW_DOWN",
    "ARROW_LEFT",
    "ARROW_RIGHT",
    "ARROW_BITMAPS",
]
