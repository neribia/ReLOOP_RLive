"""
8x8 Bitmap patterns for Sphero BOLT LED matrix display.

Each bitmap is an 8x8 grid where True = pixel on, False = pixel off.
Bitmaps are defined as list[list[bool]] with [row][column] indexing.
Row 0 is top, Column 0 is left.
"""

from typing import TypeAlias

# Type alias for 8x8 bitmap pattern
Bitmap: TypeAlias = list[list[bool]]

# fmt: off

ARROW_UP: Bitmap = [
    [False, False, False, True, True, False, False, False],
    [False, False, True, True, True, True, False, False],
    [False, True, True, True, True, True, True, False],
    [True, True, True, True, True, True, True, True],
    [False, False, True, True, True, True, False, False],
    [False, False, True, True, True, True, False, False],
    [False, False, True, True, True, True, False, False],
    [False, False, True, True, True, True, False, False],
]

ARROW_DOWN: Bitmap = [
    [False, False, True, True, True, True, False, False],
    [False, False, True, True, True, True, False, False],
    [False, False, True, True, True, True, False, False],
    [False, False, True, True, True, True, False, False],
    [True, True, True, True, True, True, True, True],
    [False, True, True, True, True, True, True, False],
    [False, False, True, True, True, True, False, False],
    [False, False, False, True, True, False, False, False],
]

ARROW_LEFT: Bitmap = [
    [False, False, False, True, False, False, False, False],
    [False, False, True, True, False, False, False, False],
    [False, True, True, True, True, True, True, True],
    [True, True, True, True, True, True, True, True],
    [True, True, True, True, True, True, True, True],
    [False, True, True, True, True, True, True, True],
    [False, False, True, True, False, False, False, False],
    [False, False, False, True, False, False, False, False],
]

ARROW_RIGHT: Bitmap = [
    [False, False, False, False, True, False, False, False],
    [False, False, False, False, True, True, False, False],
    [True, True, True, True, True, True, True, False],
    [True, True, True, True, True, True, True, True],
    [True, True, True, True, True, True, True, True],
    [True, True, True, True, True, True, True, False],
    [False, False, False, False, True, True, False, False],
    [False, False, False, False, True, False, False, False],
]

# fmt: on

# Direction name to bitmap mapping
ARROW_BITMAPS: dict[str, Bitmap] = {
    "up": ARROW_UP,
    "down": ARROW_DOWN,
    "left": ARROW_LEFT,
    "right": ARROW_RIGHT,
}
