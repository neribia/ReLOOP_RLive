"""Tests for the bitmaps module."""

import unittest

from rlive_world.bolt.bitmaps import (
    Bitmap,
    ARROW_UP,
    ARROW_DOWN,
    ARROW_LEFT,
    ARROW_RIGHT,
    ARROW_BITMAPS,
)


class TestBitmaps(unittest.TestCase):
    """Test suite for bitmap definitions."""

    def test_arrow_bitmaps_are_8x8(self):
        """Test that all arrow bitmaps are 8x8 grids."""
        for name, bitmap in ARROW_BITMAPS.items():
            self.assertEqual(len(bitmap), 8, f"{name} bitmap should have 8 rows")
            for row_idx, row in enumerate(bitmap):
                self.assertEqual(len(row), 8, f"{name} bitmap row {row_idx} should have 8 columns")

    def test_arrow_bitmaps_contain_booleans(self):
        """Test that all arrow bitmaps contain only boolean values."""
        for name, bitmap in ARROW_BITMAPS.items():
            for row_idx, row in enumerate(bitmap):
                for col_idx, pixel in enumerate(row):
                    self.assertIsInstance(
                        pixel, bool,
                        f"{name} bitmap[{row_idx}][{col_idx}] should be bool, got {type(pixel)}"
                    )

    def test_arrow_bitmaps_have_pixels(self):
        """Test that all arrow bitmaps have at least some pixels turned on."""
        for name, bitmap in ARROW_BITMAPS.items():
            pixel_count = sum(sum(row) for row in bitmap)
            self.assertGreater(
                pixel_count, 0,
                f"{name} bitmap should have at least one pixel on"
            )

    def test_arrow_bitmaps_dict_contains_all_directions(self):
        """Test that ARROW_BITMAPS contains all four directions."""
        expected_directions = {"up", "down", "left", "right"}
        self.assertEqual(set(ARROW_BITMAPS.keys()), expected_directions)

    def test_arrow_up_points_upward(self):
        """Test that ARROW_UP has more pixels in top half than bottom half."""
        top_half = sum(sum(ARROW_UP[row]) for row in range(4))
        bottom_half = sum(sum(ARROW_UP[row]) for row in range(4, 8))
        # Arrow up should have more pixels in top half (the arrow point)
        self.assertGreaterEqual(top_half, bottom_half)

    def test_arrow_down_points_downward(self):
        """Test that ARROW_DOWN has more pixels in bottom half than top half."""
        top_half = sum(sum(ARROW_DOWN[row]) for row in range(4))
        bottom_half = sum(sum(ARROW_DOWN[row]) for row in range(4, 8))
        # Arrow down should have more pixels in bottom half (the arrow point)
        self.assertGreaterEqual(bottom_half, top_half)

    def test_arrow_left_points_leftward(self):
        """Test that ARROW_LEFT has more pixels in left half than right half."""
        left_half = sum(sum(row[col] for col in range(4)) for row in ARROW_LEFT)
        right_half = sum(sum(row[col] for col in range(4, 8)) for row in ARROW_LEFT)
        # Arrow left should have more pixels in left half (the arrow point)
        self.assertGreaterEqual(left_half, right_half)

    def test_arrow_right_points_rightward(self):
        """Test that ARROW_RIGHT has more pixels in right half than left half."""
        left_half = sum(sum(row[col] for col in range(4)) for row in ARROW_RIGHT)
        right_half = sum(sum(row[col] for col in range(4, 8)) for row in ARROW_RIGHT)
        # Arrow right should have more pixels in right half (the arrow point)
        self.assertGreaterEqual(right_half, left_half)


if __name__ == "__main__":
    unittest.main()
