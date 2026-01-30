"""Tests for the SpheroBoltPlus robot class."""

import unittest
from unittest.mock import MagicMock

from rlive_world.bolt.sphero_bolt_plus import SpheroBoltPlus
from rlive_world.bolt.boltdummys import (
    DummyFinder,
    DummyToy,
    DummySpheroEduAPI
)


class TestSpheroBoltPlus(unittest.TestCase):
    """Test suite for SpheroBoltPlus robot using dummy components."""

    def test_init_with_defaults(self):
        """Test that robot initializes with default values."""
        robot = SpheroBoltPlus(register_handlers=False)

        self.assertEqual(robot.heading, 0)
        self.assertIsNone(robot.name)
        self.assertIsNone(robot.api)
        self.assertIsNone(robot.toy)

    def test_init_with_dummy_components(self):
        """Test that robot initializes with dummy components."""
        robot = SpheroBoltPlus(
            scanner_class=DummyFinder,
            api_class=DummySpheroEduAPI,  # type: ignore
            register_handlers=False
        )

        self.assertEqual(robot.heading, 0)
        self.assertIsNone(robot.name)
        self.assertIsNone(robot.api)

    # ---------------------------------------------------------------------

    def test_connect_with_dummy_components(self):
        """Test that robot connects using dummy components."""
        robot = SpheroBoltPlus(
            scanner_class=DummyFinder,
            api_class=DummySpheroEduAPI,  # type: ignore
            register_handlers=False
        )

        robot.connect("DummyBolt", timeout=0.01)

        # Assertions
        self.assertEqual(robot.name, "DummyBolt")
        self.assertIsNotNone(robot.api)
        self.assertIsNotNone(robot.toy)
        self.assertIsInstance(robot.toy, DummyToy)

        robot.disconnect()

    def test_connect_toy_not_found(self):
        """Test that connect raises RuntimeError when toy not found."""
        robot = SpheroBoltPlus(
            scanner_class=DummyFinder,
            api_class=DummySpheroEduAPI,  # type: ignore
            register_handlers=False
        )

        with self.assertRaises(RuntimeError) as context:
            robot.connect("NonExistentBolt", timeout=0.01)

        self.assertIn("not found", str(context.exception))

    # ---------------------------------------------------------------------

    def test_disconnect_calls_cleanup_and_closes_api(self):
        """Test that disconnect calls cleanup and closes the API."""
        robot = SpheroBoltPlus(
            scanner_class=DummyFinder,
            api_class=DummySpheroEduAPI,  # type: ignore
            register_handlers=False
        )

        robot.connect("DummyBolt", timeout=0.01)
        self.assertIsNotNone(robot.api)

        # Call disconnect
        robot.disconnect()

        # API must be cleared
        self.assertIsNone(robot.api)

    def test_disconnect_when_not_connected(self):
        """Test that disconnect works when not connected."""
        robot = SpheroBoltPlus(
            scanner_class=DummyFinder,
            api_class=DummySpheroEduAPI,  # type: ignore
            register_handlers=False
        )

        # Should not raise
        robot.disconnect()
        self.assertIsNone(robot.api)

    # ---------------------------------------------------------------------

    def test_require_connection_not_connected(self):
        """Test that _require_connection raises when not connected."""
        robot = SpheroBoltPlus(
            scanner_class=DummyFinder,
            api_class=DummySpheroEduAPI,  # type: ignore
            register_handlers=False
        )

        with self.assertRaises(RuntimeError) as context:
            robot._require_connection()

        self.assertIn("not connected", str(context.exception))

    def test_require_connection_connected(self):
        """Test that _require_connection does not raise when connected."""
        robot = SpheroBoltPlus(
            scanner_class=DummyFinder,
            api_class=DummySpheroEduAPI,  # type: ignore
            register_handlers=False
        )

        robot.connect("DummyBolt", timeout=0.01)
        robot._require_connection()  # should NOT raise
        robot.disconnect()

    # ---------------------------------------------------------------------

    def test_move_with_dummy_components(self):
        """Test that robot move command works correctly with dummy components."""
        robot = SpheroBoltPlus(
            scanner_class=DummyFinder,
            api_class=DummySpheroEduAPI,  # type: ignore
            register_handlers=False
        )

        robot.connect("DummyBolt", timeout=0.01)

        # Test first move
        robot.move(heading=90, speed=100, duration=1)
        self.assertEqual(robot.heading, 90)

        # Test second move (heading should accumulate)
        robot.move(heading=90, speed=100, duration=1)
        self.assertEqual(robot.heading, 180)

        # Test third move
        robot.move(heading=180, speed=100, duration=1)
        self.assertEqual(robot.heading, 0)  # 180 + 180 = 360 = 0

        robot.disconnect()

    def test_move_not_connected(self):
        """Test that move raises when not connected."""
        robot = SpheroBoltPlus(
            scanner_class=DummyFinder,
            api_class=DummySpheroEduAPI,  # type: ignore
            register_handlers=False
        )

        with self.assertRaises(RuntimeError) as context:
            robot.move(heading=90, speed=100, duration=1)

        self.assertIn("not connected", str(context.exception))

    def test_move_heading_wrapping(self):
        """Test that heading wraps correctly at 360 degrees."""
        robot = SpheroBoltPlus(
            scanner_class=DummyFinder,
            api_class=DummySpheroEduAPI,  # type: ignore
            register_handlers=False
        )

        robot.connect("DummyBolt", timeout=0.01)

        # Test positive wrapping
        robot.heading = 350
        robot.move(heading=20, speed=100, duration=1)
        self.assertEqual(robot.heading, 10)  # (350 + 20) % 360 = 10

        # Test negative wrapping (if heading goes negative)
        robot.heading = 10
        robot.move(heading=-20, speed=100, duration=1)
        self.assertEqual(robot.heading, 350)  # (10 - 20 + 360) % 360 = 350

        robot.disconnect()

    # ---------------------------------------------------------------------

    def test_get_sensor_data(self):
        """Test that get_sensor_data returns expected structure."""
        robot = SpheroBoltPlus(
            scanner_class=DummyFinder,
            api_class=DummySpheroEduAPI,  # type: ignore
            register_handlers=False
        )

        robot.connect("DummyBolt", timeout=0.01)

        # Mock the sensor methods that don't exist in DummySpheroEduAPI
        robot.api.get_luminosity = MagicMock(return_value={"ambient_light": 100})
        robot.api.get_velocity = MagicMock(return_value={"x": 0, "y": 0})
        robot.api.get_location = MagicMock(return_value={"x": 0, "y": 0})
        robot.api.get_gyroscope = MagicMock(return_value={"x": 0, "y": 0, "z": 0})
        robot.api.get_distance = MagicMock(return_value=0)
        robot.api.get_heading = MagicMock(return_value=0)

        sensor_data = robot.get_sensor_data()

        # Verify structure
        self.assertIn("ambient_light", sensor_data)
        self.assertIn("orientation", sensor_data)
        self.assertIn("velocity", sensor_data)
        self.assertIn("location", sensor_data)
        self.assertIn("gyroscope", sensor_data)
        self.assertIn("acceleration", sensor_data)
        self.assertIn("travel_distance", sensor_data)
        self.assertIn("heading", sensor_data)

        robot.disconnect()

    def test_get_sensor_data_not_connected(self):
        """Test that get_sensor_data raises when not connected."""
        robot = SpheroBoltPlus(
            scanner_class=DummyFinder,
            api_class=DummySpheroEduAPI,  # type: ignore
            register_handlers=False
        )

        with self.assertRaises(RuntimeError) as context:
            robot.get_sensor_data()

        self.assertIn("not connected", str(context.exception))

    # -----------------------------------------------------------------
    # Display Tests
    # -----------------------------------------------------------------

    def test_display_arrow_up(self):
        """Test that display_arrow with 'up' works correctly."""
        robot = SpheroBoltPlus(
            scanner_class=DummyFinder,
            api_class=DummySpheroEduAPI,  # type: ignore
            register_handlers=False
        )

        robot.connect("DummyBolt", timeout=0.01)

        # Should not raise
        robot.display_arrow("up")

        robot.disconnect()

    def test_display_arrow_all_directions(self):
        """Test that display_arrow works for all valid directions."""
        robot = SpheroBoltPlus(
            scanner_class=DummyFinder,
            api_class=DummySpheroEduAPI,  # type: ignore
            register_handlers=False
        )

        robot.connect("DummyBolt", timeout=0.01)

        for direction in ["up", "down", "left", "right"]:
            # Should not raise
            robot.display_arrow(direction)

        robot.disconnect()

    def test_display_arrow_invalid_direction(self):
        """Test that display_arrow raises ValueError for invalid direction."""
        robot = SpheroBoltPlus(
            scanner_class=DummyFinder,
            api_class=DummySpheroEduAPI,  # type: ignore
            register_handlers=False
        )

        robot.connect("DummyBolt", timeout=0.01)

        with self.assertRaises(ValueError) as context:
            robot.display_arrow("diagonal")

        self.assertIn("Invalid direction", str(context.exception))

        robot.disconnect()

    def test_display_arrow_not_connected(self):
        """Test that display_arrow raises when not connected."""
        robot = SpheroBoltPlus(
            scanner_class=DummyFinder,
            api_class=DummySpheroEduAPI,  # type: ignore
            register_handlers=False
        )

        with self.assertRaises(RuntimeError) as context:
            robot.display_arrow("up")

        self.assertIn("not connected", str(context.exception))

    def test_display_character(self):
        """Test that display_character works correctly."""
        robot = SpheroBoltPlus(
            scanner_class=DummyFinder,
            api_class=DummySpheroEduAPI,  # type: ignore
            register_handlers=False
        )

        robot.connect("DummyBolt", timeout=0.01)

        # Should not raise
        robot.display_character("A")
        robot.display_character("X")

        robot.disconnect()

    def test_display_character_not_connected(self):
        """Test that display_character raises when not connected."""
        robot = SpheroBoltPlus(
            scanner_class=DummyFinder,
            api_class=DummySpheroEduAPI,  # type: ignore
            register_handlers=False
        )

        with self.assertRaises(RuntimeError) as context:
            robot.display_character("A")

        self.assertIn("not connected", str(context.exception))

    def test_scroll_text(self):
        """Test that scroll_text works correctly."""
        robot = SpheroBoltPlus(
            scanner_class=DummyFinder,
            api_class=DummySpheroEduAPI,  # type: ignore
            register_handlers=False
        )

        robot.connect("DummyBolt", timeout=0.01)

        # Should not raise
        robot.scroll_text("HI", fps=10, wait=False)

        robot.disconnect()

    def test_scroll_text_not_connected(self):
        """Test that scroll_text raises when not connected."""
        robot = SpheroBoltPlus(
            scanner_class=DummyFinder,
            api_class=DummySpheroEduAPI,  # type: ignore
            register_handlers=False
        )

        with self.assertRaises(RuntimeError) as context:
            robot.scroll_text("HI")

        self.assertIn("not connected", str(context.exception))

    def test_clear_display(self):
        """Test that clear_display works correctly."""
        robot = SpheroBoltPlus(
            scanner_class=DummyFinder,
            api_class=DummySpheroEduAPI,  # type: ignore
            register_handlers=False
        )

        robot.connect("DummyBolt", timeout=0.01)

        # Should not raise
        robot.clear_display()

        robot.disconnect()

    def test_clear_display_not_connected(self):
        """Test that clear_display raises when not connected."""
        robot = SpheroBoltPlus(
            scanner_class=DummyFinder,
            api_class=DummySpheroEduAPI,  # type: ignore
            register_handlers=False
        )

        with self.assertRaises(RuntimeError) as context:
            robot.clear_display()

        self.assertIn("not connected", str(context.exception))

    def test_display_bitmap(self):
        """Test that display_bitmap works correctly."""
        from rlive_world.bolt.bitmaps import ARROW_UP

        robot = SpheroBoltPlus(
            scanner_class=DummyFinder,
            api_class=DummySpheroEduAPI,  # type: ignore
            register_handlers=False
        )

        robot.connect("DummyBolt", timeout=0.01)

        # Should not raise
        robot.display_bitmap(ARROW_UP)

        robot.disconnect()

    def test_display_with_custom_color(self):
        """Test that display methods accept custom color."""
        from sphero_unsw.types import Color

        robot = SpheroBoltPlus(
            scanner_class=DummyFinder,
            api_class=DummySpheroEduAPI,  # type: ignore
            register_handlers=False
        )

        robot.connect("DummyBolt", timeout=0.01)

        custom_color = Color(255, 0, 0)  # Red

        # All should accept custom color without raising
        robot.display_arrow("up", color=custom_color)
        robot.display_character("R", color=custom_color)
        robot.scroll_text("RED", color=custom_color, wait=False)

        robot.disconnect()
