"""Tests for the SpheroBoltPlus robot class."""

import unittest
from unittest.mock import MagicMock, call

from rlive_world.bolt.sphero_bolt_plus import SpheroBoltPlus


class TestBolt(unittest.TestCase):
    """Test suite for SpheroBoltPlus robot."""

    def test_connect(self):
        """Test that robot connects properly using the scanner."""
        # Fake scanner
        fake_scanner = MagicMock()
        fake_toy = MagicMock()
        fake_toy.name = "FakeBolt"

        fake_scanner.scan_toys.return_value = None
        fake_scanner.select_toy.return_value = fake_toy

        # Fake API class
        mock_api_class = MagicMock()
        mock_api = MagicMock()
        mock_api_class.return_value = mock_api
        mock_api.__enter__ = MagicMock(return_value=mock_api)
        mock_api.__exit__ = MagicMock(return_value=None)

        # Create robot with injected scanner and api_class
        robot = SpheroBoltPlus(
            scanner=fake_scanner,
            api_class=mock_api_class,
            register_handlers=False
        )

        robot.connect("FakeBolt")

        # Assertions
        fake_scanner.scan_toys.assert_called_once()
        fake_scanner.select_toy.assert_called_once_with("FakeBolt")
        mock_api_class.assert_called_once_with(fake_toy)
        mock_api.__enter__.assert_called_once()

        self.assertEqual(robot.name, "FakeBolt")
        self.assertIs(robot.api, mock_api)

    # ---------------------------------------------------------------------

    def test_disconnect_calls_cleanup_and_closes_api(self):
        """Test that disconnect calls cleanup and closes the API."""
        robot = SpheroBoltPlus(
            scanner=MagicMock(),
            register_handlers=False
        )

        # Set the api after initialization
        fake_api = MagicMock()
        robot.api = fake_api

        # Call disconnect
        robot.disconnect()

        # __exit__ should have been called
        fake_api.__exit__.assert_called_once_with(None, None, None)

        # API must be cleared
        self.assertIsNone(robot.api)

    # ---------------------------------------------------------------------

    def test_require_connection_not_connected(self):
        """Test that _require_connection raises when not connected."""
        robot = SpheroBoltPlus(
            scanner=MagicMock(),
            register_handlers=False
        )

        with self.assertRaises(RuntimeError):
            robot._require_connection()

    # ---------------------------------------------------------------------

    def test_require_connection_connected(self):
        """Test that _require_connection passes when connected."""
        robot = SpheroBoltPlus(
            scanner=MagicMock(),
            register_handlers=False
        )

        robot.api = MagicMock()  # simulate connection
        robot._require_connection()  # should NOT raise

    # ---------------------------------------------------------------------

    def test_move(self):
        """Test that robot move command works correctly."""
        fake_scanner = MagicMock()
        fake_toy = MagicMock()
        fake_toy.name = "FakeBolt"

        fake_scanner.scan_toys.return_value = None
        fake_scanner.select_toy.return_value = fake_toy

        # Fake API class
        mock_api_class = MagicMock()
        mock_api = MagicMock()
        mock_api_class.return_value = mock_api
        mock_api.__enter__ = MagicMock(return_value=mock_api)
        mock_api.__exit__ = MagicMock(return_value=None)

        robot = SpheroBoltPlus(
            scanner=fake_scanner,
            api_class=mock_api_class,
            register_handlers=False
        )

        robot.connect("FakeBolt")
        robot.move(heading=90, speed=100, duration=1)

        mock_api.roll.assert_has_calls([
            call(90, 0, 1),
            call(90, 100, 1),
        ])
