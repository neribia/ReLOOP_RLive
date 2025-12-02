"""Tests for the SpheroFinder class."""

import unittest
from unittest.mock import MagicMock

from rlive_world.bolt.sphero_finder import SpheroFinder


class FakeToy:
    """Fake toy object for testing."""

    def __init__(self, name: str) -> None:
        """Initialize fake toy with a name."""
        self.name = name


class TestSpheroFinder(unittest.TestCase):
    """Test suite for SpheroFinder class."""

    def setUp(self):
        """Set up test fixtures with fake toys."""
        self.fake_toy_x = FakeToy("Bolt-X")
        self.fake_toy_y = FakeToy("Bolt-Y")

    def test_scan_toys_returns_list(self):
        """Test that scan_toys returns a list of found toys."""
        mock_scan = MagicMock(return_value=[self.fake_toy_x, self.fake_toy_y])

        finder = SpheroFinder(scan_fn=mock_scan)

        toys = finder.scan_toys(scanning_time=5)

        mock_scan.assert_called_once_with(timeout=5)

        self.assertEqual(len(toys), 2)
        self.assertEqual(toys[0].name, "Bolt-X")
        self.assertEqual(toys[1].name, "Bolt-Y")
        self.assertEqual(finder.toys, toys)

    def test_scan_toys_empty(self):
        """Test that scan_toys handles empty result correctly."""
        mock_scan = MagicMock(return_value=[])

        finder = SpheroFinder(scan_fn=mock_scan)
        toys = finder.scan_toys(scanning_time=2)

        mock_scan.assert_called_once_with(timeout=2)
        self.assertEqual(toys, [])
        self.assertEqual(finder.toys, [])

    def test_select_toy_found(self):
        """Test that select_toy returns the correct toy when found."""
        finder = SpheroFinder(scan_fn=MagicMock())

        finder.toys = [self.fake_toy_x]

        selected = finder.select_toy("Bolt-X")

        self.assertIs(selected, self.fake_toy_x)
        self.assertEqual(finder.get_selected_toy(), self.fake_toy_x)

    # ---------------------------------------------------------

    def test_select_toy_not_found(self):
        """Test that select_toy returns None when toy not found."""
        finder = SpheroFinder(scan_fn=MagicMock())

        finder.toys = [self.fake_toy_x]

        selected = finder.select_toy("Unknown")

        self.assertIsNone(selected)
        self.assertIsNone(finder.get_selected_toy())
