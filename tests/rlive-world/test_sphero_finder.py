import unittest
from unittest.mock import MagicMock

from rlive_world.bolt.sphero_finder import SpheroFinder


class FakeToy:
    def __init__(self, name):
        self.name = name


class TestSpheroFinder(unittest.TestCase):
    def setUp(self):
        self.fake_toy_x = FakeToy("Bolt-X")
        self.fake_toy_y = FakeToy("Bolt-Y")

    def test_scan_toys_returns_list(self):
        mock_scan = MagicMock(return_value=[self.fake_toy_x, self.fake_toy_y])

        finder = SpheroFinder(scan_fn=mock_scan)

        toys = finder.scan_toys(scanning_time=5)

        mock_scan.assert_called_once_with(timeout=5)

        self.assertEqual(len(toys), 2)
        self.assertEqual(toys[0].name, "Bolt-X")
        self.assertEqual(toys[1].name, "Bolt-Y")
        self.assertEqual(finder.toys, toys)

    def test_scan_toys_empty(self):
        mock_scan = MagicMock(return_value=[])

        finder = SpheroFinder(scan_fn=mock_scan)
        toys = finder.scan_toys(scanning_time=2)

        mock_scan.assert_called_once_with(timeout=2)
        self.assertEqual(toys, [])
        self.assertEqual(finder.toys, [])

    def test_select_toy_found(self):
        finder = SpheroFinder(scan_fn=MagicMock())

        finder.toys = [self.fake_toy_x]

        selected = finder.select_toy("Bolt-X")

        self.assertIs(selected, self.fake_toy_x)
        self.assertEqual(finder.get_selected_toy(), self.fake_toy_x)

    # ---------------------------------------------------------

    def test_select_toy_not_found(self):
        finder = SpheroFinder(scan_fn=MagicMock())

        finder.toys = [self.fake_toy_x]

        selected = finder.select_toy("Unknown")

        self.assertIsNone(selected)
        self.assertIsNone(finder.get_selected_toy())
