import unittest
from unittest.mock import MagicMock, patch

from rlive_world.bolt.sphero_bolt_plus import SpheroBoltPlus


class TestBolt(unittest.TestCase):

    @patch("rlive_world.bolt.sphero_bolt_plus.SpheroEduAPI", autospec=True)
    def test_connect(self, MockAPI):

        # Fake scanner
        fake_scanner = MagicMock()
        fake_toy = MagicMock()
        fake_toy.name = "FakeBolt"

        fake_scanner.scan_toys.return_value = None
        fake_scanner.select_toy.return_value = fake_toy

        # Fake API
        mock_api = MockAPI.return_value
        mock_api.__enter__.return_value = mock_api

        # Create robot with injected scanner
        robot = SpheroBoltPlus(
            scanner=fake_scanner,
            api=None,                  # will be created inside connect()
            register_handlers=False
        )

        robot.connect("FakeBolt")

        # Assertions
        fake_scanner.scan_toys.assert_called_once()
        fake_scanner.select_toy.assert_called_once_with("FakeBolt")
        MockAPI.assert_called_once_with(fake_toy)
        mock_api.__enter__.assert_called_once()

        self.assertEqual(robot.name, "FakeBolt")
        self.assertIs(robot.api, mock_api)

    # ---------------------------------------------------------------------

    def test_disconnect_calls_cleanup_and_closes_api(self):
        robot = SpheroBoltPlus(
            scanner=MagicMock(),
            api=MagicMock(),
            register_handlers=False
        )

        fake_api = robot.api

        # Call disconnect
        robot.disconnect()

        # __exit__ should have been called
        fake_api.__exit__.assert_called_once_with(None, None, None)

        # API must be cleared
        self.assertIsNone(robot.api)

    # ---------------------------------------------------------------------

    def test_require_connection_not_connected(self):
        robot = SpheroBoltPlus(
            scanner=MagicMock(),
            register_handlers=False
        )

        with self.assertRaises(RuntimeError):
            robot._require_connection()

    # ---------------------------------------------------------------------

    def test_require_connection_connected(self):
        robot = SpheroBoltPlus(
            scanner=MagicMock(),
            register_handlers=False
        )

        robot.api = MagicMock()  # simulate connection
        robot._require_connection()  # should NOT raise

    # ---------------------------------------------------------------------

    @patch("rlive_world.bolt.sphero_bolt_plus.SpheroEduAPI", autospec=True)
    def test_move(self, MockAPI):
        fake_scanner = MagicMock()
        fake_toy = MagicMock()
        fake_toy.name = "FakeBolt"

        fake_scanner.scan_toys.return_value = None
        fake_scanner.select_toy.return_value = fake_toy

        fake_api = MockAPI.return_value
        fake_api.__enter__.return_value = fake_api

        robot = SpheroBoltPlus(
            scanner=fake_scanner,
            api=None,
            register_handlers=False
        )

        robot.connect("FakeBolt")
        robot.move(heading=90, speed=100, duration=1)

        fake_api.roll.assert_called_once_with(90, 100, 1)
