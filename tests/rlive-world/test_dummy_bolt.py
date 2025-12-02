import unittest
from unittest.mock import MagicMock, patch

from rlive_world.bolt.dummy_bolt import DummyBolt, FakeBLEDevice
from rlive_world.bolt.base_robot import BaseRobot


class TestFakeBLEDevice(unittest.TestCase):
    def test_init_defaults(self):
        device = FakeBLEDevice()
        self.assertEqual(device.name, "DummyBolt")
        self.assertEqual(device.address, "FA:KE:DE:VI:CE")

    def test_init_custom_values(self):
        device = FakeBLEDevice(name="CustomBolt", address="AA:BB:CC:DD:EE")
        self.assertEqual(device.name, "CustomBolt")
        self.assertEqual(device.address, "AA:BB:CC:DD:EE")


class TestDummyBolt(unittest.TestCase):
    def test_init(self):
        bolt = DummyBolt()
        self.assertFalse(bolt.connected)
        self.assertIsNone(bolt.device)
        self.assertIsNone(bolt.name)
        self.assertEqual(bolt.heading, 0)
        self.assertEqual(bolt.move_delay, 0.1)

    def test_is_base_robot(self):
        bolt = DummyBolt()
        self.assertIsInstance(bolt, BaseRobot)

    @patch('rlive_world.bolt.dummy_bolt.asyncio.run')
    def test_connect_successful(self, mock_asyncio_run):
        # First call for fake_scan, second for fake_connect
        mock_asyncio_run.side_effect = [
            FakeBLEDevice(name="TestBolt", address="TE:ST:AD:DR:ES"),
            True
        ]

        bolt = DummyBolt()
        result = bolt.connect()

        self.assertTrue(result)
        self.assertTrue(bolt.connected)
        self.assertIsNotNone(bolt.device)
        self.assertEqual(bolt.device.name, "TestBolt")

    @patch('rlive_world.bolt.dummy_bolt.asyncio.run')
    def test_connect_failed(self, mock_asyncio_run):
        mock_asyncio_run.side_effect = [
            FakeBLEDevice(),
            False  # Connection fails
        ]

        bolt = DummyBolt()
        result = bolt.connect()

        self.assertFalse(result)
        self.assertFalse(bolt.connected)

    @patch('rlive_world.bolt.dummy_bolt.asyncio.run')
    def test_move_when_connected(self, mock_asyncio_run):
        bolt = DummyBolt()
        bolt.connected = True  # Simulate connected state

        bolt.move(heading=90, speed=50, duration=2)

        # Verify asyncio.run was called for fake_move
        mock_asyncio_run.assert_called_once()

    @patch('rlive_world.bolt.dummy_bolt.asyncio.run')
    def test_move_when_not_connected(self, mock_asyncio_run):
        bolt = DummyBolt()
        bolt.connected = False

        bolt.move(heading=90, speed=50, duration=2)

        # Should not call asyncio.run because not connected
        mock_asyncio_run.assert_not_called()

    @patch('rlive_world.bolt.dummy_bolt.asyncio.run')
    def test_disconnect_when_connected(self, mock_asyncio_run):
        bolt = DummyBolt()
        bolt.connected = True
        bolt.device = FakeBLEDevice()

        bolt.disconnect()

        self.assertFalse(bolt.connected)
        mock_asyncio_run.assert_called_once()

    def test_disconnect_when_not_connected(self):
        bolt = DummyBolt()
        bolt.connected = False

        bolt.disconnect()  # Should not raise

        self.assertFalse(bolt.connected)


class TestBaseRobot(unittest.TestCase):
    def test_abstract_methods(self):
        # Cannot instantiate abstract class
        with self.assertRaises(TypeError):
            BaseRobot()

    def test_subclass_must_implement_methods(self):
        class IncompleteRobot(BaseRobot):
            pass

        with self.assertRaises(TypeError):
            IncompleteRobot()


if __name__ == "__main__":
    unittest.main()
