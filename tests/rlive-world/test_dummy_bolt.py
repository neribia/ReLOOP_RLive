import unittest

from rlive_world.bolt.boltdummys import (
    DummyBLEDevice,
    DummyBleakAdapter,
    DummyFinder,
    DummyToy,
    DummySpheroEduAPI
)
from rlive_world.bolt.sphero_bolt_plus import SpheroBoltPlus
from rlive_world.bolt.base_robot import BaseRobot


class TestDummyBLEDevice(unittest.TestCase):
    """Test the DummyBLEDevice class."""

    def test_init_defaults(self):
        device = DummyBLEDevice()
        self.assertEqual(device.name, "DummyBolt")
        self.assertEqual(device.address, "FA:KE:00:00:01")

    def test_init_custom_values(self):
        device = DummyBLEDevice(name="CustomBolt", address="AA:BB:CC:DD:EE")
        self.assertEqual(device.name, "CustomBolt")
        self.assertEqual(device.address, "AA:BB:CC:DD:EE")


class TestDummyBleakAdapter(unittest.TestCase):
    """Test the DummyBleakAdapter class."""

    def test_scan_toys(self):
        devices = DummyBleakAdapter.scan_toys(timeout=0.1)
        self.assertIsInstance(devices, list)
        self.assertEqual(len(devices), 1)
        self.assertEqual(devices[0].name, "DummyBolt")

    def test_scan_toy_found(self):
        device = DummyBleakAdapter.scan_toy("DummyBolt")
        self.assertIsNotNone(device)
        self.assertEqual(device.name, "DummyBolt")

    def test_scan_toy_not_found(self):
        device = DummyBleakAdapter.scan_toy("NonExistent")
        self.assertIsNone(device)

    def test_adapter_connection(self):
        adapter = DummyBleakAdapter("FA:KE:00:00:01")
        self.assertTrue(adapter.connected)
        self.assertEqual(adapter.address, "FA:KE:00:00:01")
        adapter.close()

    def test_adapter_write(self):
        adapter = DummyBleakAdapter("FA:KE:00:00:01")
        # Should not raise
        adapter.write("test-uuid", b"test data")
        adapter.close()

    def test_adapter_callback(self):
        adapter = DummyBleakAdapter("FA:KE:00:00:01")
        # Should not raise
        adapter.set_callback("test-uuid", lambda uuid, data: None)
        adapter.close()


class TestDummyFinder(unittest.TestCase):
    """Test the DummyFinder class."""

    def test_init(self):
        finder = DummyFinder()
        self.assertIsInstance(finder.toys, list)
        self.assertEqual(len(finder.toys), 0)
        self.assertIsNone(finder.selected_toy)

    def test_scan_toys(self):
        finder = DummyFinder()
        toys = finder.scan_toys(timeout=0.01)
        self.assertIsInstance(toys, list)
        self.assertEqual(len(toys), 1)
        self.assertIsInstance(toys[0], DummyToy)
        self.assertEqual(toys[0].name, "DummyBolt")

    def test_select_toy_success(self):
        finder = DummyFinder()
        finder.scan_toys(timeout=0.01)
        toy = finder.select_toy("DummyBolt")
        self.assertIsNotNone(toy)
        self.assertEqual(toy.name, "DummyBolt")
        self.assertEqual(finder.selected_toy, toy)

    def test_select_toy_not_found(self):
        finder = DummyFinder()
        finder.scan_toys(timeout=0.01)
        toy = finder.select_toy("NonExistent")
        self.assertIsNone(toy)


class TestDummyToy(unittest.TestCase):
    """Test the DummyToy class."""

    def test_init(self):
        device = DummyBLEDevice("TestBolt", "TE:ST:00:00:01")
        toy = DummyToy(device)
        self.assertEqual(toy.name, "TestBolt")
        self.assertEqual(toy.address, "TE:ST:00:00:01")
        self.assertIsInstance(toy.adapter, DummyBleakAdapter)
        toy.adapter.close()


class TestDummySpheroEduAPI(unittest.TestCase):
    """Test the DummySpheroEduAPI class."""

    def test_init(self):
        device = DummyBLEDevice()
        toy = DummyToy(device)
        api = DummySpheroEduAPI(toy)
        self.assertEqual(api.toy, toy)
        self.assertIn("pitch", api.sensors)
        self.assertIn("roll", api.sensors)
        self.assertIn("yaw", api.sensors)
        toy.adapter.close()

    def test_context_manager(self):
        device = DummyBLEDevice()
        toy = DummyToy(device)
        api = DummySpheroEduAPI(toy)

        with api:
            # Should not raise
            pass

        toy.adapter.close()

    def test_roll(self):
        device = DummyBLEDevice()
        toy = DummyToy(device)
        api = DummySpheroEduAPI(toy)

        with api:
            # Should not raise
            api.roll(heading=90, speed=100, duration=2)

        toy.adapter.close()

    def test_get_orientation(self):
        device = DummyBLEDevice()
        toy = DummyToy(device)
        api = DummySpheroEduAPI(toy)

        orientation = api.get_orientation()
        self.assertIn("pitch", orientation)
        self.assertIn("roll", orientation)
        self.assertIn("yaw", orientation)

        toy.adapter.close()

    def test_get_acceleration(self):
        device = DummyBLEDevice()
        toy = DummyToy(device)
        api = DummySpheroEduAPI(toy)

        acceleration = api.get_acceleration()
        self.assertIn("x", acceleration)
        self.assertIn("y", acceleration)
        self.assertIn("z", acceleration)

        toy.adapter.close()


class TestSpheroBoltPlusWithDummies(unittest.TestCase):
    """Test SpheroBoltPlus with dummy components."""

    def setUp(self):
        # Create a fresh robot before each test
        self.robot = SpheroBoltPlus(scanner_class=DummyFinder, api_class=DummySpheroEduAPI)  # type: ignore

    def tearDown(self):
        # Make sure we clean up if a test fails
        self.robot.disconnect()

    def test_init(self):
        self.assertIsInstance(self.robot, BaseRobot)
        self.assertEqual(self.robot.heading, 0)
        self.assertIsNone(self.robot.name)
        self.assertIsNone(self.robot.api)

    def test_connect(self):
        self.robot.connect("DummyBolt", timeout=0.01)

        self.assertEqual(self.robot.name, "DummyBolt")
        self.assertIsNotNone(self.robot.api)
        self.assertIsNotNone(self.robot.toy)

        self.robot.disconnect()

    def test_connect_not_found(self):
        with self.assertRaises(RuntimeError) as context:
            self.robot.connect("NonExistentBolt")

        self.assertIn("not found", str(context.exception))

    def test_move(self):
        self.robot.connect("DummyBolt", timeout=0.01)

        # Should not raise
        self.robot.move(heading=90, speed=100, duration=1)
        self.assertEqual(self.robot.heading, 90)

        self.robot.move(heading=90, speed=100, duration=1)
        self.assertEqual(self.robot.heading, 180)

        self.robot.disconnect()

    def test_move_not_connected(self):

        with self.assertRaises(RuntimeError) as context:
            self.robot.move(heading=90, speed=50, duration=1)

        self.assertIn("not connected", str(context.exception))

    def test_disconnect(self):
        self.robot.connect("DummyBolt", timeout=0.01)
        self.robot.disconnect()

        self.assertIsNone(self.robot.api)

    def test_disconnect_when_not_connected(self):
        self.robot = SpheroBoltPlus(scanner_class=DummyFinder, api_class=DummySpheroEduAPI)  # type: ignore
        # Should not raise
        self.robot.disconnect()


class TestBaseRobot(unittest.TestCase):
    """Test the BaseRobot abstract class."""

    def test_abstract_methods(self):
        # Cannot instantiate abstract class
        with self.assertRaises(TypeError):
            BaseRobot()

    def test_subclass_must_implement_methods(self):
        class IncompleteRobot(BaseRobot):
            def connect(self):
                pass

            def disconnect(self):
                pass

        with self.assertRaises(TypeError):
            IncompleteRobot()