import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from rlive_env.remote_env import RemoteWorldEnv


class TestRemoteWorldEnv(unittest.TestCase):
    """Test suite for RemoteWorldEnv gymnasium environment."""

    def setUp(self):
        """Set up test fixtures with mocked WorldInterface."""
        # Patch WorldInterface before creating env
        self.patcher = patch("rlive_env.remote_env.WorldInterface")
        self.MockWorldInterface = self.patcher.start()
        self.addCleanup(self.patcher.stop)

        self.mock_iface = self.MockWorldInterface.return_value
        self.mock_iface.attach_hardware.return_value = MagicMock(success=True)
        self.mock_iface.detach_hardware.return_value = MagicMock(success=True)

        # NOW create environment
        self.env = RemoteWorldEnv(base_url="http://test", timeout=20.0)

    def test_init(self):
        """Test that initialization passes parameters correctly."""
        self.MockWorldInterface.assert_called_once_with(base_url="http://test", timeout=20.0)

    def test_connect(self):
        """Test that _connect is called during initialization."""
        self.mock_iface.attach_hardware.assert_called_once()

    def test_connect_failed(self):
        """Test that _connect raises RuntimeError on hardware attachment failure."""
        self.mock_iface.attach_hardware.return_value.success = False
        with self.assertRaises(RuntimeError):
            self.env._connect()

    def test_disconnect(self):
        """Test that _disconnect calls detach and close."""
        self.env._disconnect()
        self.mock_iface.detach_hardware.assert_called_once()
        self.mock_iface.close.assert_called_once()

    def test_reset(self):
        """Test that reset returns observation and info."""
        mock_data = MagicMock()
        mock_data.observation = np.array([1, 2, 3])
        mock_data.info = {"meta": "test"}
        self.mock_iface.reset.return_value = mock_data

        obs, info = self.env.reset(seed=42, options={})

        np.testing.assert_array_equal(obs, np.array([1, 2, 3]))
        self.assertEqual(info, {"meta": "test"})
        self.mock_iface.reset.assert_called_once()

    def test_step(self):
        """Test that step returns correct gymnasium tuple."""
        mock_data = MagicMock()
        mock_data.observation = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_data.truncated = False
        mock_data.info = {"step": "ok"}

        self.mock_iface.step_json.return_value = mock_data

        obs, reward, terminated, truncated, info = self.env.step(action=0)

        np.testing.assert_array_equal(obs, mock_data.observation)
        self.assertEqual(reward, 0.0)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info, {"step": "ok"})
        self.mock_iface.step_json.assert_called_once_with(0)

    def test_close(self):
        """Test that close disconnects the environment."""
        self.env.close()
        self.mock_iface.detach_hardware.assert_called_once()
        self.mock_iface.close.assert_called_once()

    def test_close_idempotent(self):
        """Test that close can be called multiple times safely."""
        self.env.close()
        self.env.close()  # should not crash

        self.mock_iface.detach_hardware.assert_called_once()
        self.mock_iface.close.assert_called_once()
