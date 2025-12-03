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
        """Test that attach_hardware raises RuntimeError on hardware attachment failure."""
        # Create new mock with failed attach_hardware
        self.mock_iface.attach_hardware.return_value.success = False
        self.mock_iface.attach_hardware.return_value.info = {"error": "Hardware failure"}

        # Create a new environment with auto_attach=False
        env = RemoteWorldEnv(auto_attach=False, base_url="http://test")

        # Now trying to attach should raise RuntimeError
        with self.assertRaises(RuntimeError) as context:
            env.attach_hardware()

        self.assertIn("Failed to attach hardware", str(context.exception))

    def test_auto_attach_true(self):
        """Test that auto_attach=True calls attach_hardware on init."""
        # This is already tested in setUp, but explicit test
        self.mock_iface.attach_hardware.assert_called_once()
        self.assertTrue(self.env._hardware_attached)

    def test_auto_attach_false(self):
        """Test that auto_attach=False skips hardware attachment on init."""
        # Create new environment with auto_attach=False
        env = RemoteWorldEnv(auto_attach=False, base_url="http://test")

        # attach_hardware should only be called once from setUp (not from this new env)
        self.assertEqual(self.mock_iface.attach_hardware.call_count, 1)
        self.assertFalse(env._hardware_attached)

    def test_attach_hardware_idempotent(self):
        """Test that attach_hardware can be called multiple times safely."""
        # First call was in setUp
        self.env.attach_hardware()  # Second call should log warning

        # attach_hardware should only be called once on the mock
        self.mock_iface.attach_hardware.assert_called_once()

    def test_detach_hardware_when_not_attached(self):
        """Test that detach_hardware when not attached is safe."""
        # Create env without auto_attach
        env = RemoteWorldEnv(auto_attach=False, base_url="http://test")

        # Detach when not attached should not call server
        env.detach_hardware()

        # detach_hardware should not be called on the mock
        self.mock_iface.detach_hardware.assert_not_called()

    def test_detach_hardware_success(self):
        """Test that detach_hardware successfully detaches."""
        self.env.detach_hardware()

        self.mock_iface.detach_hardware.assert_called_once()
        self.assertFalse(self.env._hardware_attached)

    def test_detach_hardware_handles_exceptions(self):
        """Test that detach_hardware handles exceptions gracefully."""
        self.mock_iface.detach_hardware.side_effect = Exception("Network error")

        # Should not raise, just log
        self.env.detach_hardware()

        self.assertFalse(self.env._hardware_attached)

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
