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
        self.mock_iface.health_check.return_value = {"status": "healthy"}
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

    def test_health_check_success(self):
        """Test that health_check is called during _connect and logs success."""
        # Create a fresh mock for this test
        with patch("rlive_env.remote_env.WorldInterface") as MockWorldInterface, \
             patch("rlive_env.remote_env.logger") as mock_logger:
            mock_iface = MockWorldInterface.return_value
            mock_iface.health_check.return_value = {"status": "healthy", "version": "1.0.0"}
            mock_iface.attach_hardware.return_value = MagicMock(success=True)
            
            # Create environment to trigger _connect
            env = RemoteWorldEnv(auto_attach=False, base_url="http://test")
            
            # Verify health_check was called
            mock_iface.health_check.assert_called_once()
            
            # Verify success message was logged
            mock_logger.info.assert_any_call("Server health check passed: {'status': 'healthy', 'version': '1.0.0'}")

    def test_health_check_failure(self):
        """Test that health_check failures are handled gracefully with warning."""
        # Create a fresh mock for this test
        with patch("rlive_env.remote_env.WorldInterface") as MockWorldInterface, \
             patch("rlive_env.remote_env.logger") as mock_logger:
            mock_iface = MockWorldInterface.return_value
            mock_iface.health_check.side_effect = Exception("Connection timeout")
            mock_iface.attach_hardware.return_value = MagicMock(success=True)
            
            # Create environment to trigger _connect
            env = RemoteWorldEnv(auto_attach=False, base_url="http://test")
            
            # Verify health_check was called
            mock_iface.health_check.assert_called_once()
            
            # Verify warning was logged but execution continued
            mock_logger.warning.assert_any_call("Server health check failed: Connection timeout")
            
            # Verify environment was still created successfully
            self.assertIsNotNone(env.iface)

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

    def test_reset_with_status_check_success(self):
        """Test that reset proceeds normally when get_status succeeds."""
        # Mock get_status to return successfully
        self.mock_iface.get_status.return_value = {"status": "ok"}
        
        mock_data = MagicMock()
        mock_data.observation = np.array([1, 2, 3])
        mock_data.info = {"meta": "test"}
        self.mock_iface.reset.return_value = mock_data

        obs, info = self.env.reset(seed=42, options={})

        # Verify get_status was called
        self.mock_iface.get_status.assert_called_once()
        # Verify reset proceeded normally
        self.mock_iface.reset.assert_called_once()
        np.testing.assert_array_equal(obs, np.array([1, 2, 3]))
        self.assertEqual(info, {"meta": "test"})

    def test_reset_with_status_check_failure(self):
        """Test that reset raises RuntimeError and disconnects when get_status fails."""
        # Mock get_status to raise an exception
        self.mock_iface.get_status.side_effect = Exception("Connection lost")

        with self.assertRaises(RuntimeError) as context:
            self.env.reset()

        # Verify error message
        self.assertIn("Server connection error", str(context.exception))
        self.assertIn("Connection lost", str(context.exception))
        
        # Verify _disconnect was called (detach_hardware and close)
        self.mock_iface.detach_hardware.assert_called()
        self.mock_iface.close.assert_called()
        
        # Verify reset was not called
        self.mock_iface.reset.assert_not_called()

    def test_step(self):
        """Test that step returns correct gymnasium tuple."""
        mock_data = MagicMock()
        mock_data.observation = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_data.truncated = False
        mock_data.info = {"step": "ok"}

        self.mock_iface.step_json.return_value = mock_data

        obs, reward, terminated, truncated, info = self.env.step(action=0)

        # Check observation shape and type (not exact values, as _draw_goal modifies it)
        self.assertEqual(obs.shape, mock_data.observation.shape)
        self.assertEqual(obs.dtype, mock_data.observation.dtype)
        self.assertEqual(reward, 0.0)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        # Info includes original data plus episode tracking
        self.assertEqual(info["step"], "ok")
        self.assertIn("episode", info)
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

    def test_render_opencv_mode(self):
        """Test that render works in opencv mode."""
        env = RemoteWorldEnv(auto_attach=False, base_url="http://test", render_mode="opencv")
        env.obs = np.zeros((480, 640, 3), dtype=np.uint8)

        # Should not raise (opencv window operations are mocked/handled)
        with patch("cv2.imshow"), patch("cv2.waitKey"):
            env.render()

    def test_render_none_mode(self):
        """Test that render does nothing when render_mode is None."""
        self.env.render_mode = None
        self.env.obs = np.zeros((480, 640, 3), dtype=np.uint8)

        # Should not raise and not display anything
        self.env.render()

    def test_calculate_reward(self):
        """Test that calculate_reward returns expected values."""
        obs = np.zeros((480, 640, 3), dtype=np.uint8)

        terminated, reward = self.env.calculate_reward(observation=obs)

        self.assertFalse(terminated)
        self.assertEqual(reward, 0.0)

    def test_set_random_goal(self):
        """Test that set_random_goal sets a valid goal position."""
        self.env.set_random_goal()

        self.assertIsNotNone(self.env.goal_position)
        x, y = self.env.goal_position

        # Goal should be within observation space bounds
        height, width, _ = self.env.observation_space.shape
        self.assertGreaterEqual(x, 0)
        self.assertLess(x, width)
        self.assertGreaterEqual(y, 0)
        self.assertLess(y, height)

    def test_draw_goal(self):
        """Test that _draw_goal modifies the image."""
        self.env.goal_position = (320, 240)  # Center of 640x480 image
        original_image = np.zeros((480, 640, 3), dtype=np.uint8)

        result = self.env._draw_goal(original_image)

        # Result should be different from input (goal was drawn)
        self.assertEqual(result.shape, original_image.shape)
        self.assertEqual(result.dtype, original_image.dtype)

    def test_step_truncates_on_max_episodes(self):
        """Test that step returns truncated=True when max_episode_steps reached."""
        mock_data = MagicMock()
        mock_data.observation = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_data.truncated = False
        mock_data.info = {}
        self.mock_iface.step_json.return_value = mock_data

        # Set episode at max (truncation check happens before increment)
        self.env._max_episode_steps = 5
        self.env._episode = 5  # Already at max, so should truncate

        _, _, _, truncated, _ = self.env.step(action=0)

        self.assertTrue(truncated)

    def test_reset_sets_goal(self):
        """Test that reset sets a random goal."""
        mock_data = MagicMock()
        mock_data.observation = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_data.info = {}
        self.mock_iface.reset.return_value = mock_data

        self.env.reset()

        self.assertIsNotNone(self.env.goal_position)

