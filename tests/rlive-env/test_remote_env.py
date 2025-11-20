import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from rlive_env.remote_env import RemoteWorldEnv


class TestRemoteWorldEnv(unittest.TestCase):

    @patch("rlive_env.remote_env.WorldInterface")
    def test_reset(self, MockWorldInterface):
        mock_iface = MockWorldInterface.return_value


        mock_data = MagicMock()
        mock_data.observation = np.array([1, 2, 3])
        mock_data.info = {"meta": "test"}

        mock_iface.reset.return_value = mock_data

        env = RemoteWorldEnv(base_url="http://test")
        obs, info = env.reset(seed=42, options={})

        np.testing.assert_array_equal(obs, np.array([1, 2, 3]))
        self.assertEqual(info, {"meta": "test"})
        mock_iface.reset.assert_called_once()

    @patch("rlive_env.remote_env.WorldInterface")
    def test_step(self, MockWorldInterface):
        mock_iface = MockWorldInterface.return_value

        mock_data = MagicMock()
        mock_data.observation = np.array([5, 6])
        mock_data.truncated = False
        mock_data.info = {"step": "ok"}
        mock_data.image = np.zeros((2, 2, 3), dtype=np.uint8)

        mock_iface.step_json.return_value = mock_data

        env = RemoteWorldEnv(base_url="http://test")
        obs, reward, terminated, truncated, info = env.step(action=0)

        np.testing.assert_array_equal(obs, np.array([5, 6]))
        self.assertEqual(reward, 0.0)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info, {"step": "ok"})
        mock_iface.step_json.assert_called_once_with(0)

    @patch("rlive_env.remote_env.WorldInterface")
    def test_close(self, MockWorldInterface):
        mock_iface = MockWorldInterface.return_value

        env = RemoteWorldEnv(base_url="http://test")
        env.close()

        mock_iface.close.assert_called_once()
