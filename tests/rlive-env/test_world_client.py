import unittest
from unittest.mock import MagicMock, patch
import httpx

from rlive_common.core.response import ResetResponse, StepResponseJSON, StepResponseMultipart
from rlive_env.world_client import WorldInterface  # replace with actual import path


class TestWorldInterface(unittest.TestCase):

    def setUp(self):
        # Patch httpx.Client so we can inject a fake client
        patcher = patch("httpx.Client")
        self.addCleanup(patcher.stop)
        self.mock_client_class = patcher.start()
        self.mock_client = MagicMock()
        self.mock_client_class.return_value = self.mock_client

        self.iface = WorldInterface(base_url="http://test")

    def test_send_once_success_json(self):
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"ok": True}
        self.mock_client.request.return_value = mock_response

        result = self.iface._send_once("GET", "/path")
        self.assertEqual(result, {"ok": True})
        self.mock_client.request.assert_called_once_with("GET", "http://test/path")

    def test_send_once_invalid_json(self):
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.side_effect = ValueError("bad json")
        mock_response.text = "not-json"
        self.mock_client.request.return_value = mock_response

        with self.assertRaises(ValueError):
            self.iface._send_once("GET", "/path")

    def test_request_retries_and_succeeds(self):
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"ok": True}

        self.mock_client.request.side_effect = [
            httpx.RequestError("boom"),
            mock_response,
        ]

        result = self.iface._request("GET", "/path")
        self.assertEqual(result, {"ok": True})
        self.assertEqual(self.mock_client.request.call_count, 2)

    def test_request_exceeds_retries(self):
        self.mock_client.request.side_effect = httpx.RequestError("fail")

        with self.assertRaises(httpx.RequestError):
            self.iface._request("GET", "/path")

    @patch("time.sleep", return_value=None)
    def test_request_backoff(self, mock_sleep):
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"ok": True}

        self.mock_client.request.side_effect = [
            httpx.RequestError("fail1"),
            httpx.RequestError("fail2"),
            mock_response,
        ]

        result = self.iface._request("GET", "/path")
        self.assertEqual(result, {"ok": True})
        self.assertTrue(mock_sleep.called)

    def test_reset_returns_resetresponse(self):
        self.mock_client.request.return_value = MagicMock(
            raise_for_status=lambda: None,
            json=lambda: {"observation": [1, 2, 3], "truncated": False, "info": {}},
        )
        result = self.iface.reset()
        self.assertIsInstance(result, ResetResponse)

    def test_step_json_returns_stepresponsejson(self):
        self.mock_client.request.return_value = MagicMock(
            raise_for_status=lambda: None,
            json=lambda: {"observation": [1, 2, 3], "truncated": False, "info": {}, "image": None},
        )
        result = self.iface.step_json(action=5)
        self.assertIsInstance(result, StepResponseJSON)

    def test_step_multipart_returns_stepresponsemultipart(self):
        fake_response = MagicMock()
        fake_response.raise_for_status.return_value = None
        fake_response.headers = {"content-type": "multipart/mixed; boundary=world-step"}
        fake_response.content = (
            b"--world-step\r\nContent-Type: application/json\r\n\r\n"
            b'{"observation":[1,2,3],"truncated":false,"info":{}}'
            b"\r\n--world-step--\r\n"
        )
        self.mock_client.request.return_value = fake_response

        result = self.iface.step_multipart(action=5)
        self.assertIsInstance(result, StepResponseMultipart)
