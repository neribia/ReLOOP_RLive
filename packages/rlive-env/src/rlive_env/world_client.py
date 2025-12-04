"""HTTP client interface for communicating with the World server."""

from typing import Any, Optional
import time

import httpx
from httpx import Response

from rlive_world.config import config as cfg
from rlive_common.core.response import (
    ResetResponse,
    StepResponseJSON,
    StepResponseMultipart,
    AttachHardwareResponse,
    DetachHardwareResponse,
)
from rlive_common.core.request import ResetRequest, StepRequest, AttachHardwareRequest, DetachHardwareRequest
from rlive_env.config import config as cfg
from rlive_common.utils import get_logger

logger = get_logger(__name__)

# FIXME: Shutdown if the server throws an error and maybe print that error

class ApiError(Exception):
    def __init__(self, method, path, status, message):
        self.method = method
        self.path = path
        self.status = status
        self.message = message
        logger.debug("ApiError information:")
        logger.debug(f"method: {method}, path: {path}, status: {status}, message: {message}")
        super().__init__(f"{status} {method} {path}: {message}")



class WorldInterface:
    """
    Synchronous HTTP interface to the remote World server.

    Uses httpx.Client with retry/backoff for robustness.

    Example:
        iface = WorldInterface()
        obs = iface.reset()
        out = iface.step(action)
        iface.close()
    """

    def __init__(
            self,
            base_url: str = cfg.WORLD_BASE_URL,
            timeout: float = cfg.WORLD_INTERFACE_TIMEOUT,
            max_retries: int = cfg.WORLD_INTERFACE_MAX_RETRIES,
            backoff_factor: float = cfg.WORLD_INTERFACE_BACKOFF_FACTOR,
            max_retry_time: float = cfg.WORLD_INTERFACE_MAX_RETRIES_TIME,
            client: httpx.Client | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_retry_time = max_retry_time
        self.backoff_factor = backoff_factor

        # httpx.Client synchronous
        self._client = client or httpx.Client(timeout=self.timeout)

    # -------------------------------------------------------------

    def close(self) -> None:
        """Close the underlying httpx client."""
        self._client.close()

    # -------------------------------------------------------------

    def health_check(self) -> dict:
        """Check if the world server is healthy and responding."""
        try:
            return self._request("GET", "/health", expect_json=True)
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {"status": "unhealthy", "error": str(e)}

    def get_status(self) -> dict:
        """Get detailed status of the world server including hardware state."""
        try:
            return self._request("GET", "/status", expect_json=True)
        except Exception as e:
            logger.error(f"Status check failed: {e}")
            return {"error": str(e)}

    # -------------------------------------------------------------
    def attach_hardware(self, **kwargs) -> AttachHardwareResponse:
        """Call POST /attach_hardware on the world server with retries."""
        payload = AttachHardwareRequest(**kwargs).model_dump()
        data = self._request("POST", "/attach_hardware", json=payload)
        return AttachHardwareResponse(**data)

    def detach_hardware(self) -> DetachHardwareResponse:
        """Call POST /detach_hardware on the world server with retries."""
        payload = DetachHardwareRequest().model_dump()
        data = self._request("POST", "/detach_hardware", json=payload)
        return DetachHardwareResponse(**data)

    def reset(self) -> ResetResponse:
        """Call POST /reset on the world server with retries."""
        payload = ResetRequest().model_dump()
        data = self._request("POST", "/reset", json=payload)
        return ResetResponse(**data)

    def step_json(self, action: int) -> StepResponseJSON:
        """Call POST /step_json and decode NumPy image."""
        payload = StepRequest(action=action).model_dump()
        data = self._request("POST", "/step_json", json=payload)
        return StepResponseJSON(**data)

    def step_multipart(self, action: int) -> StepResponseMultipart:
        """Call POST /step_multipart and decode multipart/mixed response."""
        payload = StepRequest(action=action).model_dump()
        response: Response = self._request("POST", "/step_multipart", json=payload, expect_json=False)

        content_type = response.headers.get("content-type", "")
        return StepResponseMultipart.decode(response.content, content_type)

    # -------------------------------------------------------------

    def _send_once(self, method: str, path: str, expect_json: bool = True, **kwargs) -> dict[str, Any] | httpx.Response:
        """Send a single HTTP request and return parsed JSON."""
        url = f"{self.base_url}{path}"
        response = self._client.request(method, url, **kwargs)
        response.raise_for_status()

        if expect_json:
            try:
                return response.json()
            except ValueError:
                logger.error(f"Invalid JSON from {url}: {response.text[:200]}")
                raise
        else:
            # Return the full response object for manual decoding
            return response

    def _request(self, method: str, path: str, expect_json: bool = True, **kwargs):
        for attempt in range(self.max_retries + 1):
            try:
                return self._send_once(method, path, expect_json, **kwargs)

            except httpx.HTTPStatusError as e:
                retry = self._handle_http_error(method, path, e, attempt)
                if not retry:
                    raise

                time.sleep(self.backoff_factor * (2 ** attempt))

            except httpx.RequestError as e:  # network errors → retry
                if attempt >= self.max_retries:
                    raise ApiError(method, path, None, str(e))

                time.sleep(self.backoff_factor * (2 ** attempt))

        raise RuntimeError("Unreachable")

    def _handle_http_error(self, method, path, exc, attempt):
        status = exc.response.status_code
        detail = self.parse_error(exc.response)

        # no retry for server logic errors
        if not self.is_retryable_status(status):
            raise ApiError(method, path, status, detail)

        # retryable: only if attempts left
        if attempt >= self.max_retries:
            raise ApiError(method, path, status, detail)

        # return delay (so caller can sleep)
        return True  # meaning: "retry"

    @staticmethod
    def parse_error(response):
        try:
            data = response.json()
        except Exception:
            return response.text
        return data.get("message", data)

    @staticmethod
    def is_retryable_status(status: int) -> bool:
        return status in {502, 503, 504}





