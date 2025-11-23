from typing import Any, Optional
import time

import httpx
from httpx import Response

from rlive_world.config import config as cfg
from rlive_common.core.response import ResetResponse, StepResponseJSON, StepResponseMultipart, AttachHardwareResponse, DetachHardwareResponse
from rlive_common.core.request import ResetRequest, StepRequest, AttachHardwareRequest, DetachHardwareRequest
from rlive_env.config import config as cfg
from rlive_common.utils import get_logger

logger = get_logger(__name__)


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
            base_url: Optional[str] = cfg.WORLD_BASE_URL,
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
    def attach_hardware(self) -> AttachHardwareResponse:
        """Call POST /attach_hardware on the world server with retries."""
        payload = AttachHardwareRequest().model_dump()
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
        """
        Call POST /step_multipart and decode multipart/mixed response.
        """
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
        """Retry wrapper with exponential backoff."""
        start = time.monotonic()

        for attempt in range(self.max_retries + 1):
            try:
                return self._send_once(method, path, expect_json, **kwargs)

            except (httpx.RequestError, httpx.HTTPStatusError) as e:
                elapsed = time.monotonic() - start
                if elapsed > self.max_retry_time:
                    logger.error(f"Retry timeout exceeded after {elapsed:.2f}s")
                    raise

                if attempt >= self.max_retries:
                    logger.error(f"Request failed after {self.max_retries} retries: {e}")
                    raise

                delay = self.backoff_factor * (2 ** attempt)
                logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.2f}s...")
                time.sleep(delay)

            except Exception:
                logger.exception("Unexpected non-HTTP error during request, not retrying")
                raise

        raise RuntimeError("Unreachable")
