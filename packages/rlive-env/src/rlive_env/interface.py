from typing import Any, Dict, Optional
import time

import httpx

from rlive_common.core.models import (
    ResetResponse,
    ResetRequest,
    StepRequest,
    StepResponseJSON,
    StepResponseMultipart,
)
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
        base_url: Optional[str] = None,
        timeout: float = 5.0,
        max_retries: int = 3,
        backoff_factor: float = 0.3,
    ) -> None:
        self.base_url = (base_url or cfg.WORLD_BASE_URL).rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor

        # httpx.Client synchronous
        self._client = httpx.Client(timeout=self.timeout)

    # -------------------------------------------------------------

    def close(self) -> None:
        """Close the underlying httpx client."""
        self._client.close()

    # -------------------------------------------------------------

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
        response = self._request("POST", "/step_multipart", json=payload, expect_json=False)

        content_type = response.headers.get("content-type", "")
        return StepResponseMultipart.decode(response.content, content_type)

    # -------------------------------------------------------------

    def _send_once(self, method: str, path: str, expect_json: bool = True, **kwargs) -> Dict[str, Any] | Any:
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

    def _request(self, method: str, path: str, expect_json: bool = True, **kwargs) -> Dict[str, Any] | Any:
        """Retry wrapper with exponential backoff."""
        for attempt in range(self.max_retries + 1):
            try:
                return self._send_once(method, path, expect_json, **kwargs)
            except (httpx.RequestError, httpx.HTTPStatusError) as e:
                if attempt >= self.max_retries:
                    logger.error(f"Request failed after {self.max_retries} retries: {e}")
                    raise
                delay = self.backoff_factor * (2**attempt)
                logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.2f}s...")
                time.sleep(delay)
        raise RuntimeError("Unreachable")
