from typing import Any
import json

from pydantic import BaseModel, Field, ConfigDict
import numpy as np
import cv2 as cv
from email.message import Message

from rlive_common.core.custom_types import ImageArray


class AttachHardwareResponse(BaseModel):
    """Response body for POST /connect."""

    success: bool
    info: dict[str, Any]


class DetachHardwareResponse(BaseModel):
    """Response body for POST /disconnect."""

    success: bool
    info: dict[str, Any]


class BaseResponse(BaseModel):
    """Basic response body for POST."""

    observation: ImageArray  # Options to encode the image/obs: NumpyArray, ImageArray or over Multipart
    truncated: bool = False
    info: dict[str, Any]

    model_config = ConfigDict(arbitrary_types_allowed=True)


class ResetResponse(BaseResponse):
    """Response body for POST /reset."""


class StepResponseJSON(BaseResponse):
    """Response body for POST /step_json."""


class StepResponseMultipart(BaseResponse):
    """Response body for POST /step_multipart."""

    observation: np.ndarray | None = Field(default=None, exclude=True)
    boundary: str = Field(default="world-step", exclude=True)

    def encode(self) -> tuple[bytes, str]:
        """
        --boundary
        Content-Type: application/json

        {meta_data}
        --boundary
        Content-Type: image/png

        <PNG_BYTES>
        --boundary--

        Returns:
            A tuple containing (body_bytes, content_type).
        """

        meta_dict = self.model_dump(exclude={"observation", "boundary"})
        if self.observation is not None:
            meta_dict["image_ndim"] = int(self.observation.ndim)

        meta_json = json.dumps(meta_dict)

        boundary = self.boundary

        parts = [
            (
                f"--{boundary}\r\n"
                f"Content-Type: application/json\r\n\r\n"
                f"{meta_json}\r\n"
            ).encode()
        ]

        if self.observation is not None:
            ok, buf = cv.imencode(".png", self.observation)
            if not ok:
                shape = self.observation.shape
                dtype = self.observation.dtype
                raise ValueError(f"cv2.imencode failed for image shape={shape}, dtype={dtype}")
            parts.append(
                f"--{boundary}\r\nContent-Type: image/png\r\n\r\n".encode()
                + buf.tobytes() + b"\r\n"
            )

        parts.append(f"--{boundary}--\r\n".encode())

        body = b"".join(parts)
        content_type = f"multipart/form-data; boundary={boundary}"

        return body, content_type

    @classmethod
    def decode(cls, body: bytes, content_type: str) -> "StepResponseMultipart":
        """Decode a multipart response from bytes.
        Args:
            body: The response body bytes.
            content_type: The Content-Type header value.
        Returns:
            A StepResponseMultipart instance.
        Raises:
            ValueError: If the response is missing required parts.
        """
        msg = Message()
        msg["Content-Type"] = content_type
        boundary = msg.get_param("boundary")
        if not boundary:
            raise ValueError("Missing boundary in Content-Type header")

        meta = None
        image = None

        for part in body.split(f"--{boundary}".encode()):
            part = part.strip()
            if not part or part.startswith(b"--"):
                continue

            headers, _, content = part.partition(b"\r\n\r\n")
            headers = headers.lower()

            if b"application/json" in headers:
                meta = json.loads(content.decode().strip())

            elif b"image/png" in headers:
                ndim = meta.get("image_ndim", 3) if meta else 3
                flag = cv.IMREAD_GRAYSCALE if ndim == 2 else cv.IMREAD_COLOR

                arr = np.frombuffer(content.strip(), np.uint8)
                image = cv.imdecode(arr, flag)
                if image is None:
                    raise ValueError("Failed to decode image from PNG data")
        if meta is None:
            raise ValueError("Missing JSON metadata")

        meta.pop("image_ndim", None)

        return cls(**meta, observation=image, boundary=boundary)
