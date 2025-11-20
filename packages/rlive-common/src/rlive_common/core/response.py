from typing import Any, Dict, Optional
import json

from pydantic import BaseModel, Field, ConfigDict
import numpy as np
import cv2 as cv
from email.message import Message

from rlive_common.core.pydantic_types import ImageArray, NumpyArray


class Response(BaseModel):
    """Basic response body for POST."""

    observation: NumpyArray
    truncated: bool = False
    info: Dict[str, Any]

    model_config = ConfigDict(arbitrary_types_allowed=True)


class ResetResponse(Response):
    """Response body for POST /reset."""


class StepResponseJSON(Response):
    """Response body for POST /step_json."""

    image: ImageArray | None = None


class StepResponseMultipart(Response):
    """Response body for POST /step_multipart."""

    image: Optional[np.ndarray] = Field(default=None, exclude=True)
    boundary: str = Field(default="world-step", exclude=True)

    def encode(self) -> tuple[bytes, str]:
        meta_dict = self.model_dump(exclude={"image", "boundary"})
        if self.image is not None:
            meta_dict["image_ndim"] = int(self.image.ndim)

        meta_json = json.dumps(meta_dict)

        boundary = self.boundary

        parts = [
            (
                f"--{boundary}\r\n"
                f"Content-Type: application/json\r\n\r\n"
                f"{meta_json}\r\n"
            ).encode()
        ]

        if self.image is not None:
            ok, buf = cv.imencode(".png", self.image)
            if not ok:
                raise ValueError(f"cv2.imencode failed for image shape={self.image.shape}, dtype={self.image.dtype}")
            parts.append(
                f"--{boundary}\r\nContent-Type: image/png\r\n\r\n".encode()
                + buf.tobytes() + b"\r\n"
            )

        parts.append(f"--{boundary}--\r\n".encode())

        body = b"".join(parts)
        content_type = "multipart/form-data; boundary={}".format(boundary)

        return body, content_type

    @classmethod
    def decode(cls, body: bytes, content_type: str) -> "StepResponseMultipart":
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

        if meta is None:
                raise ValueError("Missing JSON metadata")

        meta.pop("image_ndim", None)

        return cls(**meta, image=image, boundary=boundary)
