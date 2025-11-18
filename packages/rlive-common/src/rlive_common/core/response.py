from typing import Any, Dict, Optional
import json

from pydantic import BaseModel, Field, ConfigDict
import numpy as np
import cv2 as cv

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
        """
        Encode JSON metadata + NumPy image into multipart/mixed bytes.
        """
        meta_json = self.model_dump_json(exclude={"image", "boundary"})

        img_bytes = b""
        if self.image is not None:
            # Encode np.ndarray as PNG bytes using OpenCV
            success, buf = cv.imencode(".png", self.image)
            if not success:
                raise ValueError(f"cv2.imencode failed for image shape={self.image.shape}, dtype={self.image.dtype}")
            img_bytes = buf.tobytes()

        boundary = self.boundary

        # --- Build multipart manually ---
        body = f"--{boundary}\r\nContent-Type: application/json\r\n\r\n{meta_json}\r\n".encode()

        if img_bytes:
            body += f"--{boundary}\r\nContent-Type: image/png\r\n\r\n".encode() + img_bytes + b"\r\n"

        body += f"--{boundary}--\r\n".encode()
        content_type = f"multipart/mixed; boundary={boundary}"
        return body, content_type

    @classmethod
    def decode(cls, body: bytes, content_type: str) -> "StepResponseMultipart":
        """
        Decode a multipart/mixed body back into StepMultipartModel.

        Expected structure:
            --<boundary>
            Content-Type: application/json

            {...JSON metadata...}
            --<boundary>
            Content-Type: image/png

            <binary image data>
            --<boundary>--
        """
        if "boundary=" not in content_type:
            raise ValueError("Missing boundary in Content-Type header")

        boundary = content_type.split("boundary=")[-1].strip()
        parts = body.split(f"--{boundary}".encode())

        meta = None
        image = None

        for part in parts:
            part = part.strip()
            if not part or part.startswith(b"--"):
                continue

            headers, _, content = part.partition(b"\r\n\r\n")

            if b"application/json" in headers:
                meta = json.loads(content.decode().strip())
            elif b"image/png" in headers:
                np_arr = np.frombuffer(content.strip(), np.uint8)
                image = cv.imdecode(np_arr, cv.IMREAD_COLOR)
                if image is None:
                    raise ValueError("Failed to decode PNG image from multipart body")

        if not meta:
            raise ValueError("Missing JSON metadata in multipart body")

        return cls(**meta, image=image, boundary=boundary)