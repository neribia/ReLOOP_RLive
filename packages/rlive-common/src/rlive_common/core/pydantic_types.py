import base64
import numpy as np
from typing import Annotated, Any, Dict
from pydantic import BeforeValidator, PlainSerializer


# ---------- Annotated ImageArray type ----------
def _encode_image(array: np.ndarray) -> Dict[str, Any]:
    """Convert a NumPy image array to a base64 JSON object."""
    return {
        "shape": array.shape,
        "dtype": str(array.dtype),
        "data": base64.b64encode(array.tobytes()).decode("utf-8"),
    }


def _decode_image(value: Any) -> np.ndarray:
    """Convert JSON/base64 dict back to NumPy array."""

    if isinstance(value, dict) and {"shape", "dtype", "data"} <= value.keys():
        data = base64.b64decode(value["data"])
        arr = np.frombuffer(data, dtype=np.dtype(value["dtype"]))
        return arr.reshape(value["shape"])

    raise TypeError(f"Invalid image payload: {type(value)}")


ImageArray = Annotated[
    np.ndarray,
    BeforeValidator(_decode_image),
    PlainSerializer(_encode_image, return_type=dict),
]


# ---------- Annotated NumpyArray type ----------
def _ndarray_before_validator(x: Any) -> np.ndarray:
    if isinstance(x, str):
        import ast

        x_list = ast.literal_eval(x)
        x = np.array(x_list)
    elif isinstance(x, list):
        x = np.array(x)
    return x


def _ndarray_serializer(x: np.ndarray) -> list:
    return x.tolist()


NumpyArray = Annotated[
    np.ndarray,
    BeforeValidator(_ndarray_before_validator),
    PlainSerializer(_ndarray_serializer, return_type=list),
]
