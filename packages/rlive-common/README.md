# rlive-common

Core utilities and shared functionality for the ReLoop_RLive project.

## 📦 What's Included

This package provides shared components used across the ReLoop_RLive ecosystem:

### Core Models
- **Request/Response Models**: Pydantic models for API communication
  - `BaseRequest`, `StepRequest`, `ResetRequest`
  - `BaseResponse`, `StepResponseJSON`, `StepResponseMultipart`
  - `AttachHardwareRequest/Response`, `DetachHardwareRequest/Response`

### Type Definitions
- **ImageArray**: Custom type for efficient image serialization (base64-encoded with shape/dtype metadata)
- **NumpyArray**: Type for serializing NumPy arrays to JSON-compatible lists

### Utilities
- **Logging**: Centralized logging configuration with customizable formats
- **Config**: Base configuration management utilities

## 🚀 Installation

### As Part of the Full Project

```bash
# From project root
uv sync --all-packages
```

### Standalone Installation

The installation command depends on your current directory:

**From Repository Root** (`ReLoop_RLive/`):
```bash
# Install only rlive-common
uv sync --package rlive-common
```

**From Package Root** (`ReLoop_RLive/packages/rlive-common/`):
```bash
# Install rlive-common and dependencies
uv sync
```

## 📚 Usage

### Using Request/Response Models

```python
from rlive_common.core.request import StepRequest, ResetRequest
from rlive_common.core.response import BaseResponse

# Create a step request
request = StepRequest(action=5)

# Create a response with observation
import numpy as np
response = BaseResponse(
    observation=np.zeros((480, 640, 3), dtype=np.uint8),
    truncated=False,
    info={"status": "ok"}
)
```

### Using ImageArray Type

The `ImageArray` type automatically handles serialization/deserialization of NumPy arrays:

```python
from rlive_common.core.types import ImageArray
from pydantic import BaseModel
import numpy as np

class MyModel(BaseModel):
    image: ImageArray

# Serialization (NumPy → dict with base64 data)
model = MyModel(image=np.zeros((100, 100, 3), dtype=np.uint8))
json_data = model.model_dump()
# {'image': {'shape': [100, 100, 3], 'dtype': 'uint8', 'data': '...'}}

# Deserialization (dict → NumPy)
reconstructed = MyModel.model_validate(json_data)
assert isinstance(reconstructed.image, np.ndarray)
```

### Logging

```python
from rlive_common.utils import get_logger

logger = get_logger(__name__)

logger.info("Starting process...")
logger.debug("Debug information")
logger.warning("Warning message")
logger.error("Error occurred")
```

Configure logging via environment variables:
- `LOG_LEVEL`: DEBUG, INFO, WARNING, ERROR
- `LOG_FORMAT`: simple, detailed, json

## 🧪 Testing

Run tests for this package:

```bash
pytest tests/rlive-common/ -v
```

## 📄 License

See LICENSE file in project root for details.


