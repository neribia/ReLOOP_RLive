# rlive-world

Server-side package for controlling physical robots and managing the world state.

## 📦 What's Included

This package provides the world server and hardware control components:

### Main Components

- **World Server**: FastAPI-based REST API server
  - Manages robot and camera hardware
  - Handles RL environment requests (reset, step, attach/detach hardware)
  - Supports both JSON and multipart responses
  
- **Robot Control**: Sphero Bolt+ robot interface (optional `bolt` dependency group)
  - Direct BLE communication via `sphero_unsw` and `bleak`
  - Movement control, sensor reading
  - Dummy implementations for hardware-free testing
  
- **Camera Support**: Multiple camera backends
  - **PiCamera**: Raspberry Pi Camera Module
  - **Webcam**: USB cameras via OpenCV
  - **Dummy Camera**: Generates random images for testing

### Features

- 🤖 **Real Hardware Control**: Direct BLE communication with Sphero Bolt+ robots
- 📷 **Flexible Camera Support**: PiCamera, USB webcams, or dummy cameras
- 🔌 **REST API**: HTTP-based interface for remote control
- 🎭 **Dummy Mode**: Full hardware simulation for development/testing
- 🔧 **Configurable**: Environment-based configuration

## 🚀 Installation

### As Part of the Full Project

```bash
# From project root
uv sync --all-packages --all-groups
```

### Standalone Installation

The installation command depends on your current directory:

**From Repository Root** (`ReLoop_RLive/`):
```bash
# With real hardware support (Sphero Bolt+)
uv sync --package rlive-world --group bolt

# Or without hardware support (dummy mode only)
uv sync --package rlive-world
```

**From Package Root** (`ReLoop_RLive/packages/rlive-world/`):
```bash
# With real hardware support
uv sync --group bolt

# Or without hardware support
uv sync
```

> **Note**: The `--group bolt` flag installs the Sphero robot control libraries (`sphero_unsw`, `bleak`). Omit this flag if you only need dummy hardware for testing.

### Hardware Requirements

**For Real Hardware:**
- Sphero Bolt+ robot
- Bluetooth adapter (built-in or USB)
- Camera (optional):
  - Raspberry Pi Camera Module v3, or
  - USB webcam

**For Development/Testing:**
- No hardware required (uses dummy implementations)

## 📚 Usage

### Running the World Server

**Quick Start with Dummy Hardware:**

```bash
# Local mode (binds to 127.0.0.1:8000)
uv run rlive-world --local

# Production mode (binds to 0.0.0.0:8000)
uv run rlive-world

# With auto-reload (for development only, NOT for real hardware!)
uv run rlive-world --local --reload
```

The server automatically uses dummy hardware if no real hardware is configured.

**With Real Hardware:**

1. Pair your Sphero Bolt+ via Bluetooth
2. Configure environment variables:

```bash
# .env file or export these
SPHEROBOLTPLUS_NAME=BP-XXXX  # Your Sphero's Bluetooth name
SPHEROBOLTPLUS_SPEED=100
CAMERA_TYPE=webcam  # or 'picam'
CAMERA_ID=0
```

3. Start the server:

```bash
# Production mode for Raspberry Pi
uv run rlive-world
```

### Alternative: Direct Python Execution

You can also run the server directly with Python:

```bash
# Activate environment first
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Run server module
python -m rlive_world.world_server
```

### API Endpoints

Once running, the server exposes these endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/attach_hardware` | POST | Connect to robot and camera |
| `/detach_hardware` | POST | Disconnect from hardware |
| `/reset` | POST | Reset environment state |
| `/step_json` | POST | Execute action, return JSON |
| `/step_multipart` | POST | Execute action, return multipart (includes raw image) |
| `/docs` | GET | Interactive API documentation (Swagger UI) |

### Testing the API

You can test the API directly with `curl`:

```bash
# Attach hardware
curl -X POST http://localhost:8000/attach_hardware \
  -H "Content-Type: application/json" \
  -d '{}'

# Reset environment
curl -X POST http://localhost:8000/reset \
  -H "Content-Type: application/json" \
  -d '{}'

# Execute a step
curl -X POST http://localhost:8000/step_json \
  -H "Content-Type: application/json" \
  -d '{"action": 90}'
```

Or visit `http://localhost:8000/docs` for interactive documentation.

### Programmatic Usage

You can also use the World class directly in Python:

```python
from rlive_world.world import World
from rlive_common.core.request import StepRequest, ResetRequest

# Create world
world = World()

# Attach hardware (uses dummy by default)
from rlive_common.core.request import AttachHardwareRequest
world.attach_hardware(AttachHardwareRequest())

# Reset
response = world.reset(ResetRequest())
print(f"Observation shape: {response.observation.shape}")

# Take a step
step_response = world.step(StepRequest(action=90))
print(f"New observation shape: {step_response.observation.shape}")

# Cleanup
from rlive_common.core.request import DetachHardwareRequest
world.detach_hardware(DetachHardwareRequest())
```

## ⚙️ Configuration

Configure the world server using environment variables:

### Server Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `WORLD_HOST` | `0.0.0.0` | Server host address |
| `WORLD_PORT` | `8000` | Server port |

### Robot Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `SPHEROBOLTPLUS_NAME` | `BP-D217` | Sphero Bolt+ Bluetooth name |
| `SPHEROBOLTPLUS_SPEED` | `100` | Default movement speed (0-255) |
| `SPHEROBOLTPLUS_DURATION` | `1.0` | Default movement duration (seconds) |

### Camera Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `CAMERA_TYPE` | `dummy` | Camera type: `dummy`, `webcam`, or `picam` |
| `CAMERA_ID` | `0` | Camera device ID (for webcam) |
| `CAMERA_WIDTH` | `640` | Image width |
| `CAMERA_HEIGHT` | `480` | Image height |

## 🧪 Testing

Run tests for this package:

```bash
pytest tests/rlive-world/ -v
```

### Testing with Dummy Hardware

All tests use dummy hardware implementations by default, allowing you to run the full test suite without physical devices.

## 🐛 Troubleshooting

### Bluetooth Connection Issues

If the robot won't connect:

1. Ensure Bluetooth is enabled on your system
2. Pair the Sphero via your OS Bluetooth settings first
3. Check the robot name matches `SPHEROBOLTPLUS_NAME`
4. Try restarting the Sphero (place in charging cradle)

### Camera Not Found

If the camera isn't detected:

- For webcam: Check `CAMERA_ID` (try 0, 1, 2)
- For PiCamera: Ensure camera is enabled in `raspi-config`
- Use dummy camera for testing: `CAMERA_TYPE=dummy`

## 📄 License

See LICENSE file in project root for details.


