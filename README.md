# ReLoop_RLive

Demonstrator for the ReLoop project — a reinforcement learning environment for controlling physical robots via a client-server architecture.

## 📖 About

**ReLoop_RLive** is a real-world reinforcement learning platform that bridges the gap between simulated and physical robot control. It provides a Gymnasium-compatible interface for training RL agents on physical hardware, specifically designed for Sphero Bolt+ robots.

### Key Features

- **Client-Server Architecture**: Separates the RL environment from the physical hardware
- **Gymnasium Compatible**: Works seamlessly with popular RL frameworks (Stable-Baselines3, Ray RLlib, etc.)
- **Real Robot Control**: Direct BLE communication with Sphero Bolt+ robots
- **Camera Integration**: Support for PiCamera, USB webcams, and dummy cameras for testing
- **Hardware Abstraction**: Dummy implementations for hardware-free development and testing
- **REST API**: HTTP-based communication for flexible deployment scenarios

## 📦 Project Structure

This project is organized as a Python monorepo with five main packages:

```
packages/
├── rlive-common/    # Shared utilities, types, request/response models
├── rlive-env/       # Gymnasium-compatible remote environment client
├── rlive-world/     # World server, robot control, and camera interfaces
├── rlive-sim/       # Simulation environment with pluggable physics/render backends
└── rlive-example/   # Example scripts and demonstration utilities
```

### Package Descriptions

- **rlive-common**: Core functionality shared across packages including:
  - Request/Response models for API communication
  - Image encoding utilities (base64 and multipart)
  - Logging configuration
  
- **rlive-env**: Client-side package providing:
  - `RemoteWorldEnv`: A Gymnasium-compatible environment for remote world interaction
  - `WorldInterface`: HTTP client for communicating with the world server
  
- **rlive-world**: Server-side package including:
  - FastAPI-based world server with REST endpoints
  - Sphero Bolt+ robot control via BLE (optional `bolt` dependency group)
  - Camera interfaces (PiCamera, Webcam, Dummy)

- **rlive-sim**: Simulation package providing:
  - `SimulationEnv`: A Gymnasium-compatible simulation environment
  - Pluggable physics backends (Simple, PyMunk)
  - Pluggable render backends (OpenCV, Mitsuba)
  - Support for integrated engines (MuJoCo, Godot)

- **rlive-example**: Example and demonstration scripts including:
  - Tracking demo for ball detection
  - Environment usage examples
  - Dataset recording with d3rlpy

## 🚀 Installation

This project uses [uv](https://docs.astral.sh/uv/) — a fast Python package and environment manager.

### Install All Packages

To install all packages and dependency groups (including development and optional groups), run:

```bash
uv sync --all-packages --all-groups --all-extras
```

This will:

- Create or update your virtual environment
- Install all workspace packages (rlive-common, rlive-env, rlive-world, rlive-sim, rlive-example)
- Install every dependency group defined in your pyproject.toml (e.g., dev, bolt, etc.)

Once complete, you can activate the environment:

```bash
source .venv/bin/activate  # On Linux/macOS
.venv\Scripts\activate     # On Windows
```

### Install Individual Packages

If you only need a specific package, you can install it individually:

**For RL Environment Client only:**
```bash
uv sync --package rlive-env
```

**For World Server only:**
```bash
# With real hardware support (Sphero Bolt+)
uv sync --package rlive-world --group bolt

# Or without hardware (dummy mode only)
uv sync --package rlive-world
```

**For Common utilities only:**
```bash
uv sync --package rlive-common
```

**For Simulation Environment only:**
```bash
uv sync --package rlive-sim
```

**For Example Scripts only:**
```bash
uv sync --package rlive-example
```

> **Note**: The `--group bolt` flag installs Sphero robot control libraries. Required for real hardware, optional for dummy mode.

## 🏁 Getting Started

### Quick Start with Dummy Hardware

Test the system without physical hardware using dummy implementations:

**1. Start the World Server:**

```bash
# Simple one-command start (local mode)
uv run rlive-world --local

# Or for production (binds to 0.0.0.0)
uv run rlive-world
```

The server will start on `http://localhost:8000` (local mode) or `http://0.0.0.0:8000` (production mode) and automatically use dummy hardware.

**2. Run the RL Environment Client (in a new terminal):**

```bash
# Simple demo with 3 steps
uv run rlive-env

# Or run with custom server URL
WORLD_BASE_URL=http://localhost:8000 uv run rlive-env
```

**3. Or write your own Python script:**

```python
from rlive_env.remote_env import RemoteWorldEnv

# Create the environment
env = RemoteWorldEnv(base_url="http://localhost:8000")

# Reset the environment
obs, info = env.reset()
print(f"Initial observation shape: {obs.shape}")

# Run a few steps
for step in range(5):
    action = env.action_space.sample()  # Random action
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Step {step}: action={action}, reward={reward}")
    
    if terminated or truncated:
        obs, info = env.reset()

# Cleanup
env.close()
```

### Using Real Hardware

**Prerequisites:**
- Sphero Bolt+ robot (paired via Bluetooth)
- Camera (PiCamera or USB webcam)

**1. Configure the environment:**

Create a `.env` file or set environment variables:

```bash
# World Server Configuration
WORLD_HOST=0.0.0.0
WORLD_PORT=8000
SPHEROBOLTPLUS_NAME=BP-XXXX  # Your Sphero's name
SPHEROBOLTPLUS_SPEED=100

# Camera Configuration  
CAMERA_TYPE=webcam  # or 'picam' for Raspberry Pi Camera
CAMERA_ID=0
```

**2. Start the World Server:**

```bash
# Production mode (Raspberry Pi or dedicated hardware server)
uv run rlive-world

# Or local testing
uv run rlive-world --local
```

**3. Run your RL agent (on your training machine):**

```bash
# Connect to remote server
WORLD_BASE_URL=http://192.168.1.100:8000 uv run rlive-env
```

The client code remains the same as the dummy hardware example above!



## 🧪 Running Tests

Run all tests with pytest:

```bash
pytest tests/ -v
```

Or run tests for a specific package:

```bash
pytest tests/rlive-common/ -v
pytest tests/rlive-env/ -v
pytest tests/rlive-world/ -v
pytest tests/rlive-sim/ -v
```

## 🛠️ Development

### Linting

This project uses [ruff](https://docs.astral.sh/ruff/) for linting. Run the linter with:

```bash
ruff check .
```

Auto-fix linting issues:

```bash
ruff check . --fix
```

### Code Style

The project follows Google-style docstrings and enforces the following ruff rules:
- `E`: pycodestyle errors
- `F`: pyflakes
- `UP`: pyupgrade
- `D`: pydocstyle
- `PL`: pylint

## 🌐 API Endpoints

The world server exposes the following REST API endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/attach_hardware` | POST | Connect to hardware (robot, camera) |
| `/detach_hardware` | POST | Disconnect from hardware |
| `/reset` | POST | Reset the environment |
| `/step_json` | POST | Execute an action, returns JSON response |
| `/step_multipart` | POST | Execute an action, returns multipart response with image |

## ⚙️ Configuration

Configuration is managed through environment variables. See the `.env_sample` files in each package for available options.

### Common Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DEBUG` | `false` | Enable debug mode |
| `LOG_LEVEL` | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR) |
| `LOG_FORMAT` | `detailed` | Log format style |

### World Server Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `WORLD_HOST` | `0.0.0.0` | Server host address |
| `WORLD_PORT` | `8000` | Server port |
| `SPHEROBOLTPLUS_NAME` | `BP-D217` | Sphero Bolt+ robot name |
| `SPHEROBOLTPLUS_SPEED` | `100` | Robot movement speed |

### Environment Client Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `WORLD_BASE_URL` | `http://localhost:8000` | World server URL |
| `WORLD_INTERFACE_TIMEOUT` | `30` | Request timeout (seconds) |
| `WORLD_INTERFACE_MAX_RETRIES` | `3` | Maximum retry attempts |

## 📚 Documentation

Build the documentation locally:

```bash
mkdocs serve
```

Then open `http://127.0.0.1:8000` in your browser.

## Built With
- **[Gymnasium](https://gymnasium.farama.org/)**: Standard API for reinforcement learning environments
- **[FastAPI](https://fastapi.tiangolo.com/)**: Modern, fast web framework for building the REST API

## 🙏 Acknowledgments

This project builds upon excellent open-source work:

- **[sphero_unsw](https://github.com/UNSW-CORG/sphero_unsw)**: Python library for controlling Sphero robots, developed at UNSW. This library provides the core BLE communication and robot control functionality.


Special thanks to the UNSW CORG team for their work on the Sphero control library.

## 📄 License

See LICENSE file for details.
