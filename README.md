# ReLoop_RLive

Demonstrator for the ReLoop project — a reinforcement learning environment for controlling physical robots via a client-server architecture.

## 📦 Project Structure

This project is organized as a Python monorepo with three main packages:

```
packages/
├── rlive-common/    # Shared utilities, types, request/response models
├── rlive-env/       # Gymnasium-compatible remote environment client
└── rlive-world/     # World server, robot control, and camera interfaces
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
  - Sphero Bolt+ robot control via BLE
  - Camera interfaces (PiCamera, Webcam, Dummy)

## 🚀 Installation

This project uses [uv](https://docs.astral.sh/uv/) — a fast Python package and environment manager.

To install all packages and dependency groups (including development and optional groups), run:

```bash
uv sync --all-packages --all-groups
```

This will:

- Create or update your virtual environment
- Install all workspace packages
- Install every dependency group defined in your pyproject.toml (e.g., dev, docs, test, etc.)

Once complete, you can activate the environment:

```bash
source .venv/bin/activate
```

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

## 📄 License

See LICENSE file for details.
