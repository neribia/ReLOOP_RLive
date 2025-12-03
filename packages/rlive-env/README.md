# rlive-env

Client-side Gymnasium-compatible environment for remote robot control.

## 📦 What's Included

This package provides the RL environment client for interacting with the remote world server:

### Main Components

- **`RemoteWorldEnv`**: A Gymnasium-compatible environment that communicates with the world server
  - Follows the standard Gymnasium API (`reset()`, `step()`, `close()`)
  - Supports both dummy and real hardware
  - Configurable action/observation spaces
  
- **`WorldInterface`**: HTTP client for REST API communication
  - Handles requests to the world server
  - Automatic retry logic
  - Connection pooling and timeout management

### Features

- 🎮 **Gymnasium Compatible**: Drop-in replacement for standard RL environments
- 🌐 **Remote Execution**: Train locally, execute on remote hardware
- 🔄 **Automatic Reconnection**: Built-in retry logic for network issues
- 🛠️ **Hardware Abstraction**: Seamlessly switch between dummy and real hardware

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
# Install only rlive-env (includes rlive-common as dependency)
uv sync --package rlive-env
```

**From Package Root** (`ReLoop_RLive/packages/rlive-env/`):
```bash
# Install rlive-env and dependencies
uv sync
```

## 📚 Usage

### Quick Start

Run the built-in demo that connects to the world server and executes a few steps:

```bash
# Connect to local server (default: http://127.0.0.1:8000)
uv run rlive-env

# Or specify a different server
WORLD_BASE_URL=http://192.168.1.100:8000 uv run rlive-env
```

### Basic Example

```python
from rlive_env.remote_env import RemoteWorldEnv

# Create environment (connects to http://localhost:8000 by default)
env = RemoteWorldEnv(base_url="http://localhost:8000")

# Reset and get initial observation
obs, info = env.reset()
print(f"Observation shape: {obs.shape}")

# Standard Gymnasium loop
for t in range(10):
    action = env.action_space.sample()  # Random action
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Step {t}: action={action}, reward={reward:.3f}")
    
    if terminated or truncated:
        obs, info = env.reset()
        print("Episode reset.")

# Cleanup
env.close()
```

This is equivalent to the code in `main_function.py`.

### Custom Configuration

```python
from rlive_env.remote_env import RemoteWorldEnv

# Connect to a different server
env = RemoteWorldEnv(base_url="http://192.168.1.100:8000")

# Or use environment variables:
# export WORLD_BASE_URL=http://192.168.1.100:8000
import os
base_url = os.getenv("WORLD_BASE_URL", "http://127.0.0.1:8000")
env = RemoteWorldEnv(base_url=base_url)
```


## ⚙️ Configuration

Configure the environment using environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `WORLD_BASE_URL` | `http://localhost:8000` | URL of the world server |
| `WORLD_INTERFACE_TIMEOUT` | `30` | Request timeout in seconds |
| `WORLD_INTERFACE_MAX_RETRIES` | `3` | Maximum number of retry attempts |

## 🧪 Testing

Run tests for this package:

```bash
pytest tests/rlive-env/ -v
```

## 📄 License

See LICENSE file in project root for details.


