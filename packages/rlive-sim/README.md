# rlive-sim

Client-side Gymnasium-compatible environment for robot simulation..

## 📦 What's Included

This package provides the RL environment for simulation for a robot.

### Main Components

- **`SimulationEnv`**: A Gymnasium-compatible environment that communicates with the world server
  - Follows the standard Gymnasium API (`reset()`, `step()`, `close()`, `render()`)


### Features

- 🎮 **Gymnasium Compatible**: Drop-in replacement for standard RL environments

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
uv sync --package rlive-sim
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
uv run rlive-sim

```

### Basic Example

```python
from rlive_env.remote_env import RemoteWorldEnv

env = RemoteWorldEnv(max_episode_steps=10)

# Reset and get initial observation
obs, info = env.reset()
print(f"Observation shape: {obs.shape}")

# Standard Gymnasium loop
done = False
while not done:
    action = env.action_space.sample()  # Random action
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"action={action}, reward={reward:.3f}")
    done = truncated
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

| Variable | Default | Description          |
|----------|---------|----------------------|
| `XXX`    | `XXX`   | Example description. |


## 🧪 Testing

Run tests for this package:

```bash
pytest tests/rlive-env/ -v
```

## 📄 License

See LICENSE file in project root for details.


