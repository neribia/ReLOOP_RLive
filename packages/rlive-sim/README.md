# rlive-sim

A flexible simulation package for reinforcement learning with pluggable physics and rendering backends.

## 📦 What's Included

This package provides a modular simulation architecture with support for multiple physics and rendering backends, all wrapped in a Gymnasium-compatible environment interface.

### Main Components

- **`SimulationEnv`**: A Gymnasium-compatible environment for RL training
  - Follows the standard Gymnasium API (`reset()`, `step()`, `render()`, `close()`)
  - Flexible engine architecture supporting different backend combinations

- **`SimulationEngine`**: Core orchestrator for physics and rendering
  - Unified API for integrated engines (Godot, MuJoCo) and separate engines (physics + render)
  - Handles state management and action application
  - Supports efficient combined step-and-render operations

- **Physics Engines**:
  - `SimplePhysicsEngine`: Basic physics simulation using numpy
  - `PyMunkPhysicsEngine`: Advanced 2D physics (stub implementation)
  - Integrated backends: MuJoCo, Isaac Sim, Godot (via unified interface)

- **Render Engines**:
  - `OpenCVRenderEngine`: Fast OpenCV-based rendering
  - `MitsubaRenderEngine`: Advanced ray tracing (stub implementation)
  - Integrated backends: Built-in rendering for Godot, MuJoCo, etc.

### Architecture

The package supports two complementary architectures:

1. **Separate Engines**: Physics and rendering are independent
   - Flexibility: Choose best-of-breed solutions
   - Modularity: Easy to swap components
   - Trade-off: Potential performance overhead

2. **Integrated Engines**: Physics and rendering tightly coupled
   - Efficiency: Shared resources and optimizations
   - Examples: Godot, MuJoCo, Isaac Sim

`SimulationEngine` provides a unified interface for both, so your RL code doesn't need to know which architecture is being used.

### Features

- 🎮 **Gymnasium Compatible**: Drop-in replacement for standard RL environments
- 🔌 **Pluggable Backends**: Swap physics/render engines without changing RL code
- 🏗️ **Unified API**: Works seamlessly with integrated or separate engines
- ⚡ **Configurable**: Registry-based factory pattern for clean extensibility

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
# Install only rlive-sim (includes rlive-common as dependency)
uv sync --package rlive-sim
```

**From Package Root** (`ReLoop_RLive/packages/rlive-sim/`):
```bash
# Install rlive-sim and dependencies
uv sync
```

## 📚 Usage

### Quick Start with Demo

Run the interactive demo that shows ball-in-box simulation:

```bash
# Modern approach (recommended)
python -m rlive_sim.demo_ball_in_box

# Legacy approach (direct instantiation)
python -m rlive_sim.demo_ball_in_box --legacy
```

Controls: Press any key to apply a random action, press 'q' or ESC to quit.

### Basic Example: Default Configuration

```python
from rlive_sim import SimulationEnv

# Create environment with default configuration
env = SimulationEnv(max_episode_steps=100)

# Reset and get initial observation
obs, info = env.reset()
print(f"Observation shape: {obs.shape}")

# Standard Gymnasium loop
done = False
while not done:
    action = env.action_space.sample()  # Random action
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"action={action}, reward={reward:.3f}")
    done = terminated or truncated
    if done:
        obs, info = env.reset()
        print("Episode reset.")

# Cleanup
env.close()
```

### Separate Engines (Modern: Factory Pattern)

Use the registry-based factory pattern for clean separation of concerns:

```python
from rlive_sim import (
    SimulationEnv,
    SimulationEngine,
    SimplePhysicsEngine,
    OpenCVRenderEngine,
)
from rlive_sim.config import SimulationConfig, PhysicsConfig, RenderConfig, PhysicsBackend, RenderBackend

# Create configuration objects
sim_config = SimulationConfig(
    use_integrated=False,  # Use separate engines
    physics=PhysicsConfig(
        backend=PhysicsBackend.SIMPLE,
        box_width=640,
        box_height=480,
        ball_radius=20,
        dt=0.01,
    ),
    render=RenderConfig(
        backend=RenderBackend.OPENCV,
        width=640,
        height=480,
    ),
)

# Create engines from configuration
physics = SimplePhysicsEngine(
    box_width=sim_config.physics.box_width,
    box_height=sim_config.physics.box_height,
    ball_radius=sim_config.physics.ball_radius,
    dt=sim_config.physics.dt,
)

render = OpenCVRenderEngine(
    width=sim_config.render.width,
    height=sim_config.render.height,
)

# Combine into simulation engine
sim = SimulationEngine(physics_engine=physics, render_engine=render)

# Create environment
env = SimulationEnv(engine=sim, max_episode_steps=100)

# Use as normal Gymnasium environment
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step([45.0, 50.0])
image = env.render()
```

### Separate Engines (Legacy: Direct Instantiation)

For quick prototyping, directly instantiate engines:

```python
from rlive_sim import SimulationEnv, SimulationEngine, SimplePhysicsEngine, OpenCVRenderEngine

# Directly create engines with parameters
physics = SimplePhysicsEngine(box_width=640, box_height=480, ball_radius=20)
render = OpenCVRenderEngine(width=640, height=480)

# Combine them
sim = SimulationEngine(physics_engine=physics, render_engine=render)

# Create environment
env = SimulationEnv(engine=sim)

obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step([45.0, 50.0])
```

### Integrated Engine Example

For efficient simulation with integrated physics and rendering (e.g., MuJoCo, Godot):

```python
from rlive_sim import SimulationEnv, SimulationEngine
from rlive_sim.config import IntegratedConfig, IntegratedBackend
from rlive_sim.engine import GodotIntegratedEngine  # or MuJoCoIntegratedEngine

# Note: This is a conceptual example. Implementation depends on available backends.

# Create integrated engine configuration
config = IntegratedConfig(backend=IntegratedBackend.GODOT)

# Create integrated engine
integrated = GodotIntegratedEngine(config)

# Wrap in SimulationEngine for consistent API
sim = SimulationEngine(integrated_engine=integrated)

# Create environment
env = SimulationEnv(engine=sim)

obs, info = env.reset()

# Efficient combined step-and-render
obs, reward, terminated, truncated, info = env.step([1.0, 0.5])
image = env.render()
```

### Custom Configuration

Override default configuration by providing a custom `SimulationConfig`:

```python
from rlive_sim import SimulationEnv
from rlive_sim.config import SimulationConfig, PhysicsConfig, RenderConfig

config = SimulationConfig(
    max_episode_steps=200,
    physics=PhysicsConfig(
        dt=0.005,  # Smaller timestep for more accuracy
        box_width=800,
        box_height=600,
    ),
    render=RenderConfig(
        width=800,
        height=600,
    ),
)

env = SimulationEnv(config=config)
obs, info = env.reset()
```

## ⚙️ Configuration

Configure the environment through `SimulationConfig`:

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `max_episode_steps` | int | 500 | Maximum steps per episode |
| `use_integrated` | bool | False | Use integrated engine if available |
| `physics` | PhysicsConfig | default | Physics engine configuration |
| `render` | RenderConfig | default | Render engine configuration |

### Physics Configuration

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `backend` | PhysicsBackend | SIMPLE | Physics engine backend |
| `dt` | float | 0.01 | Simulation timestep (seconds) |
| `box_width` | int | 640 | Simulation box width (pixels) |
| `box_height` | int | 480 | Simulation box height (pixels) |
| `ball_radius` | int | 20 | Ball radius (pixels) |

### Render Configuration

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `backend` | RenderBackend | OPENCV | Render engine backend |
| `width` | int | 640 | Output image width (pixels) |
| `height` | int | 480 | Output image height (pixels) |

## 🧪 Testing

Run tests for this package:

```bash
pytest tests/rlive-sim/ -v
```

## 📄 License

See LICENSE file in project root for details.


