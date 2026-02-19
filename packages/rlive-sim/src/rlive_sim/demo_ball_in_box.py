"""Demo: Ball in Box Simulation

This demo shows two approaches for creating simulation environments:

1. **Modern (Recommended)**: Config-based with SimulationEnv
   - Pass SimulationConfig to SimulationEnv
   - Factory pattern handled internally
   - Cleanest API: SimulationEnv(config=config)

2. **Legacy**: Direct Engine Instantiation
   - Create engines directly
   - Manually assemble SimulationEngine
   - More control, more code

Choose based on your use case:
- **Production/Research**: Use config-based SimulationEnv (recommended)
- **Quick prototyping**: Direct instantiation

Usage:
    python -m rlive_sim.demo_ball_in_box              # Modern (config)
    python -m rlive_sim.demo_ball_in_box --legacy     # Legacy (direct)

Controls:
    - Press any key to make a random move
    - Press 'q' or ESC to quit
"""

import argparse
import cv2
import numpy as np

from rlive_sim.engine import (
    SimulationEngine,
    SimplePhysicsEngine,
    OpenCVRenderEngine,
)
from rlive_sim.config import (
    SimulationConfig,
    PhysicsConfig,
    RenderConfig,
    PhysicsBackend,
    RenderBackend,
)
from rlive_sim.simulation_env import SimulationEnv


def main():
    """Run demo using config-based SimulationEnv (modern, recommended approach)."""
    print("=" * 60)
    print("Modern: Config-Based SimulationEnv (Recommended)")
    print("=" * 60)
    print()
    print("SimulationEnv(config=config) → factory handled internally")
    print()

    # Create config
    sim_config = SimulationConfig(
        use_integrated=False,
        physics=PhysicsConfig(
            backend=PhysicsBackend.SIMPLE,
            dt=0.01,
            extra={
                "box_width": 640,
                "box_height": 480,
                "ball_radius": 20,
            }
        ),
        render=RenderConfig(
            backend=RenderBackend.OPENCV,
            width=640,
            height=480,
            extra={
                "ball_radius": 20,
                "ball_color": (255, 0, 0),
                "box_color": (128, 128, 128),
                "bg_color": (0, 0, 0),
                "box_thickness": 2,
            }
        ),
    )

    # Create environment - factory is called internally
    env = SimulationEnv(config=sim_config)

    print(f"✓ SimulationEnv created from config")
    print(f"  - Engine: {type(env.engine).__name__}")
    print()

    run_demo(env)


def main_legacy():
    """Run demo using legacy direct engine instantiation."""
    print("=" * 60)
    print("Legacy: Direct Engine Instantiation")
    print("=" * 60)
    print()
    print("SimplePhysicsEngine(...) → direct instantiation")
    print("OpenCVRenderEngine(...)  → direct instantiation")
    print("SimulationEngine(...)    → manual assembly")
    print()

    # Create engines directly
    physics = SimplePhysicsEngine(
        box_width=640,
        box_height=480,
        ball_radius=20,
        dt=0.01,
    )

    render = OpenCVRenderEngine(
        width=640,
        height=480,
        ball_radius=20,
        ball_color=(255, 0, 0),
        box_color=(128, 128, 128),
        bg_color=(0, 0, 0),
        box_thickness=2,
    )

    print(f"✓ Physics: {type(physics).__name__}(...)")
    print(f"✓ Render:  {type(render).__name__}(...)")
    print()

    # Create simulation engine
    sim = SimulationEngine(physics_engine=physics, render_engine=render)

    print(f"✓ SimulationEngine: {type(sim).__name__}(...)")
    print()

    # Wrap in environment for consistent API
    env = SimulationEnv(engine=sim)

    run_demo(env)


def run_demo(env: SimulationEnv):
    """Main demo loop."""
    # Reset and get initial state
    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    print("Press any key to make a random move, 'q' or ESC to quit.\n")

    # Main loop
    running = True
    step_count = 0

    while running:
        # Current observation
        image = obs

        # Convert RGB to BGR for OpenCV display
        display_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Add info text
        info_text = f"Step: {step_count}"
        cv2.putText(
            display_image, info_text, (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1
        )

        # Show image
        cv2.imshow("Ball in Box Demo", display_image)

        # Wait for key press
        key = cv2.waitKey(0) & 0xFF

        if key == ord('q') or key == 27:  # 'q' or ESC
            running = False
        else:
            # Random action: heading/angle only (0-360 degrees)
            # Engine will use default distance
            action = np.random.randint(0, 360)

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            step_count += 1

            print(f"Step {step_count}: action=[{action}°]")

            if terminated or truncated:
                print("Episode ended!")
                running = False

    cv2.destroyAllWindows()
    env.close()
    print("\nDemo finished!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ball in Box Demo - Shows Modern and Legacy Approaches"
    )
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="Use legacy direct engine instantiation instead of config-based approach"
    )
    args = parser.parse_args()

    if args.legacy:
        main_legacy()
    else:
        main()
