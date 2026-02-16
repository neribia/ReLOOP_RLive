"""Demo: Ball in Box Simulation

This demo shows two approaches for creating simulation engines:

1. **Modern**: Registry-Based Factory Pattern
   - Engines auto-register via @register_*_backend decorators
   - Clean factory dispatch: PhysicsEngine(config) → specific engine
   - Type-safe, extensible architecture

2. **Legacy**: Direct Class Instantiation
   - Directly instantiate engine classes with parameters
   - Simple but less structured
   - Good for quick tests

Both approaches are valid. Choose based on your use case:
- **Production**: Use factories with config objects
- **Quick prototyping**: Direct instantiation

Usage:
    python -m rlive_sim.demo_ball_in_box              # Modern (factory)
    python -m rlive_sim.demo_ball_in_box --legacy     # Legacy (direct)

Controls:
    - Press any key to make a random move
    - Press 'q' or ESC to quit
"""

import argparse
import cv2
import numpy as np

from rlive_sim.engine import SimulationEngine, PhysicsEngine, RenderEngine, SimplePhysicsEngine, OpenCVRenderEngine
from rlive_sim.config import SimulationConfig, PhysicsConfig, RenderConfig, PhysicsBackend, RenderBackend


def main():
    """Run demo using registry-based factory pattern (modern approach)."""
    print("=" * 60)
    print("Modern: Registry-Based Factory Pattern")
    print("=" * 60)
    print()
    print("PhysicsEngine(config) → dispatches to SimplePhysicsEngine")
    print("RenderEngine(config)  → dispatches to OpenCVRenderEngine")
    print()

    # Create configs
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

    # Create engines using factories
    # No if/elif, no .create(), just clean factory dispatch
    physics = PhysicsEngine(sim_config.physics)
    render = RenderEngine(sim_config.render)

    print(f"✓ Physics: {type(physics).__name__}(PhysicsConfig)")
    print(f"✓ Render:  {type(render).__name__}(RenderConfig)")
    print()

    # Create simulation engine
    sim = SimulationEngine(physics_engine=physics, render_engine=render)

    run_demo(sim)


def main_legacy():
    """Run demo using legacy direct class instantiation."""
    print("=" * 60)
    print("Legacy: Direct Class Instantiation")
    print("=" * 60)
    print()
    print("SimplePhysicsEngine(...) → direct instantiation")
    print("OpenCVRenderEngine(...)  → direct instantiation")
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

    run_demo(sim)


def run_demo(sim: SimulationEngine):
    """Main demo loop."""
    # Reset and get initial state
    state = sim.reset()
    print(f"Initial position: ({state.position[0]:.1f}, {state.position[1]:.1f})")
    print("Press any key to make a random move, 'q' or ESC to quit.\n")

    # Main loop
    running = True
    step_count = 0

    while running:
        # Render current state
        image = sim.render()

        # Convert RGB to BGR for OpenCV display
        display_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Add info text
        info_text = f"Step: {step_count} | Pos: ({state.position[0]:.1f}, {state.position[1]:.1f})"
        cv2.putText(
            display_image, info_text, (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1
        )

        if state.extra.get("hit_wall", False):
            cv2.putText(
                display_image, "HIT WALL!", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
            )

        # Show image
        cv2.imshow("Ball in Box Demo", display_image)

        # Wait for key press
        key = cv2.waitKey(0) & 0xFF

        if key == ord('q') or key == 27:  # 'q' or ESC
            running = False
        else:
            # Random action: angle (0-360) and distance (10-50)
            angle = np.random.randint(0, 360)
            distance = np.random.randint(10, 51)

            # Apply action and step
            sim.apply_action([angle, distance])
            state = sim.step()
            step_count += 1

            print(f"Step {step_count}: angle={angle}°, distance={distance}px -> "
                  f"pos=({state.position[0]:.1f}, {state.position[1]:.1f})"
                  f"{' [HIT WALL]' if state.extra.get('hit_wall') else ''}")

    cv2.destroyAllWindows()
    sim.close()
    print("\nDemo finished!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ball in Box Demo - Shows Modern and Legacy Engine Creation"
    )
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="Use legacy direct class instantiation instead of factory pattern"
    )
    args = parser.parse_args()

    if args.legacy:
        main_legacy()
    else:
        main()
