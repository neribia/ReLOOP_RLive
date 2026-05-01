"""SAPIEN Simulation Example

This example demonstrates how to use the SimulationEnv with SAPIEN integrated backend.
It shows how to use 3 different robot loading options:
1. Rigid Body - Programmatically built sphere (default, fastest)
2. URDF - Load from URDF file (resources/sapian/sphere_robot.urdf)
3. GLB - Load from GLB/GLTF file (resources/sapian/bolt_shell.glb)

Run with different ROBOT_TYPE values to test each option.
"""
import cv2 as cv
import numpy as np

from rlive_common import ActionSpaceType
from rlive_sim import (
    SimulationConfig,
    IntegratedConfig,
    IntegratedBackend,
    SimulationEnv,
)
from rlive_common.utils import get_logger

logger = get_logger(__name__)

# Number of episodes to run
NUM_EPISODES = 10


def extract_goal_position(
    obs: np.ndarray,
    goal_colour=(255, 0, 0),
    color_space: str = "bgr",
    min_area: int = 50,
):
    img = obs.astype(np.uint8)

    # OpenCV images are usually BGR.
    if color_space.lower() == "bgr":
        target_bgr = np.array(goal_colour, dtype=np.float32)
    elif color_space.lower() == "rgb":
        target_bgr = np.array(goal_colour[::-1], dtype=np.float32)
        img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
    else:
        raise ValueError("color_space must be 'bgr' or 'rgb'")

    b, g, r = cv.split(img)

    # Detect color dominance.
    dominant_channel = int(np.argmax(target_bgr))

    if dominant_channel == 0:      # blue-ish
        mask = (b > r + 25) & (b > g + 25) & (b > 60)
    elif dominant_channel == 1:    # green-ish
        mask = (g > r + 25) & (g > b + 25) & (g > 60)
    else:                          # red-ish
        mask = (r > g + 25) & (r > b + 25) & (r > 60)

    mask = mask.astype(np.uint8) * 255

    # Clean mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    if not contours:
        raise ValueError("No contours found for the specified goal colour.")

    contour = max(contours, key=cv.contourArea)

    if cv.contourArea(contour) < min_area:
        raise ValueError("No contours found for the specified goal colour.")

    (x, y), _ = cv.minEnclosingCircle(contour)

    return int(round(x)), int(round(y))


def extract_bolt_position_real(
    obs: np.ndarray,
    threshold: int = 40,
    min_area: int = 20,
):
    img = obs.astype(np.uint8)

    # Convert to grayscale if image is RGB/BGR
    if img.ndim == 3:
        gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    else:
        gray = img

    # Slight blur helps reduce small texture/noise
    gray_blur = cv.GaussianBlur(gray, (5, 5), 0)
    mask = (gray_blur > threshold).astype(np.uint8) * 255

    # Sometimes the object is darker than the background.
    # Keep the polarity with fewer foreground pixels.
    inv_mask = cv.bitwise_not(mask)

    if np.count_nonzero(inv_mask) < np.count_nonzero(mask):
        mask = inv_mask

    # Clean mask
    kernel = np.ones((3, 3), np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

    # Connected components
    _, _, stats, _ = cv.connectedComponentsWithStats(mask)

    # Choose largest non-background component
    areas = stats[1:, cv.CC_STAT_AREA]
    largest_label = 1 + np.argmax(areas)

    if areas.max() < min_area:
        raise ValueError("No sufficiently large object detected.")

    x = stats[largest_label, cv.CC_STAT_LEFT]
    y = stats[largest_label, cv.CC_STAT_TOP]
    w = stats[largest_label, cv.CC_STAT_WIDTH]
    h = stats[largest_label, cv.CC_STAT_HEIGHT]

    return x + w/2, y+h/2



def extract_bolt_position_sim(
    obs: np.ndarray,
    min_area: int = 20,
    lower_blue=(90, 40, 40),
    upper_blue=(140, 255, 255),
):
    img = obs.astype(np.uint8)

    # Convert RGB to HSV
    hsv = cv.cvtColor(img, cv.COLOR_RGB2HSV)

    # Mask blue-ish pixels
    mask = cv.inRange(
        hsv,
        np.array(lower_blue, dtype=np.uint8),
        np.array(upper_blue, dtype=np.uint8),
    )

    # Clean mask
    kernel = np.ones((3, 3), np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

    # Connected components
    num_labels, _, stats, _ = cv.connectedComponentsWithStats(mask)

    if num_labels <= 1:
        raise ValueError("No sufficiently large blue object detected.")

    # Choose largest non-background component
    areas = stats[1:, cv.CC_STAT_AREA]
    largest_label = 1 + np.argmax(areas)

    if areas.max() < min_area:
        raise ValueError("No sufficiently large blue object detected.")

    x = stats[largest_label, cv.CC_STAT_LEFT]
    y = stats[largest_label, cv.CC_STAT_TOP]
    w = stats[largest_label, cv.CC_STAT_WIDTH]
    h = stats[largest_label, cv.CC_STAT_HEIGHT]

    cx = x + w / 2
    cy = y + h / 2

    return cx, cy


class EasyGoalLoggingPolicy:

    def __init__(
        self,
        action_scale: float = 1.0,
        max_action: float = 1.0,
        calibration_action_scale: float = 1,
        min_movement: float = 1e-3,
        exploration_prob: float = 0.05,
        seed: int | None = None,
        simulation: bool = False,
    ):
        self.action_scale = action_scale
        self.max_action = max_action
        self.simulation = simulation
        self.calibration_action_scale = calibration_action_scale
        self.min_movement = min_movement
        self.exploration_prob = exploration_prob

        self.rng = np.random.default_rng(seed)

        # Calibration actions used before we know the local action coordinate system
        s = calibration_action_scale
        self.calibration_actions = [
            np.array([ s,  0.0], dtype=np.float32),
            np.array([-s,  0.0], dtype=np.float32),
            np.array([0.0,  s], dtype=np.float32),
            np.array([0.0, -s], dtype=np.float32),
        ]

    def restart(self):

        self.calibration_index = 0
        self.actions = []
        self.displacements = []

        self.prev_bolt_pos = None
        self.prev_action = None
        self.M = None

    def _clip_action(self, action):
        return np.clip(action, -self.max_action, self.max_action).astype(np.float32)

    def _update_model(self, bolt_pos):
        """
        Uses the previous action and the newly observed ball position to update M.
        """
        if self.prev_bolt_pos is None or self.prev_action is None:
            return

        displacement = bolt_pos- self.prev_bolt_pos

        if np.linalg.norm(displacement) < self.min_movement:
            return

        self.actions.append(self.prev_action.copy())
        self.displacements.append(displacement.copy())

        # Need at least two non-collinear actions for a 2D mapping
        if len(self.actions) < 2:
            return

        A = np.stack(self.actions, axis=0)          # shape: (N, 2)
        D = np.stack(self.displacements, axis=0)    # shape: (N, 2)

        B, _, _, _ = np.linalg.lstsq(A, D, rcond=None)

        # Convert to displacement ≈ M @ action
        self.M = B.T

    def act(self, obs):

        if self.simulation:
            bolt_x, bolt_y = extract_bolt_position_sim(obs)
        else:
            bolt_x, bolt_y = extract_bolt_position_real(obs)
        goal_x, goal_y = extract_goal_position(obs)
        bolt_pos = np.array([bolt_x, bolt_y], dtype=np.float32)
        goal_pos = np.array([goal_x, goal_y], dtype=np.float32)

        # Update model from the last transition
        self._update_model(bolt_pos)

        # First perform calibration actions
        if self.calibration_index < len(self.calibration_actions):
            action = self.calibration_actions[self.calibration_index]
            self.calibration_index += 1

            self.prev_bolt_pos = bolt_pos
            self.prev_action = action

            return self._clip_action(action)

        # Direction we want to move in image space
        to_goal = goal_pos - bolt_pos
        distance = np.linalg.norm(to_goal)

        if distance < 1e-6:
            action = np.zeros(2, dtype=np.float32)
        else:
            desired_displacement = self.action_scale * to_goal / distance

            # Find action that should produce that image-space displacement
            action = np.linalg.pinv(self.M) @ desired_displacement

        action = self._clip_action(action)

        self.prev_bolt_pos = bolt_pos
        self.prev_action = action
        return action



def main() -> None:
    """Run SAPIEN simulation with configurable robot loading."""

    # Configure integrated backend with robot loading option
    integrated_config = IntegratedConfig(
        backend=IntegratedBackend.SAPIEN,
        width=640,
        height=480,
        extra={
            # # Controller configuration
            # "max_speed_ms": 0.5,
            # "acceleration_time": 0.3,
            # "deceleration_time": 0.2,
            # "controller_kp": 0.1,
            # "controller_kd": 0.1,
            
            # Viewer configuration
            "use_viewer": False,
        },
    )
    
    # Create simulation config using integrated backend
    sim_config = SimulationConfig(
        use_integrated=True,
        integrated=integrated_config,
    )

    # Create environment
    env = SimulationEnv(
        config=sim_config,
        render_mode="opencv",
        max_episode_steps=100,
        action_space_type=ActionSpaceType.CARTESIAN,
    )
    policy = EasyGoalLoggingPolicy(action_scale=10,  simulation=True)
    for episode in range(100):
        obs, info = env.reset()
        env.render()
        # policy.reset()
        done = False
        policy.restart()
        while not done:
            try:
                action = policy.act(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                env.render(visualize=True)
                cv.waitKey(400) # Wait 1000ms between frames
                done = terminated or truncated
            except:
                done = True


if __name__ == "__main__":
    main()
