"""SB3 Evaluation - SAPIEN Simulation

This example demonstrates how to evaluate an existing Stable Baselines 3 agent
using the SimulationEnv with SAPIEN integrated backend.
"""
import sys
import gymnasium as gym

# Spoof the gym module to suppress unmaintained gym warnings
sys.modules["gym"] = gym

from stable_baselines3 import PPO

from rlive_train.config.config import LOGS_DIR
from rlive_common.utils import get_logger
from rlive_train.utils.sb3_env import EvalVecBackend, build_sim_eval_env

logger = get_logger(__name__)

# Replace with the actual run folder name you want to evaluate
RUN_ID = "PPO_Sapien_YYYYMMDD_HHMMSS"


def main() -> None:
    env = build_sim_eval_env(n_stack=4, backend=EvalVecBackend.DUMMY)

    # Load from the specific run's best_model directory
    model_path = LOGS_DIR / RUN_ID / "best_model" / "best_model.zip"

    if not model_path.exists():
        logger.error(f"Could not find model at {model_path}. Please update RUN_ID.")
        return

    logger.info(f"Loading model from {model_path}")
    model = PPO.load(model_path, env=env)

    logger.info("Evaluating the trained model for 5 episodes...")
    for episode in range(5):
        obs = env.reset()
        # env.render(visualize=True)

        done = False
        step_count = 0
        episode_reward = 0.0

        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, dones, info = env.step(action)
            episode_reward += reward
            print(action)
            # Visualize the current state
            image = obs
            # env.render(visualize=True)
            # cv2.waitKey(100)  # Optional small delay to regulate playback speed

            done = dones[0]
            step_count += 1

        # logger.info(f"Episode {episode + 1}/5 completed - Steps: {step_count}, Reward: {episode_reward:.3f}")

    env.close()


if __name__ == "__main__":
    main()

