from pathlib import Path

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import time


# Define your 4 fixed actions (adjust values to your action space shape)
FIXED_ACTIONS = [
    np.array([-0.4, -0.6], dtype=np.float32),
    np.array([-1.0, -0.5], dtype=np.float32),
    np.array([0.5, -0.8], dtype=np.float32),
    np.array([-0.5, -0.5], dtype=np.float32),
    ]

RANDOM_SEED = 42

# Fixed goal position applied on every reset (pixel coordinates x, y)
FIXED_GOAL: tuple[int, int] = (160, 120)


_LABEL_HEADER = "filename, action_x, action_y, reward\n"


def _obs_filename(idx: int) -> str:
    return f"obs_{idx:04d}.png"


def _format_label_row(idx: int, label, reward) -> str:
    if label is None or isinstance(label, str):
        action_str = "None, None"
    else:
        action_str = f"{label[0]:.4f}, {label[1]:.4f}"
    reward_str = "None" if reward is None or isinstance(reward, str) else f"{reward:.4f}"
    return f"{_obs_filename(idx)}, {action_str}, {reward_str}\n"


def get_action_list(action_space, fixed_actions: list = FIXED_ACTIONS, n_random: int = 96) -> list:
    """
    Returns a list of actions: fixed_actions first, then n_random sampled ones.
    Uses RANDOM_SEED for reproducibility.
    """
    action_space.seed(RANDOM_SEED)
    random_actions = [action_space.sample() for _ in range(n_random)]
    return list(fixed_actions) + random_actions


def run_action_list(env, action_list: list, reset_options: dict | None = None, save: bool = False, save_dir: str = "image_folder") -> tuple[list, list, list]:
    """
    Runs a pre-built action list on any gym-compatible env.
    Calls env.reset() once at the start and again on termination/truncation.

    The goal is always fixed to FIXED_GOAL so every episode is comparable.
    Any ``initial_state`` passed via reset_options is preserved.

    Args:
        env: Gymnasium-compatible environment.
        action_list: List of actions to execute.
        reset_options: Optional dict passed to env.reset(options=...) on every reset.
                       Use e.g. {"initial_state": PhysicsState(...)} to fix spawn position.
        save: If True, saves each image and updates labels.txt instantly after each step.
        save_dir: Directory to save images and labels to (only used if save=True).

    Returns:
        obs_list, actions_list, rewards_list  — all length len(action_list) + 1, fully aligned.
    """
    options: dict = dict(reset_options) if reset_options else {}
    options.setdefault("goal_position", FIXED_GOAL)

    obs_list, actions_list, rewards_list = [], [], []
    total = len(action_list)
    checkpoints = {int(total * p / 10) for p in range(1, 11)}

    if save:
        folder = Path(save_dir)
        folder.mkdir(parents=True, exist_ok=True)
        lf = (folder / "labels.txt").open("w")
        lf.write(_LABEL_HEADER)

    obs, info = env.reset(options=options)
    obs_list.append(obs)
    rewards_list.append(None)
    actions_list.append(None)

    if save:
        cv.imwrite(str(folder / _obs_filename(0)), cv.cvtColor(obs, cv.COLOR_RGB2BGR))
        lf.write(_format_label_row(0, None, None))
        lf.flush()

    print(f"Starting {total} steps...")

    for i, action in enumerate(action_list):
        obs, reward, terminated, truncated, info = env.step(action)
        obs_list.append(obs)
        actions_list.append(action)
        rewards_list.append(reward)

        if save:
            idx = i + 1
            cv.imwrite(str(folder / _obs_filename(idx)), cv.cvtColor(obs, cv.COLOR_RGB2BGR))
            lf.write(_format_label_row(idx, action, reward))
            lf.flush()

        steps_done = i + 1
        if steps_done in checkpoints:
            pct = int(steps_done / total * 100)
            bar = ("█" * (pct // 10)).ljust(10)
            print(f"  [{bar}] {pct:>3}%  ({steps_done}/{total} steps)")

    if save:
        lf.close()

    env.close()
    return obs_list, actions_list, rewards_list


def plot_image_grid(image_list, label_list, reward_list, rows=4, cols=4):
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes_flat = axes.flatten()

    for i in range(len(axes_flat)):
        ax = axes_flat[i]
        if i < len(image_list):
            ax.imshow(image_list[i])
            label = label_list[i]
            reward = reward_list[i]
            if label is None or isinstance(label, str):
                action_str = "reset"
            else:
                action_str = f"({label[0]:.2f}, {label[1]:.2f})"
            reward_str = "—" if reward is None or isinstance(reward, str) else f"{reward:.3f}"
            ax.set_title(f"{action_str}\nr={reward_str}", fontsize=9)
        ax.axis('off')

    plt.tight_layout()
    plt.show()

def save_images(image_list, label_list, reward_list, save_dir="image_folder"):
    folder = Path(save_dir)
    folder.mkdir(parents=True, exist_ok=True)

    for idx, image in enumerate(image_list):
        img_path = folder / _obs_filename(idx)
        cv.imwrite(str(img_path), cv.cvtColor(image, cv.COLOR_RGB2BGR))

    label_file = folder / "labels.txt"
    with label_file.open("w") as f:
        f.write(_LABEL_HEADER)
        for idx, (label, reward) in enumerate(zip(label_list, reward_list)):
            f.write(_format_label_row(idx, label, reward))

    print(f"Successfully saved {len(image_list)} images to {folder}")