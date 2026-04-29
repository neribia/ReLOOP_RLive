"""SB3 PPO Training - SAPIEN Simulation with Weights & Biases logging."""
import os
import sys
from datetime import datetime

import gymnasium

# Spoof gym to avoid unmaintained gym warning checks from optional dependencies.
sys.modules["gym"] = gymnasium

from stable_baselines3 import PPO

import wandb

from rlive_common.utils import get_logger
from rlive_train.config.config import LOGS_DIR
from rlive_train.utils.sb3_callbacks import build_wandb_eval_callbacks
from rlive_train.utils.sb3_env import EvalVecBackend, build_sim_eval_env, build_sim_train_env

logger = get_logger(__name__)


def main() -> None:

	run_name = f"PPO_Sapien_WandB_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
	run_dir = LOGS_DIR / run_name
	run_dir.mkdir(parents=True, exist_ok=True)

	tensorboard_log = str(run_dir / "tensorboard")
	best_model_dir = str(run_dir / "best_model")

	project_name = os.getenv("WANDB_PROJECT", "rlive-train")
	entity_name = os.getenv("WANDB_ENTITY")

	policy = "CnnPolicy"
	n_envs = 2
	n_steps = 2 * 20 * 2
	batch_size = 10
	n_epochs = 2
	ent_coef = 0.01
	learning_rate = 3e-4
	total_timesteps = 5_000

	hyperparams = {
		"policy": policy,
		"n_envs": n_envs,
		"n_steps": n_steps,
		"batch_size": batch_size,
		"n_epochs": n_epochs,
		"ent_coef": ent_coef,
		"learning_rate": learning_rate,
		"total_timesteps": total_timesteps,
	}

	logger.info("Initializing Weights & Biases run...")
	wandb_run = wandb.init(
		project=project_name,
		entity=entity_name,
		name=run_name,
		dir=str(run_dir),
		sync_tensorboard=True,
		monitor_gym=False,
		save_code=True,
		config=hyperparams,
	)

	env = build_sim_train_env(num_envs=n_envs, n_stack=4)
	eval_env = build_sim_eval_env(n_stack=4, backend=EvalVecBackend.DUMMY)

	try:
		logger.info(f"Initializing PPO Model with {policy}...")
		model = PPO(
			policy=policy,
			env=env,
			n_steps=n_steps,
			batch_size=batch_size,
			n_epochs=n_epochs,
			verbose=1,
			tensorboard_log=tensorboard_log,
			ent_coef=ent_coef,
			learning_rate=learning_rate,
		)

		callbacks = build_wandb_eval_callbacks(
			env_eval=eval_env,
			log_path=str(run_dir),
			save_path=best_model_dir,
			eval_freq=200,
			n_eval_episodes=5,
			wandb_model_save_path=best_model_dir,
			wandb_model_save_freq=200,
		)

		logger.info(f"Starting training for {total_timesteps:,} timesteps...")
		model.learn(
			total_timesteps=total_timesteps,
			callback=callbacks,
			tb_log_name="PPO_Sapien",
		)


	finally:
		model_path = str(run_dir / "last_model")
		if model is not None:
			model.save(model_path)
		logger.info(f"Model saved to {model_path}.zip")
		env.close()
		eval_env.close()
		wandb.finish()


if __name__ == "__main__":
	main()



