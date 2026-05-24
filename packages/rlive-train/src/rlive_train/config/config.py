from pathlib import Path

TRAIN_ROOT = Path(__file__).resolve().parent.parent.parent.parent
# The root of the package is the rlive-train directory (where pyproject.toml is)
# __file__ is in src/rlive_train/config/config.py
# parent: config
# parent.parent: rlive_train
# parent.parent.parent: src
# parent.parent.parent.parent: rlive-train

RESOURCES_DIR = TRAIN_ROOT / "resources"
LOGS_DIR = RESOURCES_DIR / "logs"
MODELS_DIR = RESOURCES_DIR / "models"

