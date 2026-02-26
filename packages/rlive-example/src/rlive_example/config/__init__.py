"""Configuration module for rlive-example package."""

from pathlib import Path

# Base directory for the rlive-example workspace (four levels up from this file)
REPO_ROOT = Path(__file__).parent.parent.parent.parent

# Source directory (src) of the rlive-example package
PACKAGE_DIR = Path(__file__).parent.parent.parent
# Base directory of the rlive-example package
PACKAGE_ROOT = PACKAGE_DIR.parent

# Resources directory
RESOURCES_DIR = PACKAGE_ROOT / "resources"


__all__ = [
    "PACKAGE_DIR",
    "REPO_ROOT",
    "RESOURCES_DIR",
]
