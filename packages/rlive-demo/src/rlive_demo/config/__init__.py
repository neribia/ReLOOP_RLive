"""Configuration module for rlive-demo package."""

from pathlib import Path

# Base directory for the entire repository (two levels up from this file)
REPO_ROOT = Path(__file__).parent.parent.parent.parent

# Base directory of the rlive-demo package
PACKAGE_DIR = Path(__file__).parent.parent.parent
PACKAGE_ROOT = PACKAGE_DIR.parent

# Resources directory
RESOURCES_DIR = PACKAGE_ROOT / "resources"


__all__ = [
    "PACKAGE_DIR",
    "REPO_ROOT",
    "RESOURCES_DIR",
]
