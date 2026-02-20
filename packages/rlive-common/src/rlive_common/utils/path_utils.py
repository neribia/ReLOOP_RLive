"""Path utility functions for project root detection."""
import sys
from pathlib import Path


def find_project_root(caller_file: str | None = None) -> Path:
    """
    Finds the package root directory (not the workspace root).

    Args:
        caller_file: The __file__ attribute of the calling module.
                    If None, uses this module's __file__.

    Priority order:
    1. PyInstaller bundles: Uses sys._MEIPASS
    2. Package root: Searches for directory containing src/<package_dir>
    3. Fallback: Any pyproject.toml (closest to config.py)
    4. Final fallback: Relative directory structure

    Returns:
        Path: The package root directory (e.g., packages/rlive-common)
    """
    # PyInstaller Bundle
    if hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)

    resolved = Path(caller_file or __file__).resolve()

    # Derive package directory name (e.g., 'rlive_common')
    # Path structure: .../packages/rlive-*/src/rlive_*/config/config.py
    try:
        package_dir = resolved.parents[1].name  # rlive_common
    except (IndexError, AttributeError):
        package_dir = None

    # 1) Search for package root: ancestor containing src/<package_dir>
    if package_dir:
        current = resolved.parent
        for _ in range(10):  # Max 10 levels up
            if (current / "src" / package_dir).exists():
                return current
            if current.parent == current:
                break
            current = current.parent

    # 2) Fallback: Search for any pyproject.toml (closest one)
    current = resolved.parent
    for _ in range(10):  # Max 10 levels up
        if (current / "pyproject.toml").exists():
            return current
        if current.parent == current:
            break
        current = current.parent

    # 3) Final fallback: Relative to config.py location
    # For packages/rlive-*/src/rlive_*/config/config.py -> go up 3 levels to package root
    try:
        return resolved.parents[3]
    except IndexError:
        # If the path is shallower than expected, fall back to the immediate parent
        return resolved.parent
