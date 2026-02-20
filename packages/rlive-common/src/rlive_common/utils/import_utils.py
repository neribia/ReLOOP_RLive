import os


def get_rgb_env(var_name: str, default: int) -> int:
    """
    Read an environment variable and ensure it is a valid RGB value (0-255).
    Falls back to default if invalid and clamps to valid range.
    """
    try:
        value = int(os.getenv(var_name, str(default)))
    except (TypeError, ValueError):
        return default

    # Clamp to valid RGB range
    return max(0, min(255, value))