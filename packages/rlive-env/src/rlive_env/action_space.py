"""Action space transformations for converting environment actions to robot commands.

This module provides pluggable action space transformers that convert agent actions
(from various spaces like Discrete, Box, etc.) into the robot action format
[heading, speed, duration] required by the SpheroBolt.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Type

import math
import numpy as np
import gymnasium as gym

from rlive_env.config import config as cfg
from rlive_common.utils import get_logger

logger = get_logger(__name__)


class ActionSpaceType(Enum):
    """Enumeration of supported action space types."""
    POLAR = "polar"
    CARTESIAN = "cartesian"
    CONTINUOUS_POLAR = "continuous_polar"


# Registry for action space transformers - uses ActionSpaceType enum as key
_ACTION_SPACE_REGISTRY: dict[ActionSpaceType, Type["BaseActionTransformer"]] = {}


def register_action_transformer(action_type: ActionSpaceType):
    """Decorator to register an action space transformer.

    Args:
        action_type: ActionSpaceType enum to register this transformer for
    """
    def decorator(cls: Type["BaseActionTransformer"]):
        _ACTION_SPACE_REGISTRY[action_type] = cls
        logger.debug(f"Registered action transformer: {action_type} -> {cls.__name__}")
        return cls
    return decorator


class BaseActionTransformer(ABC):
    """Abstract base class for action space transformers.

    Transforms actions from the agent's action space into the robot command format
    [heading, speed, duration].
    """

    @abstractmethod
    def get_action_space(self) -> gym.Space:
        """Return the gymnasium action space for this transformer.

        Returns:
            gymnasium.Space: The action space (e.g., Discrete, Box, etc.)
        """
        pass

    @abstractmethod
    def transform(self, action: Any) -> np.ndarray:
        """Transform agent action to robot command format.

        Args:
            action: Action from the agent's action space

        Returns:
            np.ndarray: Array with [heading (0-359), speed (0-255), duration (seconds)]

        Raises:
            ValueError: If action has invalid shape or type
        """
        pass

    def _calculate_heading_from_vector(self, x: float, y: float) -> int:
        """Calculate heading angle from 2D vector coordinates.

        Args:
            x: X component of vector
            y: Y component of vector
        Returns:
            int: Heading angle in degrees, range [-179, 180]
        """
        heading_rad = math.atan2(y, x)
        return self._normalize_heading(math.degrees(heading_rad))

    def _normalize_heading(self, heading: float) -> int:
        """Normalize heading to range [-179, 180].

        Converts heading value to the valid range, treating -180 as 180.

        Args:
            heading: Raw heading angle in degrees

        Returns:
            int: Normalized heading in range [-179, 180]
        """
        # Normalize to [-180, 180] first
        heading = ((heading + 180) % 360) - 180

        # Convert -180 to 180
        if heading == -180:
            heading = 180

        return int(round(heading))



@register_action_transformer(ActionSpaceType.POLAR)
class PolarActionTransformer(BaseActionTransformer):
    """Transform discrete heading actions to robot commands.

    The agent provides a heading in range [-179, 180] degrees, and speed/duration
    are configured defaults.
    """

    def get_action_space(self) -> gym.Space:
        """Return a discrete action space for heading angles in range [-179, 180]."""
        return gym.spaces.Discrete(360, start=-179)

    def transform(self, action: Any) -> np.ndarray:
        """Transform heading action to [heading, speed, duration].

        Args:
            action: Integer heading (-179 - 180) or iterable with single element

        Returns:
            np.ndarray: [heading, speed, duration]

        Raises:
            ValueError: If action is not a valid heading in [-179, 180]
        """
        try:
            if isinstance(action, (list, tuple, np.ndarray)):
                if len(action) != 1:
                    raise ValueError(f"Expected 1 action value, got {len(action)}")
                heading = int(round(action[0]))
            elif isinstance(action, (int, np.integer)):
                heading = int(action)
            else:
                raise ValueError(f"Cannot convert {type(action)} to heading")

            # Validate heading range [-179, 180]
            if not (-179 <= heading <= 180):
                raise ValueError(f"Heading must be in range [-179, 180], got {heading}")

            speed = int(cfg.SPHEROBOLTPLUS_SPEED)
            duration = float(cfg.SPHEROBOLTPLUS_DURATION)

            logger.debug(f"Transformed action {action} -> [{heading}, {speed}, {duration}]")
            return np.array([heading, speed, duration], dtype=np.float32)

        except (TypeError, ValueError) as e:
            logger.error(f"Failed to transform action {action}: {e}")
            raise ValueError(f"Invalid action format: {e}") from e


@register_action_transformer(ActionSpaceType.CARTESIAN)
class CartesianActionTransformer(BaseActionTransformer):
    """Transform continuous (x, y) velocity to robot heading and speed.

    The agent provides 2D velocity [vx, vy], which is converted to:
    - heading: angle from atan2(vy, vx)
    - speed: magnitude of velocity vector scaled to [0, 255]
    - duration: configured default
    """

    SPEED_SCALING_FACTOR: float = 100.0  # Scale velocity magnitude to speed (0-255)

    def get_action_space(self) -> gym.Space:
        """Return a continuous 2D velocity action space."""
        # Velocity components in range [-1, 1]
        return gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    def transform(self, action: Any) -> np.ndarray:
        """Transform [vx, vy] to [heading, speed, duration].

        Args:
            action: 2D velocity vector [vx, vy]

        Returns:
            np.ndarray: [heading (0-359), speed (0-255), duration]

        Raises:
            ValueError: If action is not a 2D vector
        """
        try:
            action = np.asarray(action, dtype=np.float32)

            if action.shape != (2,):
                raise ValueError(f"Expected 2D velocity, got shape {action.shape}")

            vx, vy = action[0], action[1]

            # Calculate heading from atan2 (normalize to [-179, 180])
            heading = self._calculate_heading_from_vector(vx, vy)

            # Calculate speed from magnitude, scaled to [0, 255]
            magnitude = math.sqrt(vx ** 2 + vy ** 2)
            speed = int(round(min(magnitude * self.SPEED_SCALING_FACTOR, 255)))

            duration = float(cfg.SPHEROBOLTPLUS_DURATION)

            logger.debug(f"Transformed velocity [{vx}, {vy}] -> [{heading}, {speed}, {duration}]")
            return np.array([heading, speed, duration], dtype=np.float32)

        except (TypeError, ValueError) as e:
            logger.error(f"Failed to transform action {action}: {e}")
            raise ValueError(f"Invalid action format: {e}") from e


@register_action_transformer(ActionSpaceType.CONTINUOUS_POLAR)
class ContinuousPolarActionTransformer(BaseActionTransformer):
    """Transform continuous (x, y) position to robot heading only.

    The agent provides 2D position [x, y], which is converted to a heading angle.
    Speed and duration are configured defaults.

    This transformer is useful when the agent learns to point towards targets.
    The heading is calculated from atan2(y, x).
    """

    def get_action_space(self) -> gym.Space:
        """Return a continuous 2D position action space.

        Returns:
            gymnasium.Space: Box space with 2D coordinates in range [-1, 1]
        """
        # Position components in range [-1, 1] (normalized coordinates)
        return gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    def transform(self, action: Any) -> np.ndarray:
        """Transform [x, y] position to [heading, speed, duration].

        Calculates heading as the angle from origin to the (x, y) position.
        Speed and duration are configured defaults.

        Args:
            action: 2D position vector [x, y] in range [-1, 1]

        Returns:
            np.ndarray: [heading (-179 - 180), speed (0-255), duration]

        Raises:
            ValueError: If action is not a 2D vector
        """
        try:
            action = np.asarray(action, dtype=np.float32)

            if action.shape != (2,):
                raise ValueError(f"Expected 2D position, got shape {action.shape}")

            x, y = action[0], action[1]

            # atan2 returns angle in radians from -pi to pi, normalize to [-179, 180]
            heading = self._calculate_heading_from_vector(x, y)

            # Use configured defaults for speed and duration
            speed = int(cfg.SPHEROBOLTPLUS_SPEED)
            duration = float(cfg.SPHEROBOLTPLUS_DURATION)

            logger.debug(f"Transformed position [{x}, {y}] -> heading {heading}° -> [{heading}, {speed}, {duration}]")
            return np.array([heading, speed, duration], dtype=np.float32)

        except (TypeError, ValueError) as e:
            logger.error(f"Failed to transform action {action}: {e}")
            raise ValueError(f"Invalid action format: {e}") from e


def get_action_transformer(action_space_type: ActionSpaceType | str) -> BaseActionTransformer:
    """Factory function to get an action transformer instance.

    Args:
        action_space_type: ActionSpaceType enum or string name of transformer
                          (e.g., ActionSpaceType.POLAR or 'polar')

    Returns:
        BaseActionTransformer: Instance of the requested transformer

    Raises:
        ValueError: If transformer type is not registered
    """
    # Convert string to ActionSpaceType enum if needed
    if isinstance(action_space_type, str):
        try:
            action_space_type = ActionSpaceType(action_space_type)
        except ValueError as e:
            available = [t.value for t in ActionSpaceType]
            raise ValueError(f"Unknown action space type '{action_space_type}'. "
                           f"Available: {available}") from e

    if action_space_type not in _ACTION_SPACE_REGISTRY:
        available = list(_ACTION_SPACE_REGISTRY.keys())
        raise ValueError(f"Unknown action space type '{action_space_type}'. "
                        f"Available: {available}")

    return _ACTION_SPACE_REGISTRY[action_space_type]()
