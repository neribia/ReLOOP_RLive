"""Ball location class for tracking the Sphero ball position."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class BallLocation:
    """Represents the (x, y) position of the ball in pixel coordinates.

    Attributes:
        x: Horizontal position in pixels (0 = left edge).
        y: Vertical position in pixels (0 = top edge).
    """

    x: int
    y: int

    def distance_to(self, goal: tuple[int, int]) -> float:
        """Calculate Euclidean distance to a goal position.

        Args:
            goal: Target position as (x, y) tuple.

        Returns:
            Distance in pixels.
        """
        goal_x, goal_y = goal
        return math.sqrt((self.x - goal_x) ** 2 + (self.y - goal_y) ** 2)

    def is_within_radius(self, goal: tuple[int, int], radius: float) -> bool:
        """Check if ball is within a given radius of the goal.

        Args:
            goal: Target position as (x, y) tuple.
            radius: Threshold radius in pixels.

        Returns:
            True if ball is within radius of goal.
        """
        return self.distance_to(goal) <= radius

    def as_tuple(self) -> tuple[int, int]:
        """Return position as (x, y) tuple."""
        return (self.x, self.y)

