import math
import cv2 as cv
import numpy as np

from rlive_common.config import config as cfg
from rlive_common.core.ball_location import BallLocation


def compute_goal_radius(
    image_shape: tuple[int, ...],
    factor: float = cfg.GOAL_RADIUS_FACTOR,
) -> int:
    """Compute a goal radius in pixels that scales with the image diagonal.

    The diagonal is defined as √(width² + height²).  A *factor* of 0.0625
    yields 50 px for a 640×480 image (diagonal = 800 px), which matches the
    legacy hard-coded GOAL_RADIUS default.

    Args:
        image_shape: Image shape tuple – at minimum ``(height, width)`` or the
            full NumPy ``array.shape`` ``(height, width, channels)``.
        factor: Fraction of the diagonal to use as the radius.
            Defaults to ``cfg.GOAL_RADIUS_FACTOR``.

    Returns:
        Pixel radius as an integer (minimum 1).
    """
    h, w = image_shape[:2]
    diagonal = math.sqrt(w ** 2 + h ** 2)
    return max(1, int(diagonal * factor))


def draw_goal(
    image: np.ndarray,
    goal_position: tuple[int, int] | np.ndarray,
    goal_radius: int | None = None,
    goal_colour: tuple[int, int, int] | None = cfg.GOAL_COLOUR,
    goal_alpha: float | None = cfg.GOAL_ALPHA,
) -> np.ndarray:
    """Draw transparent goal indicator.

    If visual parameters are not provided, uses defaults from common config.
    When *goal_radius* is ``None`` (the default), the radius is computed
    automatically from the image size via :func:`compute_goal_radius` so that
    the circle scales with the resolution.

    Args:
        image: Original image.
        goal_position: Goal coordinates (x, y).
        goal_radius: Explicit radius in pixels.  Pass ``None`` (default) to
            derive the radius automatically from the image diagonal.
        goal_colour: Colour of the goal circle in (R, G, B).
        goal_alpha: Transparency factor (0.0 = invisible, 1.0 = fully visible).

    Returns:
        Annotated image with the goal overlay.
    """
    if goal_radius is None:
        goal_radius = compute_goal_radius(image.shape)
    annotated_image = image.copy()

    # Create overlay (same size as image)
    overlay = annotated_image.copy()

    thickness = -1  # Filled circle to allow transparency

    # Ensure position is a tuple of ints
    pos = (int(goal_position[0]), int(goal_position[1]))

    cv.circle(overlay, pos, goal_radius, goal_colour, thickness)

    # Blend overlay onto original
    cv.addWeighted(overlay, goal_alpha, annotated_image, 1 - goal_alpha, 0, annotated_image)

    # Draw outline for better visibility (optional)
    cv.circle(annotated_image, pos, goal_radius, (0, 180, 0), 2)

    return annotated_image


def annotate_image(
    image: np.ndarray,
    ball_location: BallLocation | None = None,
    goal_position: tuple[int, int] | np.ndarray | None = None,
    goal_radius: int | None = None,
    ball_radius: int | None = 15,
) -> np.ndarray | None:
    """Annotate image with goal and ball tracking information.

    Args:
        image: Base image to annotate
        ball_location: Optional current ball location
        goal_position: Optional target goal coordinate (x, y)
        goal_radius: Optional goal radius. If None, uses config default.

    Returns:
        Annotated copy of the image
    """
    if image is None:
        return None

    annotated = image.copy()

    # Draw ball if detected
    if ball_location is not None:
        pos = ball_location.as_tuple()
        cv.circle(annotated, pos, ball_radius, (255, 0, 0), 2)
        cv.circle(annotated, pos, 5, (255, 0, 0), -1)

        # Draw distance line to goal if both available
        if goal_position is not None:
            cv.line(annotated, pos, goal_position, (255, 255, 0), 1)
            distance = ball_location.distance_to(goal_position)
            mid_x = (pos[0] + goal_position[0]) // 2
            mid_y = (pos[1] + goal_position[1]) // 2
            cv.putText(
                annotated,
                f"{distance:.1f}px",
                (mid_x, mid_y - 10),
                cv.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                1,
            )

    return annotated

