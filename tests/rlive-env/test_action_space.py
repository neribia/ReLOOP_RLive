"""Tests for action space transformers."""

import unittest
import numpy as np
import gymnasium as gym

from rlive_env.action_space import (
    ActionSpaceType,
    BaseActionTransformer,
    PolarActionTransformer,
    CartesianActionTransformer,
    ContinuousPolarActionTransformer,
    get_action_transformer,
    _ACTION_SPACE_REGISTRY,
)


class TestPolarActionTransformer(unittest.TestCase):
    """Test suite for PolarActionTransformer."""

    def setUp(self):
        """Create transformer instance for testing."""
        self.transformer = PolarActionTransformer()

    def test_get_action_space(self):
        """Test that action space is Discrete(360)."""
        space = self.transformer.get_action_space()
        self.assertIsInstance(space, gym.spaces.Discrete)
        self.assertEqual(space.n, 360)

    def test_transform_action_integer(self):
        """Test transforming integer heading action."""
        result = self.transformer.transform_action(90)

        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (3,))
        self.assertEqual(result[0], 90)  # heading
        self.assertGreater(result[1], 0)  # speed
        self.assertGreater(result[2], 0)  # duration

    def test_transform_action_list(self):
        """Test transforming list with single heading."""
        result = self.transformer.transform_action([180])
        self.assertEqual(result[0], 180)

    def test_transform_action_numpy_array(self):
        """Test transforming numpy array with single heading."""
        result = self.transformer.transform_action(np.array([270]))
        self.assertEqual(result[0], 270)

    def test_heading_range_valid(self):
        """Test valid heading values."""
        for heading in [0, 45, 90, 180, 270, 359]:
            result = self.transformer.transform_action(heading)
            self.assertEqual(result[0], heading)

    def test_heading_range_invalid_negative(self):
        """Test that negative heading raises ValueError."""
        with self.assertRaises(ValueError) as context:
            self.transformer.transform_action(-10)
        self.assertIn("range", str(context.exception).lower())

    def test_heading_range_invalid_too_large(self):
        """Test that heading >= 360 raises ValueError."""
        with self.assertRaises(ValueError) as context:
            self.transformer.transform_action(360)
        self.assertIn("range", str(context.exception).lower())

    def test_invalid_action_type(self):
        """Test that invalid action type raises ValueError."""
        with self.assertRaises(ValueError):
            self.transformer.transform_action("invalid")

    def test_invalid_action_list_length(self):
        """Test that list with wrong length raises ValueError."""
        with self.assertRaises(ValueError):
            self.transformer.transform_action([90, 100])

    def test_rounding(self):
        """Test that floating point headings are rounded correctly."""
        result = self.transformer.transform_action([90.4])
        self.assertEqual(result[0], 90)

        result = self.transformer.transform_action([90.6])
        self.assertEqual(result[0], 91)


class TestCartesianActionTransformer(unittest.TestCase):
    """Test suite for CartesianActionTransformer."""

    def setUp(self):
        """Create transformer instance for testing."""
        self.transformer = CartesianActionTransformer()

    def test_get_action_space(self):
        """Test that action space is Box with shape (2,)."""
        space = self.transformer.get_action_space()
        self.assertIsInstance(space, gym.spaces.Box)
        self.assertEqual(space.shape, (2,))
        self.assertTrue(np.all(space.low == -1.0))
        self.assertTrue(np.all(space.high == 1.0))

    def test_transform_velocity_east(self):
        """Test velocity pointing east (positive x)."""
        result = self.transformer.transform_action([1.0, 0.0])
        self.assertEqual(result[0], 0)  # East is 0 degrees
        self.assertGreater(result[1], 0)  # speed > 0
        self.assertGreater(result[2], 0)  # duration > 0

    def test_transform_velocity_north(self):
        """Test velocity pointing north (positive y)."""
        result = self.transformer.transform_action([0.0, 1.0])
        self.assertEqual(result[0], 90)  # North is 90 degrees

    def test_transform_velocity_west(self):
        """Test velocity pointing west (negative x)."""
        result = self.transformer.transform_action([-1.0, 0.0])
        self.assertEqual(result[0], 180)  # West is 180 degrees

    def test_transform_velocity_south(self):
        """Test velocity pointing south (negative y)."""
        result = self.transformer.transform_action([0.0, -1.0])
        self.assertEqual(result[0], 270)  # South is 270 degrees

    def test_transform_velocity_northeast(self):
        """Test velocity pointing northeast."""
        result = self.transformer.transform_action([1.0, 1.0])
        self.assertEqual(result[0], 45)  # Northeast is 45 degrees

    def test_speed_scaling(self):
        """Test that speed is scaled from magnitude."""
        result_small = self.transformer.transform_action([0.1, 0.0])
        result_large = self.transformer.transform_action([1.0, 0.0])
        self.assertLess(result_small[1], result_large[1])

    def test_speed_capped_at_255(self):
        """Test that speed is capped at 255."""
        result = self.transformer.transform_action([10.0, 10.0])
        self.assertLessEqual(result[1], 255)

    def test_invalid_shape(self):
        """Test that wrong shape raises ValueError."""
        with self.assertRaises(ValueError):
            self.transformer.transform_action([1.0])

        with self.assertRaises(ValueError):
            self.transformer.transform_action([1.0, 1.0, 1.0])

    def test_zero_velocity(self):
        """Test zero velocity is handled correctly."""
        result = self.transformer.transform_action([0.0, 0.0])
        self.assertEqual(result[1], 0)


class TestContinuousPolarActionTransformer(unittest.TestCase):
    """Test suite for ContinuousPolarActionTransformer."""

    def setUp(self):
        """Create transformer instance for testing."""
        self.transformer = ContinuousPolarActionTransformer()

    def test_get_action_space(self):
        """Test that action space is Box with shape (2,)."""
        space = self.transformer.get_action_space()
        self.assertIsInstance(space, gym.spaces.Box)
        self.assertEqual(space.shape, (2,))
        self.assertTrue(np.all(space.low == -1.0))
        self.assertTrue(np.all(space.high == 1.0))

    def test_transform_position_east(self):
        """Test position pointing east (positive x)."""
        result = self.transformer.transform_action([1.0, 0.0])
        self.assertEqual(result[0], 0)  # East is 0 degrees
        self.assertGreater(result[1], 0)
        self.assertGreater(result[2], 0)

    def test_transform_position_north(self):
        """Test position pointing north (positive y)."""
        result = self.transformer.transform_action([0.0, 1.0])
        self.assertEqual(result[0], 90)  # North is 90 degrees

    def test_transform_position_west(self):
        """Test position pointing west (negative x)."""
        result = self.transformer.transform_action([-1.0, 0.0])
        self.assertEqual(result[0], 180)  # West is 180 degrees

    def test_transform_position_south(self):
        """Test position pointing south (negative y)."""
        result = self.transformer.transform_action([0.0, -1.0])
        self.assertEqual(result[0], 270)  # South is 270 degrees

    def test_transform_position_northeast(self):
        """Test position pointing northeast."""
        result = self.transformer.transform_action([1.0, 1.0])
        self.assertEqual(result[0], 45)  # Northeast is 45 degrees

    def test_transform_position_northwest(self):
        """Test position pointing northwest."""
        result = self.transformer.transform_action([-1.0, 1.0])
        self.assertEqual(result[0], 135)  # Northwest is 135 degrees

    def test_transform_position_southeast(self):
        """Test position pointing southeast."""
        result = self.transformer.transform_action([1.0, -1.0])
        self.assertEqual(result[0], 315)  # Southeast is 315 degrees

    def test_transform_position_southwest(self):
        """Test position pointing southwest."""
        result = self.transformer.transform_action([-1.0, -1.0])
        self.assertEqual(result[0], 225)  # Southwest is 225 degrees

    def test_speed_from_config(self):
        """Test that speed comes from config (not magnitude)."""
        result_small = self.transformer.transform_action([0.1, 0.0])
        result_large = self.transformer.transform_action([1.0, 0.0])
        self.assertEqual(result_small[1], result_large[1])

    def test_invalid_shape(self):
        """Test that wrong shape raises ValueError."""
        with self.assertRaises(ValueError):
            self.transformer.transform_action([1.0])

        with self.assertRaises(ValueError):
            self.transformer.transform_action([1.0, 1.0, 1.0])

    def test_invalid_type(self):
        """Test that invalid type raises ValueError."""
        with self.assertRaises(ValueError):
            self.transformer.transform_action("invalid")

    def test_zero_position(self):
        """Test zero position is handled gracefully."""
        result = self.transformer.transform_action([0.0, 0.0])
        self.assertEqual(result.shape, (3,))
        self.assertIsInstance(result[0], (int, np.integer))


class TestActionTransformerFactory(unittest.TestCase):
    """Test suite for action transformer factory function."""

    def test_get_polar_transformer(self):
        """Test getting PolarActionTransformer via factory."""
        transformer = get_action_transformer('polar')
        self.assertIsInstance(transformer, PolarActionTransformer)

    def test_get_cartesian_transformer(self):
        """Test getting CartesianActionTransformer via factory."""
        transformer = get_action_transformer('cartesian')
        self.assertIsInstance(transformer, CartesianActionTransformer)

    def test_get_continuous_polar_transformer(self):
        """Test getting ContinuousPolarActionTransformer via factory."""
        transformer = get_action_transformer('continuous_polar')
        self.assertIsInstance(transformer, ContinuousPolarActionTransformer)

    def test_unknown_transformer(self):
        """Test that unknown transformer type raises ValueError."""
        with self.assertRaises(ValueError) as context:
            get_action_transformer('unknown_type')

        self.assertIn("Unknown", str(context.exception))
        self.assertIn("Available", str(context.exception))

    def test_registry_contains_all_transformers(self):
        """Test that all transformers are registered."""
        self.assertIn('polar', _ACTION_SPACE_REGISTRY)
        self.assertIn('cartesian', _ACTION_SPACE_REGISTRY)
        self.assertIn('continuous_polar', _ACTION_SPACE_REGISTRY)

    def test_registry_values_are_classes(self):
        """Test that registry values are transformer classes."""
        for name, transformer_class in _ACTION_SPACE_REGISTRY.items():
            self.assertTrue(issubclass(transformer_class, BaseActionTransformer))


class TestActionSpaceConsistency(unittest.TestCase):
    """Test consistency across all transformers."""

    def test_all_transformers_return_three_element_array(self):
        """Test that all transformers return [heading, speed, duration]."""
        transformers = [
            PolarActionTransformer(),
            CartesianActionTransformer(),
            ContinuousPolarActionTransformer(),
        ]

        test_actions = [
            90,
            np.array([1.0, 0.0]),
            np.array([1.0, 0.0]),
        ]

        for transformer, action in zip(transformers, test_actions):
            result = transformer.transform_action(action)

            self.assertEqual(result.shape, (3,),
                           f"{transformer.__class__.__name__} didn't return 3-element array")

            heading, speed, duration = result
            self.assertIsInstance(heading, (int, np.integer))
            self.assertIsInstance(speed, (int, np.integer))
            self.assertIsInstance(duration, (int, np.integer, float, np.floating))

    def test_heading_in_valid_range(self):
        """Test that all transformers produce headings in [0, 360)."""
        # Test each transformer separately with appropriate inputs

        # Polar transformer - single heading values
        polar = PolarActionTransformer()
        for heading in [0, 45, 90, 180, 270, 359]:
            result = polar.transform_action(heading)
            self.assertTrue(0 <= result[0] < 360,
                          f"Polar heading {result[0]} out of range")

        # Cartesian transformer - 2D velocity vectors
        cartesian = CartesianActionTransformer()
        for x in [1, 0.5, 0, -0.5, -1]:
            result = cartesian.transform_action([x, 0.0])
            self.assertTrue(0 <= result[0] < 360,
                          f"Cartesian heading {result[0]} out of range")

        # Continuous polar transformer - 2D position vectors
        continuous = ContinuousPolarActionTransformer()
        for x in [1, 0.5, 0, -0.5, -1]:
            result = continuous.transform_action([x, 0.0])
            self.assertTrue(0 <= result[0] < 360,
                          f"Continuous polar heading {result[0]} out of range")

    def test_speed_in_valid_range(self):
        """Test that all transformers produce speeds in [0, 255]."""
        transformers = [
            PolarActionTransformer(),
            CartesianActionTransformer(),
            ContinuousPolarActionTransformer(),
        ]

        test_actions = [
            90,
            np.array([1.0, 1.0]),
            np.array([1.0, 1.0]),
        ]

        for transformer, action in zip(transformers, test_actions):
            result = transformer.transform_action(action)
            speed = result[1]
            self.assertTrue(0 <= speed <= 255,
                          f"Speed {speed} out of range for {transformer.__class__.__name__}")

    def test_duration_is_positive(self):
        """Test that all transformers produce positive durations."""
        transformers = [
            PolarActionTransformer(),
            CartesianActionTransformer(),
            ContinuousPolarActionTransformer(),
        ]

        test_actions = [
            90,
            np.array([1.0, 1.0]),
            np.array([1.0, 1.0]),
        ]

        for transformer, action in zip(transformers, test_actions):
            result = transformer.transform_action(action)
            duration = result[2]
            self.assertGreater(duration, 0,
                             f"Duration {duration} not positive for {transformer.__class__.__name__}")


if __name__ == '__main__':
    unittest.main()

