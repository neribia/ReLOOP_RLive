import unittest
import logging


class TestLogger(unittest.TestCase):
    def setUp(self):
        # Reset the CONFIGURED flag before each test
        import rlive_common.utils.logger as logger_module
        logger_module.CONFIGURED = False

    def test_get_logger_returns_logger(self):
        from rlive_common.utils import get_logger
        logger = get_logger(__name__)

        self.assertIsInstance(logger, logging.Logger)

    def test_get_logger_with_name(self):
        from rlive_common.utils import get_logger
        logger = get_logger("test_module")

        self.assertEqual(logger.name, "test_module")

    def test_get_logger_without_name(self):
        from rlive_common.utils import get_logger
        logger = get_logger()

        self.assertIsNotNone(logger.name)

    def test_format_styles_exist(self):
        from rlive_common.utils.logger import FORMAT_STYLES

        self.assertIn("simple", FORMAT_STYLES)
        self.assertIn("detailed", FORMAT_STYLES)
        self.assertIn("color", FORMAT_STYLES)
        self.assertIn("thread", FORMAT_STYLES)
        self.assertIn("json", FORMAT_STYLES)

    def test_date_styles_exist(self):
        from rlive_common.utils.logger import DATE_STYLES

        self.assertIn("short", DATE_STYLES)
        self.assertIn("long", DATE_STYLES)
        self.assertIn("iso", DATE_STYLES)
        self.assertIn("date_only", DATE_STYLES)
        self.assertIn("time_ms", DATE_STYLES)

    def test_setup_logger_with_valid_formatter(self):
        from rlive_common.utils.logger import setup_logger

        # Should not raise
        setup_logger(formatter="simple")

    def test_setup_logger_with_invalid_formatter(self):
        from rlive_common.utils.logger import setup_logger

        with self.assertRaises(ValueError):
            setup_logger(formatter="invalid_formatter")

    def test_setup_logger_with_invalid_date_style(self):
        from rlive_common.utils.logger import setup_logger

        with self.assertRaises(ValueError):
            setup_logger(date_style="invalid_date_style")

    def test_setup_logger_sets_configured_flag(self):
        import rlive_common.utils.logger as logger_module
        from rlive_common.utils.logger import setup_logger

        self.assertFalse(logger_module.CONFIGURED)
        setup_logger()
        self.assertTrue(logger_module.CONFIGURED)

    def test_logger_levels(self):
        from rlive_common.utils import get_logger

        logger = get_logger("test_levels")

        # These should not raise
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")
