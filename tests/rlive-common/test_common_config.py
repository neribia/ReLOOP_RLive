import unittest


class TestCommonConfig(unittest.TestCase):
    def test_package_name(self):
        from rlive_common.config import config as cfg
        self.assertEqual(cfg.PACKAGE_NAME, "rlive-common")

    def test_app_home_exists(self):
        from rlive_common.config import config as cfg
        self.assertIsNotNone(cfg.APP_HOME)

    def test_bundle_dir_exists(self):
        from rlive_common.config import config as cfg
        self.assertIsNotNone(cfg.BUNDLE_DIR)

    def test_debug_default(self):
        from rlive_common.config import config as cfg
        # Default should be False unless DEBUG env var is set
        self.assertIsInstance(cfg.DEBUG, bool)

    def test_logging_level_is_string(self):
        from rlive_common.config import config as cfg
        self.assertIsInstance(cfg.LOGGING_LEVEL, str)
        self.assertIn(cfg.LOGGING_LEVEL, ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])

    def test_logging_format_is_string(self):
        from rlive_common.config import config as cfg
        self.assertIsInstance(cfg.LOGGING_FORMAT, str)

    def test_logging_date_is_string(self):
        from rlive_common.config import config as cfg
        self.assertIsInstance(cfg.LOGGING_DATE, str)

    def test_logging_stream_is_bool(self):
        from rlive_common.config import config as cfg
        self.assertIsInstance(cfg.LOGGING_STREAM, bool)
