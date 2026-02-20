import unittest
from pathlib import Path


class TestCommonConfig(unittest.TestCase):
    def test_package_name(self):
        from rlive_common.config import config as cfg
        self.assertEqual(cfg.PACKAGE_NAME, "rlive-common")

    def test_project_root_exists(self):
        from rlive_common.config import config as cfg
        self.assertIsNotNone(cfg.PROJECT_ROOT)
        self.assertIsInstance(cfg.PROJECT_ROOT, Path)
        self.assertTrue(cfg.PROJECT_ROOT.exists())

    def test_project_root_contains_pyproject_toml(self):
        from rlive_common.config import config as cfg
        # The package root should contain pyproject.toml
        self.assertTrue((cfg.PROJECT_ROOT / "pyproject.toml").exists())

    def test_project_root_is_package_root(self):
        from rlive_common.config import config as cfg
        # PROJECT_ROOT should point to the package root (packages/rlive-common)
        # which contains src/rlive_common
        self.assertTrue((cfg.PROJECT_ROOT / "src" / "rlive_common").exists())
        # Verify it's not the workspace root
        self.assertFalse((cfg.PROJECT_ROOT / "packages").exists())

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

    def test_logging_file_path_is_in_project_root(self):
        from rlive_common.config import config as cfg
        self.assertIsInstance(cfg.LOGGING_FILE_PATH, Path)
        # Logging path should be relative to PROJECT_ROOT
        self.assertTrue(str(cfg.LOGGING_FILE_PATH).startswith(str(cfg.PROJECT_ROOT)))

