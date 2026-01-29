import unittest
from pathlib import Path


class TestEnvConfig(unittest.TestCase):
    def test_project_root_exists(self):
        from rlive_env.config import config as cfg
        self.assertIsNotNone(cfg.PROJECT_ROOT)
        self.assertIsInstance(cfg.PROJECT_ROOT, Path)
        self.assertTrue(cfg.PROJECT_ROOT.exists())

    def test_project_root_contains_pyproject_toml(self):
        from rlive_env.config import config as cfg
        # The package root should contain pyproject.toml
        self.assertTrue((cfg.PROJECT_ROOT / "pyproject.toml").exists())

    def test_project_root_is_package_root(self):
        from rlive_env.config import config as cfg
        # PROJECT_ROOT should point to the package root (packages/rlive-env)
        # which contains src/rlive_env
        self.assertTrue((cfg.PROJECT_ROOT / "src" / "rlive_env").exists())
        # Verify it's not the workspace root
        self.assertFalse((cfg.PROJECT_ROOT / "packages").exists())

    def test_world_base_url_default(self):
        from rlive_env.config import config as cfg
        # Default value
        self.assertIsInstance(cfg.WORLD_BASE_URL, str)
        self.assertTrue(cfg.WORLD_BASE_URL.startswith("http"))

    def test_world_interface_timeout_is_float(self):
        from rlive_env.config import config as cfg
        self.assertIsInstance(cfg.WORLD_INTERFACE_TIMEOUT, float)
        self.assertGreater(cfg.WORLD_INTERFACE_TIMEOUT, 0)

    def test_world_interface_max_retries_is_int(self):
        from rlive_env.config import config as cfg
        self.assertIsInstance(cfg.WORLD_INTERFACE_MAX_RETRIES, int)
        self.assertGreaterEqual(cfg.WORLD_INTERFACE_MAX_RETRIES, 0)

    def test_world_interface_max_retries_time_is_float(self):
        from rlive_env.config import config as cfg
        self.assertIsInstance(cfg.WORLD_INTERFACE_MAX_RETRIES_TIME, float)
        self.assertGreater(cfg.WORLD_INTERFACE_MAX_RETRIES_TIME, 0)

    def test_world_interface_backoff_factor_is_float(self):
        from rlive_env.config import config as cfg
        self.assertIsInstance(cfg.WORLD_INTERFACE_BACKOFF_FACTOR, float)
        self.assertGreater(cfg.WORLD_INTERFACE_BACKOFF_FACTOR, 0)
