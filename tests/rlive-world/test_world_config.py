import unittest
from pathlib import Path


class TestWorldConfig(unittest.TestCase):
    def test_project_root_exists(self):
        from rlive_world.config import config as cfg
        self.assertIsNotNone(cfg.PROJECT_ROOT)
        self.assertIsInstance(cfg.PROJECT_ROOT, Path)
        self.assertTrue(cfg.PROJECT_ROOT.exists())

    def test_project_root_contains_pyproject_toml(self):
        from rlive_world.config import config as cfg
        # The package root should contain pyproject.toml
        self.assertTrue((cfg.PROJECT_ROOT / "pyproject.toml").exists())

    def test_project_root_is_package_root(self):
        from rlive_world.config import config as cfg
        # PROJECT_ROOT should point to the package root (packages/rlive-world)
        # which contains src/rlive_world
        self.assertTrue((cfg.PROJECT_ROOT / "src" / "rlive_world").exists())
        # Verify it's not the workspace root
        self.assertFalse((cfg.PROJECT_ROOT / "packages").exists())

    def test_world_host_is_string(self):
        from rlive_world.config import config as cfg
        self.assertIsInstance(cfg.WORLD_HOST, str)

    def test_world_port_is_int(self):
        from rlive_world.config import config as cfg
        self.assertIsInstance(cfg.WORLD_PORT, int)
        self.assertGreater(cfg.WORLD_PORT, 0)

    def test_spheroboltplus_name_is_string(self):
        from rlive_world.config import config as cfg
        self.assertIsInstance(cfg.SPHEROBOLTPLUS_NAME, str)

    def test_spheroboltplus_speed_is_int(self):
        from rlive_world.config import config as cfg
        self.assertIsInstance(cfg.SPHEROBOLTPLUS_SPEED, int)

    def test_spheroboltplus_duration_is_float(self):
        from rlive_world.config import config as cfg
        self.assertIsInstance(cfg.SPHEROBOLTPLUS_DURATION, float)
        self.assertGreater(cfg.SPHEROBOLTPLUS_DURATION, 0)
