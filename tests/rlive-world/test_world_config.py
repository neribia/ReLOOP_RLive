import unittest
from pathlib import Path

from rlive_world.config import config as cfg


class TestWorldConfig(unittest.TestCase):
    def test_project_root_exists(self):
        self.assertIsNotNone(cfg.PROJECT_ROOT)
        self.assertIsInstance(cfg.PROJECT_ROOT, Path)
        self.assertTrue(cfg.PROJECT_ROOT.exists())

    def test_project_root_contains_pyproject_toml(self):
        # The package root should contain pyproject.toml
        self.assertTrue((cfg.PROJECT_ROOT / "pyproject.toml").exists())

    def test_project_root_is_package_root(self):
        # PROJECT_ROOT should point to the package root (packages/rlive-world)
        # which contains src/rlive_world
        self.assertTrue((cfg.PROJECT_ROOT / "src" / "rlive_world").exists())
        # Verify it's not the workspace root
        self.assertFalse((cfg.PROJECT_ROOT / "packages").exists())

    def test_world_host_is_string(self):
        self.assertIsInstance(cfg.WORLD_HOST, str)

    def test_world_port_is_int(self):
        self.assertIsInstance(cfg.WORLD_PORT, int)
        self.assertGreater(cfg.WORLD_PORT, 0)

    def test_spheroboltplus_name_is_string(self):
        self.assertIsInstance(cfg.SPHEROBOLTPLUS_NAME, str)

    def test_spheroboltplus_speed_is_int(self):
        self.assertIsInstance(cfg.SPHEROBOLTPLUS_SPEED, int)

    def test_spheroboltplus_duration_is_float(self):
        self.assertIsInstance(cfg.SPHEROBOLTPLUS_DURATION, float)
        self.assertGreater(cfg.SPHEROBOLTPLUS_DURATION, 0)

    def test_spheroboltplus_display_color_r_is_int(self):
        self.assertIsInstance(cfg.SPHEROBOLTPLUS_DISPLAY_COLOR_R, int)
        self.assertGreaterEqual(cfg.SPHEROBOLTPLUS_DISPLAY_COLOR_R, 0)
        self.assertLessEqual(cfg.SPHEROBOLTPLUS_DISPLAY_COLOR_R, 255)

    def test_spheroboltplus_display_color_g_is_int(self):
        self.assertIsInstance(cfg.SPHEROBOLTPLUS_DISPLAY_COLOR_G, int)
        self.assertGreaterEqual(cfg.SPHEROBOLTPLUS_DISPLAY_COLOR_G, 0)
        self.assertLessEqual(cfg.SPHEROBOLTPLUS_DISPLAY_COLOR_G, 255)

    def test_spheroboltplus_display_color_b_is_int(self):
        self.assertIsInstance(cfg.SPHEROBOLTPLUS_DISPLAY_COLOR_B, int)
        self.assertGreaterEqual(cfg.SPHEROBOLTPLUS_DISPLAY_COLOR_B, 0)
        self.assertLessEqual(cfg.SPHEROBOLTPLUS_DISPLAY_COLOR_B, 255)

