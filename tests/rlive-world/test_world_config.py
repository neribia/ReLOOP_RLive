import unittest


class TestWorldConfig(unittest.TestCase):
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
