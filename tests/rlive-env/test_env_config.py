import unittest


class TestEnvConfig(unittest.TestCase):
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
