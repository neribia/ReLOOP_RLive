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


if __name__ == "__main__":
    unittest.main()
