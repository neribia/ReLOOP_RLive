"""Tests for path_utils module."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


class TestFindProjectRoot(unittest.TestCase):
    """Test the find_project_root function."""

    def test_find_project_root_with_caller_file(self):
        """Test that find_project_root works with a caller_file parameter."""
        from rlive_common.utils.path_utils import find_project_root

        # Use this test file as caller
        result = find_project_root(__file__)

        self.assertIsInstance(result, Path)
        self.assertTrue(result.exists())

    def test_find_project_root_without_caller_file(self):
        """Test that find_project_root works without parameters."""
        from rlive_common.utils.path_utils import find_project_root

        result = find_project_root()

        self.assertIsInstance(result, Path)
        self.assertTrue(result.exists())

    @patch('sys._MEIPASS', '/fake/meipass', create=True)
    def test_find_project_root_pyinstaller(self):
        """Test PyInstaller bundle detection."""
        from rlive_common.utils.path_utils import find_project_root

        result = find_project_root()

        # Should return sys._MEIPASS when it exists
        self.assertEqual(result, Path('/fake/meipass'))

    def test_find_project_root_finds_package_root(self):
        """Test that it finds the correct package root."""
        from rlive_common.utils.path_utils import find_project_root

        # Mock a config file path
        fake_config = Path(__file__).parent.parent.parent / "packages" / "rlive-common" / "src" / "rlive_common" / "config" / "config.py"

        if fake_config.exists():
            result = find_project_root(str(fake_config))

            # Should find packages/rlive-common
            self.assertTrue((result / "src" / "rlive_common").exists())
            self.assertTrue((result / "pyproject.toml").exists())


if __name__ == '__main__':
    unittest.main()
