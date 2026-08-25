"""
Tests for CLI interface.
"""

import pytest
from unittest.mock import patch
import sys

from physcons.cli import main, demo_command


def test_demo_command_runs_without_error():
    """Demo command should execute without raising exceptions."""
    # This test just verifies the demo can run
    # It doesn't verify output since we're not executing it for real
    try:
        demo_command()
    except Exception as e:
        pytest.fail(f"Demo command raised unexpected exception: {e}")


def test_main_with_demo_command():
    """Main should handle demo command."""
    with patch.object(sys, 'argv', ['physcons', 'demo']):
        try:
            main()
        except SystemExit:
            # SystemExit is OK (clean exit)
            pass
        except Exception as e:
            pytest.fail(f"Main with demo command raised unexpected exception: {e}")


def test_main_with_no_command_shows_help():
    """Main with no command should show help and exit."""
    with patch.object(sys, 'argv', ['physcons']):
        with pytest.raises(SystemExit) as exc_info:
            main()
        
        # Should exit with code 1 (error)
        assert exc_info.value.code == 1


def test_main_with_invalid_command_shows_help():
    """Main with invalid command should show help and exit."""
    with patch.object(sys, 'argv', ['physcons', 'invalid']):
        with pytest.raises(SystemExit):
            main()
