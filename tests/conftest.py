"""Pytest configuration for genMoPlan tests."""

import os
import pytest
import shutil

# Use project-local tmp directory instead of system /tmp
PROJECT_TMP = "/common/home/st1122/Projects/genMoPlan/tmp"


@pytest.fixture
def test_tmp_dir():
    """Create a temporary directory for tests within the project."""
    import uuid
    test_dir = os.path.join(PROJECT_TMP, f"test_{uuid.uuid4().hex[:8]}")
    os.makedirs(test_dir, exist_ok=True)
    yield test_dir
    # Cleanup after test
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)


def pytest_configure(config):
    """Configure pytest to use project-local tmp directory."""
    os.makedirs(PROJECT_TMP, exist_ok=True)
    os.environ["TMPDIR"] = PROJECT_TMP
    os.environ["TMP"] = PROJECT_TMP
    os.environ["TEMP"] = PROJECT_TMP
