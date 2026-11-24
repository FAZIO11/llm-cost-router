"""Pytest configuration and shared fixtures."""

import pytest
import os


@pytest.fixture(autouse=True)
def test_env():
    """Set test environment variables."""
    os.environ["ROUTER_MODE"] = "mock"
    os.environ["DATABASE_PATH"] = ":memory:"
    os.environ["LOG_LEVEL"] = "ERROR"
