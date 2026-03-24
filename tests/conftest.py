import pytest

def pytest_configure(config):
    config.addinivalue_line("markers", "requires_910b: requires 910B hardware")
