"""Shared test isolation for process-global plotting state."""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def close_matplotlib_figures_after_test():
    """Prevent plots created by one test leaking into later tests.

    Several public analysis APIs intentionally return Matplotlib figures so
    applications can further customize them. Tests that exercise these APIs
    must not leave pyplot's process-global figure registry populated for the
    next test.
    """
    yield
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    plt.close("all")
