"""Shared fixtures.

The API client lives here rather than in one test module so that every
integration file uses the same session and the same skip behaviour. Tests that
need a running stack must *skip* when it is absent, not fail: a developer
running unit tests on a laptop with no Docker should see a clean run, and a CI
job that forgot to start the stack should see skips it can notice rather than
failures it will learn to ignore.
"""

from __future__ import annotations

import os

import httpx
import pytest

API_BASE = os.environ.get("BRAIN_API", "http://localhost:8000")


@pytest.fixture(scope="session")
def client():
    """An HTTP client against the running API, or a skip."""
    with httpx.Client(base_url=API_BASE, timeout=60) as session:
        try:
            session.get("/health/live").raise_for_status()
        except Exception:
            pytest.skip(f"API not reachable at {API_BASE}; start the stack first")
        yield session
