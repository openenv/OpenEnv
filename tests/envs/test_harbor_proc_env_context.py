"""Concurrent rollouts of a credential-by-env harness must see DIFFERENT keys from `os.environ`.

claude-code, gemini-cli and goose read `os.environ` inside `run()` to build the env dict they pass to
the sandbox, and the API key IS the rollout's session id — so N concurrent rollouts need N different
values of one variable at one instant. That is what forced `_PROC_ENV_LOCK` and made those three
harnesses serialise.

These tests assert the property that replaces the lock: an overlay is visible to the task that set it
and invisible to every other, including while they interleave.
"""

from __future__ import annotations

import asyncio
import os

import pytest
from openenv.harbor import proc_env_context as ctx


@pytest.fixture(autouse=True)
def _restore():
    """Uninstall the proxy between tests: it replaces a process-global."""
    real = os.environ
    ctx._installed = False
    yield
    os.environ = real
    ctx._installed = False


def test_the_overlay_is_visible_to_reads():
    ctx.install()
    with ctx.overlay({"OPENAI_API_KEY": "session-abc"}):
        assert os.environ.get("OPENAI_API_KEY") == "session-abc"
        assert os.environ["OPENAI_API_KEY"] == "session-abc"
        assert "OPENAI_API_KEY" in os.environ


def test_it_does_not_leak_after_the_block():
    ctx.install()
    with ctx.overlay({"OPENENV_TEST_ONLY": "x"}):
        pass
    assert os.environ.get("OPENENV_TEST_ONLY") is None


def test_concurrent_tasks_see_their_own_key():
    """The property the lock used to provide, now without serialising."""
    ctx.install()
    observed: dict[str, str | None] = {}

    async def rollout(name: str, key: str):
        with ctx.overlay({"OPENAI_API_KEY": key}):
            # Yield control repeatedly so the tasks genuinely interleave inside their overlays;
            # a process-global would be clobbered by whichever task ran last.
            for _ in range(5):
                await asyncio.sleep(0)
                assert os.environ.get("OPENAI_API_KEY") == key
            observed[name] = os.environ.get("OPENAI_API_KEY")

    async def main():
        await asyncio.gather(*(rollout(f"r{i}", f"session-{i}") for i in range(8)))

    asyncio.run(main())
    assert observed == {f"r{i}": f"session-{i}" for i in range(8)}


def test_copy_is_merged_because_subprocess_uses_it():
    """`subprocess` builds a child's env from `os.environ`; hiding the overlay would launch it
    without credentials, and that failure would look like a bad key rather than a bad proxy."""
    ctx.install()
    with ctx.overlay({"OPENENV_OVERLAY_ONLY": "yes"}):
        assert os.environ.copy().get("OPENENV_OVERLAY_ONLY") == "yes"
        assert "OPENENV_OVERLAY_ONLY" in dict(os.environ)
        assert "OPENENV_OVERLAY_ONLY" in list(os.environ)


def test_the_real_environment_still_shows_through():
    ctx.install()
    os.environ["OPENENV_REAL"] = "base"
    try:
        with ctx.overlay({"OTHER": "1"}):
            assert os.environ.get("OPENENV_REAL") == "base"
    finally:
        del os.environ["OPENENV_REAL"]


def test_writes_reach_the_real_environment():
    """Only reads are context-local; a write that vanished would break unrelated libraries."""
    ctx.install()
    with ctx.overlay({"A": "1"}):
        os.environ["OPENENV_WRITTEN"] = "persisted"
    assert os.environ.get("OPENENV_WRITTEN") == "persisted"
    del os.environ["OPENENV_WRITTEN"]


def test_it_can_be_switched_off():
    """This swaps a global the whole process reads, so it must be disableable without a rollback."""
    os.environ["OPENENV_CONCURRENT_PROC_ENV"] = "0"
    try:
        assert ctx.enabled() is False
        assert ctx.install() is False
    finally:
        del os.environ["OPENENV_CONCURRENT_PROC_ENV"]


def test_an_overlay_nests():
    """A rollout inside a rollout's context must not lose the outer values."""
    ctx.install()
    with ctx.overlay({"OUTER": "1"}):
        with ctx.overlay({"INNER": "2"}):
            assert os.environ.get("OUTER") == "1"
            assert os.environ.get("INNER") == "2"
        assert os.environ.get("INNER") is None
