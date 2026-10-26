# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for UVProvider readiness handling.

These tests cover the health-poll loop and the ``context_timeout_s`` knob
without launching a real ``uv`` subprocess or hitting the network.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import requests
from openenv.core.containers.runtime import uv_provider
from openenv.core.containers.runtime.uv_provider import _poll_health, UVProvider


@pytest.fixture()
def provider():
    """A UVProvider that skips the real `uv --version` check."""
    with patch.object(uv_provider, "_check_uv_installed"):
        return UVProvider(project_path="/tmp/some-env", context_timeout_s=300.0)


class TestGitProjectPath:
    """`project_path="git+<url>"` (as constructed by ``EnvClient.from_env``)
    must be cloned locally before being handed to ``uv run --project``, which
    only discovers projects in local directories.
    """

    def test_git_url_is_not_abspath_mangled(self):
        """`os.path.abspath` on a git URL collapses `https://` to `https:/`
        and prefixes cwd -- it must be left untouched until `start()` clones it.
        """
        with patch.object(uv_provider, "_check_uv_installed"):
            provider = UVProvider(
                project_path="git+https://huggingface.co/spaces/org/env"
            )

        assert provider.project_path == "git+https://huggingface.co/spaces/org/env"

    def test_start_clones_git_url_and_uses_local_path(self, tmp_path):
        """`start()` must resolve the git spec to a local clone directory and
        pass *that* to `uv run --project`, not the raw `git+...` string.
        """
        with patch.object(uv_provider, "_check_uv_installed"):
            provider = UVProvider(
                project_path="git+https://huggingface.co/spaces/org/env"
            )

        clone_dir = str(tmp_path / "clone")
        with (
            patch.object(
                uv_provider, "_clone_git_project", return_value=clone_dir
            ) as mock_clone,
            patch.object(uv_provider.subprocess, "Popen") as mock_popen,
        ):
            mock_popen.return_value = MagicMock(poll=lambda: None)
            provider.start(port=12345)

        mock_clone.assert_called_once_with(
            "git+https://huggingface.co/spaces/org/env", provider.context_timeout_s
        )
        command = mock_popen.call_args.args[0]
        assert clone_dir in command
        assert "git+https://huggingface.co/spaces/org/env" not in command

    def test_clone_uses_timeout(self, tmp_path):
        """`_clone_git_project` is passed a `timeout_s` so a hung/slow remote
        can't block `start()` (and the caller's readiness wait) forever.
        """
        with (
            patch.object(uv_provider.tempfile, "mkdtemp", return_value=str(tmp_path)),
            patch.object(uv_provider.subprocess, "run") as mock_run,
        ):
            mock_run.return_value = MagicMock()
            uv_provider._clone_git_project("git+https://example.com/x", 42.0)

        assert mock_run.call_args.kwargs["timeout"] == 42.0

    def test_clone_timeout_raises_and_cleans_up(self, tmp_path):
        """A `git clone` that exceeds `timeout_s` raises and removes the
        partial clone directory rather than leaving it on disk.
        """
        with (
            patch.object(uv_provider.tempfile, "mkdtemp", return_value=str(tmp_path)),
            patch.object(
                uv_provider.subprocess,
                "run",
                side_effect=uv_provider.subprocess.TimeoutExpired(
                    cmd="git", timeout=1.0
                ),
            ),
            patch.object(uv_provider.shutil, "rmtree") as mock_rmtree,
        ):
            with pytest.raises(RuntimeError, match="Timed out cloning"):
                uv_provider._clone_git_project("git+https://example.com/x", 1.0)

        mock_rmtree.assert_called_once_with(str(tmp_path), ignore_errors=True)

    def test_stop_cleans_up_clone_dir(self, tmp_path):
        """A temp clone dir from `start()` must not leak after `stop()`."""
        clone_dir = tmp_path / "clone"
        clone_dir.mkdir()

        with patch.object(uv_provider, "_check_uv_installed"):
            provider = UVProvider(project_path="git+https://example.com/x")
        provider._clone_dir = str(clone_dir)

        provider.stop()

        assert not clone_dir.exists()
        assert provider._clone_dir is None

    def test_restart_after_process_death_cleans_up_stale_clone_dir(self, tmp_path):
        """If the spawned process died on its own (poll() is not None), the
        "already running" guard doesn't block a second `start()` call. That
        second call must not silently leak the first clone directory when it
        clones again.
        """
        stale_clone_dir = tmp_path / "stale-clone"
        stale_clone_dir.mkdir()

        with patch.object(uv_provider, "_check_uv_installed"):
            provider = UVProvider(project_path="git+https://example.com/x")
        provider._process = MagicMock(poll=lambda: 0)  # exited on its own
        provider._clone_dir = str(stale_clone_dir)

        new_clone_dir = tmp_path / "new-clone"
        with (
            patch.object(
                uv_provider, "_clone_git_project", return_value=str(new_clone_dir)
            ),
            patch.object(uv_provider.subprocess, "Popen") as mock_popen,
        ):
            mock_popen.return_value = MagicMock(poll=lambda: None)
            provider.start(port=12346)

        assert not stale_clone_dir.exists()
        assert provider._clone_dir == str(new_clone_dir)

    def test_popen_failure_after_clone_cleans_up_clone_dir(self, tmp_path):
        """If `uv run` itself fails to launch right after a successful clone,
        the just-created clone directory must not leak.
        """
        with patch.object(uv_provider, "_check_uv_installed"):
            provider = UVProvider(project_path="git+https://example.com/x")

        clone_dir = tmp_path / "clone"
        clone_dir.mkdir()
        with (
            patch.object(
                uv_provider, "_clone_git_project", return_value=str(clone_dir)
            ),
            patch.object(uv_provider.subprocess, "Popen", side_effect=OSError("no uv")),
        ):
            with pytest.raises(RuntimeError, match="Failed to launch"):
                provider.start(port=12347)

        assert not clone_dir.exists()
        assert provider._clone_dir is None

    def test_local_path_is_still_abspath_resolved(self):
        """Plain local paths keep the existing `os.path.abspath` behavior."""
        with patch.object(uv_provider, "_check_uv_installed"):
            provider = UVProvider(project_path="relative/env")

        assert provider.project_path == uv_provider.os.path.abspath("relative/env")


class TestPollHealth:
    """`_poll_health` must back off between attempts, not busy-spin."""

    def test_sleeps_between_refused_connections(self):
        """A refused connection returns instantly; without a pause between
        attempts the loop would peg a CPU core for the whole timeout window.
        The loop must therefore sleep on the connection-error path too.
        """
        with (
            patch("requests.get", side_effect=requests.ConnectionError("refused")),
            patch.object(uv_provider.time, "sleep") as mock_sleep,
        ):
            with pytest.raises(TimeoutError, match="did not become ready"):
                _poll_health("http://localhost:9/health", timeout_s=0.05)

        assert mock_sleep.called, (
            "health poll busy-spun on connection errors without sleeping"
        )

    def test_returns_when_healthy(self):
        """Returns as soon as /health responds 200."""
        ok = MagicMock(status_code=200)
        with (
            patch("requests.get", return_value=ok) as mock_get,
            patch.object(uv_provider.time, "sleep"),
        ):
            _poll_health("http://localhost:9/health", timeout_s=5.0)

        mock_get.assert_called()


class TestWaitForReadyTimeout:
    """`wait_for_ready()` must honor the configured `context_timeout_s`."""

    def test_defaults_to_context_timeout(self, provider):
        """Called with no argument (as EnvClient does), the readiness wait uses
        the provider's `context_timeout_s` rather than a hardcoded default.
        """
        provider._base_url = "http://localhost:8000"
        provider._process = None

        with patch.object(uv_provider, "_poll_health") as mock_poll:
            provider.wait_for_ready()

        mock_poll.assert_called_once()
        assert mock_poll.call_args.kwargs["timeout_s"] == 300.0

    def test_explicit_timeout_overrides(self, provider):
        """An explicit timeout_s still wins over context_timeout_s."""
        provider._base_url = "http://localhost:8000"
        provider._process = None

        with patch.object(uv_provider, "_poll_health") as mock_poll:
            provider.wait_for_ready(timeout_s=5.0)

        assert mock_poll.call_args.kwargs["timeout_s"] == 5.0
