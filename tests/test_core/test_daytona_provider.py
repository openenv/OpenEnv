# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for DaytonaProvider. All tests mock the daytona SDK."""

from __future__ import annotations

import os
import pathlib
import sys
import types
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Fake daytona SDK module (so tests run without ``pip install daytona``)
# ---------------------------------------------------------------------------
def _install_fake_daytona():
    """Install a minimal fake ``daytona`` package into sys.modules."""
    daytona_mod = types.ModuleType("daytona")

    class _FakeConfig:
        def __init__(self, **kwargs):
            self.api_key = kwargs.get("api_key")
            self.target = kwargs.get("target")

    class _FakeDaytona:
        def __init__(self, config=None):
            self.config = config
            self._created = []

        def create(self, params=None, **kwargs):
            sandbox = MagicMock()

            signed_preview = MagicMock()
            signed_preview.url = "https://8000-signed-tok.proxy.daytona.works"
            signed_preview.token = "signed-tok"
            sandbox.create_signed_preview_url = MagicMock(return_value=signed_preview)

            # Default process.exec responds to openenv.yaml discovery calls
            def _default_exec(cmd, **kw):
                if "test -f /app/env/openenv.yaml" in cmd:
                    return "found"
                if cmd.startswith("cat /app/env/openenv.yaml"):
                    return (
                        "spec_version: 1\nname: test\napp: server.app:app\nport: 8000\n"
                    )
                return ""

            sandbox.process.exec = MagicMock(side_effect=_default_exec)
            self._created.append((params, kwargs))
            return sandbox

        def delete(self, sandbox, **kwargs):
            pass

    class _CreateFromImage:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    class _CreateFromSnapshot:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    class _Resources:
        def __init__(self, cpu=None, memory=None):
            self.cpu = cpu
            self.memory = memory

    class _FakeImage:
        """Minimal fake of daytona.Image."""

        def __init__(self, dockerfile_path=None, _content=None):
            self.dockerfile_path = dockerfile_path
            self._content = _content  # captured for test assertions

        @staticmethod
        def from_dockerfile(path):
            # Capture file content at call time (before cleanup)
            content = pathlib.Path(path).read_text()
            return _FakeImage(dockerfile_path=str(path), _content=content)

    daytona_mod.Daytona = _FakeDaytona
    daytona_mod.DaytonaConfig = _FakeConfig
    daytona_mod.CreateSandboxFromImageParams = _CreateFromImage
    daytona_mod.CreateSandboxFromSnapshotParams = _CreateFromSnapshot
    daytona_mod.Resources = _Resources
    daytona_mod.Image = _FakeImage

    sys.modules["daytona"] = daytona_mod
    return daytona_mod


_fake_daytona = _install_fake_daytona()

# Now safe to import the provider
from openenv.core.containers.runtime.daytona_provider import DaytonaProvider


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture()
def provider():
    """Return a DaytonaProvider with default settings."""
    return DaytonaProvider(api_key="test-key")


@pytest.fixture()
def public_provider():
    """Return a DaytonaProvider with public=True."""
    return DaytonaProvider(api_key="test-key", public=True)


@pytest.fixture(autouse=True)
def _fast_provider_sleep():
    """Avoid real sleeps in DaytonaProvider (start_container and wait_for_ready)."""
    with patch("openenv.core.containers.runtime.daytona_provider.time.sleep"):
        yield


@pytest.fixture(autouse=True)
def _clean_dockerfile_registry():
    """Clear the Dockerfile registry between tests."""
    DaytonaProvider._dockerfile_registry.clear()
    yield
    DaytonaProvider._dockerfile_registry.clear()


def _assert_exec_called_with_fragment(sandbox, expected_fragment: str) -> str:
    """Assert sandbox.process.exec was called with a command containing a fragment."""
    commands = [
        call.args[0] for call in sandbox.process.exec.call_args_list if call.args
    ]
    assert any(expected_fragment in cmd for cmd in commands), (
        f"Expected process.exec command containing: {expected_fragment!r}\n"
        f"Observed commands: {commands}"
    )
    return next(cmd for cmd in commands if expected_fragment in cmd)


# ---------------------------------------------------------------------------
# Tests: start_container — image string parsing
# ---------------------------------------------------------------------------
class TestStartContainer:
    def test_registry_image(self, provider):
        """A normal image string uses CreateSandboxFromImageParams."""
        url = provider.start_container("echo-env:latest")
        assert url.startswith("https://")
        params, _ = provider._daytona._created[0]
        assert isinstance(params, _fake_daytona.CreateSandboxFromImageParams)
        assert params.image == "echo-env:latest"

    def test_snapshot_prefix(self, provider):
        """An image starting with 'snapshot:' uses CreateSandboxFromSnapshotParams."""
        url = provider.start_container("snapshot:my-snap")
        assert url.startswith("https://")
        params, _ = provider._daytona._created[0]
        assert isinstance(params, _fake_daytona.CreateSandboxFromSnapshotParams)
        assert params.snapshot == "my-snap"

    def test_create_signed_preview_url_called(self, provider):
        """start_container calls sandbox.create_signed_preview_url(8000, ...)."""
        provider.start_container("echo-env:latest")
        provider._sandbox.create_signed_preview_url.assert_called_once_with(
            8000, expires_in_seconds=86400
        )

    def test_returns_signed_preview_url(self, provider):
        """start_container returns the signed preview URL."""
        url = provider.start_container("echo-env:latest")
        assert url == "https://8000-signed-tok.proxy.daytona.works"

    def test_constructor_image_used_when_start_omits_image(self):
        """A provider-owned image lets the outer client call start_container()."""
        provider = DaytonaProvider(
            api_key="test-key",
            image="constructor-env:latest",
            env_vars={"DEBUG": "1"},
        )
        url = provider.start_container()
        assert url.startswith("https://")
        params, _ = provider._daytona._created[0]
        assert isinstance(params, _fake_daytona.CreateSandboxFromImageParams)
        assert params.image == "constructor-env:latest"
        assert params.env_vars == {"DEBUG": "1"}

    def test_start_image_overrides_constructor_image(self):
        """Explicit start_container image takes precedence."""
        provider = DaytonaProvider(api_key="test-key", image="constructor-env:latest")
        provider.start_container("explicit-env:latest")
        params, _ = provider._daytona._created[0]
        assert params.image == "explicit-env:latest"

    def test_requires_image_from_constructor_or_start(self, provider):
        with pytest.raises(ValueError, match="requires an image"):
            provider.start_container()


# ---------------------------------------------------------------------------
# Tests: port validation
# ---------------------------------------------------------------------------
class TestPortValidation:
    def test_port_none_accepted(self, provider):
        """port=None is fine (default)."""
        url = provider.start_container("echo-env:latest", port=None)
        assert url is not None

    def test_port_8000_accepted(self, provider):
        """port=8000 is explicitly accepted."""
        url = provider.start_container("echo-env:latest", port=8000)
        assert url is not None

    def test_other_port_raises(self, provider):
        """Any port other than None/8000 raises ValueError."""
        with pytest.raises(ValueError, match="only supports port 8000"):
            provider.start_container("echo-env:latest", port=3000)


# ---------------------------------------------------------------------------
# Tests: env_vars forwarding
# ---------------------------------------------------------------------------
class TestEnvVars:
    def test_env_vars_passed_through(self, provider):
        """env_vars are forwarded to the SDK create params."""
        provider.start_container(
            "echo-env:latest", env_vars={"DEBUG": "1", "FOO": "bar"}
        )
        params, _ = provider._daytona._created[0]
        assert params.env_vars == {"DEBUG": "1", "FOO": "bar"}

    def test_no_env_vars(self, provider):
        """When env_vars is None, the params don't include env_vars."""
        provider.start_container("echo-env:latest")
        params, _ = provider._daytona._created[0]
        assert not hasattr(params, "env_vars")


# ---------------------------------------------------------------------------
# Tests: public flag forwarding
# ---------------------------------------------------------------------------
class TestPublicFlag:
    def test_public_true_forwarded(self, public_provider):
        """public=True is forwarded to the SDK create params."""
        public_provider.start_container("echo-env:latest")
        params, _ = public_provider._daytona._created[0]
        assert params.public is True

    def test_public_false_by_default(self, provider):
        """By default, public is not set on create params."""
        provider.start_container("echo-env:latest")
        params, _ = provider._daytona._created[0]
        assert not hasattr(params, "public")


# ---------------------------------------------------------------------------
# Tests: auto_stop_interval forwarding
# ---------------------------------------------------------------------------
class TestAutoStopInterval:
    def test_non_default_forwarded(self):
        """Non-default auto_stop_interval is forwarded to create params."""
        p = DaytonaProvider(api_key="k", auto_stop_interval=0)
        p.start_container("echo-env:latest")
        params, _ = p._daytona._created[0]
        assert params.auto_stop_interval == 0

    def test_default_not_set(self, provider):
        """Default auto_stop_interval (15) is omitted from create params."""
        provider.start_container("echo-env:latest")
        params, _ = provider._daytona._created[0]
        assert not hasattr(params, "auto_stop_interval")


# ---------------------------------------------------------------------------
# Tests: stop_container
# ---------------------------------------------------------------------------
class TestStopContainer:
    def test_delete_called(self, provider):
        """stop_container calls daytona.delete(sandbox)."""
        provider.start_container("echo-env:latest")
        sandbox = provider._sandbox
        provider._daytona.delete = MagicMock()
        provider.stop_container()
        provider._daytona.delete.assert_called_once_with(sandbox)

    def test_stop_clears_state(self, provider):
        """After stop, internal state is cleared."""
        provider.start_container("echo-env:latest")
        provider.stop_container()
        assert provider._sandbox is None
        assert provider._preview_url is None

    def test_stop_noop_when_no_sandbox(self, provider):
        """stop_container is a no-op if no sandbox was started."""
        provider.stop_container()  # Should not raise


# ---------------------------------------------------------------------------
# Tests: refresh_preview_url
# ---------------------------------------------------------------------------
class TestRefreshPreviewUrl:
    def test_returns_new_signed_url(self, provider):
        """refresh_preview_url returns a fresh signed URL."""
        provider.start_container("echo-env:latest")
        # Reset the mock so we can distinguish the refresh call
        new_signed = MagicMock()
        new_signed.url = "https://8000-refreshed.proxy.daytona.works"
        provider._sandbox.create_signed_preview_url = MagicMock(return_value=new_signed)
        url = provider.refresh_preview_url()
        assert url == "https://8000-refreshed.proxy.daytona.works"
        provider._sandbox.create_signed_preview_url.assert_called_once_with(
            8000, expires_in_seconds=86400
        )

    def test_updates_internal_state(self, provider):
        """refresh_preview_url updates _preview_url."""
        provider.start_container("echo-env:latest")
        new_signed = MagicMock()
        new_signed.url = "https://8000-refreshed.proxy.daytona.works"
        provider._sandbox.create_signed_preview_url = MagicMock(return_value=new_signed)
        provider.refresh_preview_url()
        assert provider._preview_url == "https://8000-refreshed.proxy.daytona.works"

    def test_no_sandbox_raises(self, provider):
        """refresh_preview_url raises RuntimeError if no sandbox is active."""
        with pytest.raises(RuntimeError, match="No active sandbox"):
            provider.refresh_preview_url()


# ---------------------------------------------------------------------------
# Tests: wait_for_ready
# ---------------------------------------------------------------------------
class TestWaitForReady:
    def test_health_polling(self, provider):
        """wait_for_ready polls /health until 200."""
        provider.start_container("echo-env:latest")
        url = provider._preview_url

        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("requests.get", return_value=mock_response) as mock_get:
            provider.wait_for_ready(url)
            mock_get.assert_called()
            assert f"{url}/health" == mock_get.call_args.args[0]

    def test_timeout_raises(self, provider):
        """wait_for_ready raises TimeoutError if health never returns 200."""
        provider.start_container("echo-env:latest")
        url = provider._preview_url

        import requests

        with patch("requests.get", side_effect=requests.ConnectionError("nope")):
            with pytest.raises(TimeoutError, match="did not become ready"):
                provider.wait_for_ready(url, timeout_s=0.1)


# ---------------------------------------------------------------------------
# Tests: API key from env var
# ---------------------------------------------------------------------------
class TestApiKeyFromEnv:
    def test_fallback_to_env_var(self):
        """When no api_key is passed, falls back to DAYTONA_API_KEY."""
        with patch.dict(os.environ, {"DAYTONA_API_KEY": "env-key-123"}):
            p = DaytonaProvider()
            assert p._daytona.config.api_key == "env-key-123"

    def test_explicit_key_overrides_env(self):
        """Explicit api_key takes precedence over env var."""
        with patch.dict(os.environ, {"DAYTONA_API_KEY": "env-key"}):
            p = DaytonaProvider(api_key="explicit-key")
            assert p._daytona.config.api_key == "explicit-key"


# ---------------------------------------------------------------------------
# Tests: resources forwarding
# ---------------------------------------------------------------------------
class TestResources:
    def test_resources_passed_to_image_params(self):
        """Resources are forwarded to CreateSandboxFromImageParams."""
        resources = _fake_daytona.Resources(cpu=4, memory=8)
        p = DaytonaProvider(api_key="k", resources=resources)
        p.start_container("echo-env:latest")
        params, _ = p._daytona._created[0]
        assert params.resources is resources

    def test_resources_not_set_for_snapshot(self):
        """Snapshot params don't receive resources (not supported)."""
        resources = _fake_daytona.Resources(cpu=4, memory=8)
        p = DaytonaProvider(api_key="k", resources=resources)
        p.start_container("snapshot:my-snap")
        params, _ = p._daytona._created[0]
        assert not hasattr(params, "resources")


# ---------------------------------------------------------------------------
# Tests: on_snapshot_create_logs callback
# ---------------------------------------------------------------------------
class TestSnapshotCreateLogs:
    def test_callback_forwarded(self):
        """on_snapshot_create_logs is forwarded to daytona.create()."""
        log_fn = MagicMock()
        p = DaytonaProvider(api_key="k", on_snapshot_create_logs=log_fn)
        p.start_container("echo-env:latest")
        _, create_kwargs = p._daytona._created[0]
        assert create_kwargs["on_snapshot_create_logs"] is log_fn


# ---------------------------------------------------------------------------
# Tests: _discover_server_cmd
# ---------------------------------------------------------------------------
class TestDiscoverServerCmd:
    def test_modern_layout_discovered(self):
        """openenv.yaml found at /app/env/ on the fast path."""
        p = DaytonaProvider(api_key="k")
        p.start_container("echo-env:latest")
        _assert_exec_called_with_fragment(
            p._sandbox, "cd /app/env && python -m uvicorn server.app:app"
        )

    def test_fallback_find(self):
        """Fast path misses, find locates openenv.yaml in old layout."""
        p = DaytonaProvider(api_key="k")

        def _exec(cmd, **kw):
            if "test -f /app/env/openenv.yaml" in cmd:
                return ""  # fast path miss
            if cmd.startswith("find /app"):
                return "/app/envs/atari_env/openenv.yaml\n"
            if cmd.startswith("cat /app/envs/atari_env/openenv.yaml"):
                return "spec_version: 1\napp: server.app:app\n"
            return ""

        # Patch the fake create to use our custom exec
        original_create = p._daytona.create

        def patched_create(params=None, **kwargs):
            sandbox = original_create(params, **kwargs)
            sandbox.process.exec = MagicMock(side_effect=_exec)
            return sandbox

        p._daytona.create = patched_create
        p.start_container("some-image:latest")
        _assert_exec_called_with_fragment(
            p._sandbox, "cd /app/envs/atari_env && python -m uvicorn server.app:app"
        )

    def test_no_yaml_raises(self):
        """No openenv.yaml anywhere raises ValueError."""
        p = DaytonaProvider(api_key="k")

        def _exec(cmd, **kw):
            return ""

        original_create = p._daytona.create

        def patched_create(params=None, **kwargs):
            sandbox = original_create(params, **kwargs)
            sandbox.process.exec = MagicMock(side_effect=_exec)
            return sandbox

        p._daytona.create = patched_create
        with pytest.raises(ValueError, match="Could not find openenv.yaml"):
            p.start_container("no-yaml-image:latest")

    def test_yaml_without_app_field_raises(self):
        """openenv.yaml found but no app key raises ValueError."""
        p = DaytonaProvider(api_key="k")

        def _exec(cmd, **kw):
            if "test -f /app/env/openenv.yaml" in cmd:
                return "found"
            if cmd.startswith("cat /app/env/openenv.yaml"):
                return "spec_version: 1\nname: test\nport: 8000\n"
            return ""

        original_create = p._daytona.create

        def patched_create(params=None, **kwargs):
            sandbox = original_create(params, **kwargs)
            sandbox.process.exec = MagicMock(side_effect=_exec)
            return sandbox

        p._daytona.create = patched_create
        with pytest.raises(ValueError, match="does not contain an 'app' field"):
            p.start_container("bad-yaml-image:latest")


# ---------------------------------------------------------------------------
# Tests: _parse_app_field
# ---------------------------------------------------------------------------
class TestParseAppField:
    def test_standard_format(self):
        content = "spec_version: 1\napp: server.app:app\nport: 8000\n"
        assert DaytonaProvider._parse_app_field(content) == "server.app:app"

    def test_double_quoted_value(self):
        content = 'app: "server.app:app"\n'
        assert DaytonaProvider._parse_app_field(content) == "server.app:app"

    def test_single_quoted_value(self):
        content = "app: 'server.app:app'\n"
        assert DaytonaProvider._parse_app_field(content) == "server.app:app"

    def test_missing_field(self):
        content = "spec_version: 1\nname: test\nport: 8000\n"
        assert DaytonaProvider._parse_app_field(content) is None

    def test_empty_value(self):
        content = "app:\n"
        assert DaytonaProvider._parse_app_field(content) is None

    def test_inline_comment_stripped(self):
        content = "app: server.app:app  # the ASGI app\n"
        assert DaytonaProvider._parse_app_field(content) == "server.app:app"

    def test_inline_comment_only_returns_none(self):
        content = "app: # comment only\n"
        assert DaytonaProvider._parse_app_field(content) is None

    def test_quoted_value_with_inline_comment(self):
        content = 'app: "server.app:app"  # comment\n'
        assert DaytonaProvider._parse_app_field(content) == "server.app:app"

    def test_nested_app_key_ignored(self):
        content = "server:\n  app: nested\napp: top_level\n"
        assert DaytonaProvider._parse_app_field(content) == "top_level"

    def test_nested_app_key_only_returns_none(self):
        content = "server:\n  app: nested\n"
        assert DaytonaProvider._parse_app_field(content) is None


# ---------------------------------------------------------------------------
# Tests: _parse_dockerfile_cmd
# ---------------------------------------------------------------------------
class TestParseDockerfileCmd:
    def test_shell_form(self):
        content = "FROM python:3.11\nCMD uvicorn app:app\n"
        assert DaytonaProvider._parse_dockerfile_cmd(content) == "uvicorn app:app"

    def test_exec_form(self):
        content = 'FROM python:3.11\nCMD ["uvicorn", "app:app", "--port", "8000"]\n'
        assert (
            DaytonaProvider._parse_dockerfile_cmd(content)
            == "uvicorn app:app --port 8000"
        )

    def test_last_cmd_wins(self):
        content = "FROM python:3.11\nCMD first\nCMD second\n"
        assert DaytonaProvider._parse_dockerfile_cmd(content) == "second"

    def test_comment_ignored(self):
        content = "FROM python:3.11\n# CMD fake\nCMD real\n"
        assert DaytonaProvider._parse_dockerfile_cmd(content) == "real"

    def test_no_cmd_returns_none(self):
        content = "FROM python:3.11\nRUN pip install flask\n"
        assert DaytonaProvider._parse_dockerfile_cmd(content) is None

    def test_case_insensitive(self):
        content = "FROM python:3.11\ncmd uvicorn app:app\n"
        assert DaytonaProvider._parse_dockerfile_cmd(content) == "uvicorn app:app"

    def test_exec_form_invalid_json(self):
        content = "FROM python:3.11\nCMD [not valid json\n"
        assert DaytonaProvider._parse_dockerfile_cmd(content) == "[not valid json"

    def test_empty_cmd(self):
        content = "FROM python:3.11\nCMD \n"
        assert DaytonaProvider._parse_dockerfile_cmd(content) is None


# ---------------------------------------------------------------------------
# Tests: cmd / server start
# ---------------------------------------------------------------------------
class TestServerCmd:
    def test_explicit_cmd_used(self):
        """Constructor cmd is used and process.exec is called."""
        p = DaytonaProvider(api_key="k", cmd="python -m myserver")
        p.start_container("some-image:latest")
        _assert_exec_called_with_fragment(p._sandbox, "python -m myserver")

    def test_kwargs_cmd_overrides(self):
        """cmd passed via kwargs takes precedence."""
        p = DaytonaProvider(api_key="k", cmd="default-cmd")
        p.start_container("img:latest", cmd="override-cmd")
        _assert_exec_called_with_fragment(p._sandbox, "override-cmd")

    def test_auto_detected_cmd(self):
        """Without explicit cmd, discovery produces correct command."""
        p = DaytonaProvider(api_key="k")
        p.start_container("lovrepesut/openenv-connect4:latest")
        _assert_exec_called_with_fragment(
            p._sandbox, "cd /app/env && python -m uvicorn server.app:app"
        )


# ---------------------------------------------------------------------------
# Tests: strip_buildkit_syntax
# ---------------------------------------------------------------------------
class TestStripBuildkitSyntax:
    def test_strips_single_mount(self):
        """A single --mount=... flag is removed from a RUN line."""
        line = "RUN --mount=type=cache,target=/root/.cache/uv uv sync"
        result = DaytonaProvider.strip_buildkit_syntax(line)
        assert result == "RUN uv sync"

    def test_strips_multiple_mounts(self):
        """Multiple --mount flags on one RUN line are all removed."""
        line = "RUN --mount=type=cache,target=/a --mount=type=bind,src=/b pip install"
        result = DaytonaProvider.strip_buildkit_syntax(line)
        assert result == "RUN pip install"

    def test_preserves_run_without_mount(self):
        """A RUN line without --mount is returned unchanged."""
        line = "RUN apt-get update"
        assert DaytonaProvider.strip_buildkit_syntax(line) == line

    def test_preserves_non_run_lines(self):
        """Non-RUN lines (FROM, COPY, etc.) are untouched."""
        content = "FROM python:3.11\nCOPY . /app"
        assert DaytonaProvider.strip_buildkit_syntax(content) == content

    def test_multiline_mount_continuation(self):
        """--mount on a continuation line after RUN is stripped."""
        content = (
            "FROM python:3.12-slim\n"
            "RUN --mount=type=cache,target=/root/.cache/pip \\\n"
            "    pip install requests\n"
        )
        result = DaytonaProvider.strip_buildkit_syntax(content)
        assert "--mount=" not in result
        assert "pip install requests" in result

    def test_multi_mount_across_continuations(self):
        """Multiple --mount flags on separate continuation lines are all stripped."""
        content = (
            "FROM python:3.12-slim\n"
            "RUN --mount=type=cache,target=/root/.cache/pip \\\n"
            "    --mount=type=bind,source=req.txt,target=/tmp/req.txt \\\n"
            "    pip install -r /tmp/req.txt\n"
        )
        result = DaytonaProvider.strip_buildkit_syntax(content)
        assert "--mount=" not in result
        assert "pip install -r /tmp/req.txt" in result
        # The FROM line must survive
        assert "FROM python:3.12-slim" in result

    def test_real_echo_env_dockerfile(self):
        """Stripping the echo_env Dockerfile removes --mount but keeps everything else."""
        import pathlib

        dockerfile = (
            pathlib.Path(__file__).resolve().parents[2]
            / "envs/echo_env/server/Dockerfile"
        )
        if not dockerfile.exists():
            pytest.skip("echo_env Dockerfile not found")
        original = dockerfile.read_text()
        stripped = DaytonaProvider.strip_buildkit_syntax(original)
        assert "--mount=" not in stripped
        # All non-mount lines should still be present
        for line in original.splitlines():
            if "--mount=" not in line:
                assert line in stripped

    def test_empty_string(self):
        """Empty input returns empty output."""
        assert DaytonaProvider.strip_buildkit_syntax("") == ""


# ---------------------------------------------------------------------------
# Tests: image_from_dockerfile
# ---------------------------------------------------------------------------
class TestImageFromDockerfile:
    def test_returns_dockerfile_uri(self, tmp_path):
        """Returns a 'dockerfile:' prefixed string with absolute path."""
        df = tmp_path / "Dockerfile"
        df.write_text("FROM python:3.11\nRUN pip install flask\n")
        result = DaytonaProvider.image_from_dockerfile(str(df))
        assert isinstance(result, str)
        assert result.startswith("dockerfile:")
        assert result == f"dockerfile:{df.resolve()}"

    def test_buildkit_stripped_in_registry(self, tmp_path):
        """BuildKit --mount syntax is stripped in the stored registry entry."""
        df = tmp_path / "Dockerfile"
        df.write_text(
            "FROM python:3.11\nRUN --mount=type=cache,target=/x pip install flask\n"
        )
        result = DaytonaProvider.image_from_dockerfile(str(df))
        key = result[len("dockerfile:") :]
        stripped = DaytonaProvider._dockerfile_registry[key]["stripped_content"]
        assert "--mount=" not in stripped
        assert "pip install flask" in stripped

    def test_context_dir_same_as_parent(self, tmp_path):
        """Explicit context_dir pointing to Dockerfile's parent works."""
        df = tmp_path / "Dockerfile"
        df.write_text("FROM python:3.11\n")
        result = DaytonaProvider.image_from_dockerfile(
            str(df), context_dir=str(tmp_path)
        )
        assert result.startswith("dockerfile:")

    def test_context_dir_different(self, tmp_path):
        """Dockerfile in a subdirectory, context_dir is the parent."""
        server = tmp_path / "server"
        server.mkdir()
        df = server / "Dockerfile"
        df.write_text("FROM python:3.11\nCOPY . /app\n")
        result = DaytonaProvider.image_from_dockerfile(
            str(df), context_dir=str(tmp_path)
        )
        assert result.startswith("dockerfile:")

    def test_context_dir_stored_in_registry(self, tmp_path):
        """The resolved context_dir is stored in the registry."""
        server = tmp_path / "server"
        server.mkdir()
        df = server / "Dockerfile"
        df.write_text("FROM python:3.11\nCOPY . /app\n")
        result = DaytonaProvider.image_from_dockerfile(
            str(df), context_dir=str(tmp_path)
        )
        key = result[len("dockerfile:") :]
        assert DaytonaProvider._dockerfile_registry[key]["context_dir"] == str(tmp_path)

    def test_file_not_found(self, tmp_path):
        """Nonexistent Dockerfile raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            DaytonaProvider.image_from_dockerfile(str(tmp_path / "nope"))

    def test_context_dir_not_found(self, tmp_path):
        """Valid Dockerfile + nonexistent context_dir raises ValueError."""
        df = tmp_path / "Dockerfile"
        df.write_text("FROM python:3.11\n")
        with pytest.raises(ValueError, match="context_dir"):
            DaytonaProvider.image_from_dockerfile(str(df), context_dir="/no/such/dir")

    def test_no_temp_files_created(self, tmp_path):
        """image_from_dockerfile does not create temp files (Image is built later)."""
        df = tmp_path / "Dockerfile"
        df.write_text(
            "FROM python:3.11\nRUN --mount=type=cache,target=/x pip install\n"
        )
        DaytonaProvider.image_from_dockerfile(str(df), context_dir=str(tmp_path))
        leftover = list(tmp_path.glob("*.dockerfile"))
        assert leftover == [], f"Unexpected temp files: {leftover}"

    def test_copy_source_not_found_raises(self, tmp_path):
        """COPY source missing under context_dir raises ValueError."""
        server = tmp_path / "server"
        server.mkdir()
        df = server / "Dockerfile"
        df.write_text("FROM python:3.11\nCOPY nonexistent_dir /app\n")
        with pytest.raises(ValueError, match="COPY source.*not found"):
            DaytonaProvider.image_from_dockerfile(str(df), context_dir=str(tmp_path))

    def test_cmd_stored_in_registry(self, tmp_path):
        """Parsed CMD is stored as server_cmd in the registry."""
        df = tmp_path / "Dockerfile"
        df.write_text("FROM python:3.11\nCMD uvicorn app:app --port 8000\n")
        result = DaytonaProvider.image_from_dockerfile(str(df))
        key = result[len("dockerfile:") :]
        assert (
            DaytonaProvider._dockerfile_registry[key]["server_cmd"]
            == "uvicorn app:app --port 8000"
        )

    def test_no_cmd_means_none_in_registry(self, tmp_path):
        """Without CMD in Dockerfile, server_cmd is None in the registry."""
        df = tmp_path / "Dockerfile"
        df.write_text("FROM python:3.11\nRUN pip install flask\n")
        result = DaytonaProvider.image_from_dockerfile(str(df))
        key = result[len("dockerfile:") :]
        assert DaytonaProvider._dockerfile_registry[key]["server_cmd"] is None


# ---------------------------------------------------------------------------
# Tests: start_container with "dockerfile:" prefix
# ---------------------------------------------------------------------------
class TestStartContainerWithDockerfilePrefix:
    def test_dockerfile_prefix_uses_image_params(self, provider, tmp_path):
        """A 'dockerfile:' string uses CreateSandboxFromImageParams."""
        df = tmp_path / "Dockerfile"
        df.write_text("FROM python:3.11\n")
        image = DaytonaProvider.image_from_dockerfile(str(df))
        provider.start_container(image, cmd="python serve.py")
        params, _ = provider._daytona._created[0]
        assert isinstance(params, _fake_daytona.CreateSandboxFromImageParams)

    def test_string_image_still_works(self, provider):
        """Backward compat: plain string images still work."""
        url = provider.start_container("echo-env:latest")
        assert url.startswith("https://")

    def test_snapshot_string_still_works(self, provider):
        """Backward compat: snapshot: prefix still works."""
        provider.start_container("snapshot:my-snap")
        params, _ = provider._daytona._created[0]
        assert isinstance(params, _fake_daytona.CreateSandboxFromSnapshotParams)

    def test_dockerfile_prefix_with_resources(self, tmp_path):
        """dockerfile: + resources are both forwarded."""
        df = tmp_path / "Dockerfile"
        df.write_text("FROM python:3.11\n")
        image = DaytonaProvider.image_from_dockerfile(str(df))
        resources = _fake_daytona.Resources(cpu=4, memory=8)
        p = DaytonaProvider(api_key="k", resources=resources)
        p.start_container(image, cmd="python serve.py")
        params, _ = p._daytona._created[0]
        assert params.resources is resources

    def test_dockerfile_prefix_cmd_discovery(self, tmp_path):
        """dockerfile: triggers same openenv.yaml auto-discovery."""
        df = tmp_path / "Dockerfile"
        df.write_text("FROM python:3.11\n")
        image = DaytonaProvider.image_from_dockerfile(str(df))
        p = DaytonaProvider(api_key="k")
        p.start_container(image)
        _assert_exec_called_with_fragment(
            p._sandbox, "cd /app/env && python -m uvicorn server.app:app"
        )

    def test_dockerfile_prefix_without_registry_raises(self, provider):
        """Passing 'dockerfile:...' without calling image_from_dockerfile raises."""
        with pytest.raises(ValueError, match="No registered Dockerfile metadata"):
            provider.start_container("dockerfile:/no/such/path")

    def test_temp_files_cleaned_after_start(self, provider, tmp_path):
        """Temp .dockerfile files created during start are cleaned up."""
        df = tmp_path / "Dockerfile"
        df.write_text(
            "FROM python:3.11\nRUN --mount=type=cache,target=/x pip install\n"
        )
        image = DaytonaProvider.image_from_dockerfile(
            str(df), context_dir=str(tmp_path)
        )
        provider.start_container(image, cmd="python serve.py")
        leftover = list(tmp_path.glob("*.dockerfile"))
        assert leftover == [], f"Temp files not cleaned up: {leftover}"


# ---------------------------------------------------------------------------
# Tests: server process crash detection
# ---------------------------------------------------------------------------
class TestServerCrashDetection:
    def test_dead_process_raises_with_log(self):
        """wait_for_ready raises RuntimeError with log when server process is dead."""
        p = DaytonaProvider(api_key="k", cmd="python -m broken_server")

        def _exec(cmd, **kw):
            if "kill -0" in cmd:
                return "DEAD"
            if "cat /tmp/openenv-server.log" in cmd:
                return "ModuleNotFoundError: No module named 'broken_server'"
            return ""

        original_create = p._daytona.create

        def patched_create(params=None, **kwargs):
            sandbox = original_create(params, **kwargs)
            sandbox.process.exec = MagicMock(side_effect=_exec)
            return sandbox

        p._daytona.create = patched_create

        import requests

        url = p.start_container("img:latest")
        with patch("requests.get", side_effect=requests.ConnectionError("refused")):
            with pytest.raises(RuntimeError, match="Server process died") as exc_info:
                p.wait_for_ready(url)
        assert "broken_server" in str(exc_info.value)

    def test_dead_process_cleans_up_sandbox(self):
        """Sandbox can be cleaned up after wait_for_ready detects a crash."""
        p = DaytonaProvider(api_key="k", cmd="python -m broken")

        def _exec(cmd, **kw):
            if "kill -0" in cmd:
                return "DEAD"
            if "cat /tmp/openenv-server.log" in cmd:
                return "crash log"
            return ""

        original_create = p._daytona.create

        def patched_create(params=None, **kwargs):
            sandbox = original_create(params, **kwargs)
            sandbox.process.exec = MagicMock(side_effect=_exec)
            return sandbox

        p._daytona.create = patched_create

        import requests

        url = p.start_container("img:latest")
        assert p._sandbox is not None
        with patch("requests.get", side_effect=requests.ConnectionError("refused")):
            with pytest.raises(RuntimeError):
                p.wait_for_ready(url)
        # Caller is responsible for cleanup
        p.stop_container()
        assert p._sandbox is None


# ---------------------------------------------------------------------------
# Tests: Dockerfile CMD fallback when openenv.yaml is missing
# ---------------------------------------------------------------------------
class TestDockerfileCmdFallback:
    def test_fallback_to_dockerfile_cmd(self, tmp_path):
        """When openenv.yaml is missing, falls back to CMD parsed from Dockerfile."""
        df = tmp_path / "Dockerfile"
        df.write_text(
            "FROM python:3.11\nCMD uvicorn myapp:app --host 0.0.0.0 --port 8000\n"
        )
        image = DaytonaProvider.image_from_dockerfile(str(df))
        p = DaytonaProvider(api_key="k")

        def _exec(cmd, **kw):
            # No openenv.yaml anywhere
            if "test -f /app/env/openenv.yaml" in cmd:
                return ""
            if cmd.startswith("find /app"):
                return ""
            return ""

        original_create = p._daytona.create

        def patched_create(params=None, **kwargs):
            sandbox = original_create(params, **kwargs)
            sandbox.process.exec = MagicMock(side_effect=_exec)
            return sandbox

        p._daytona.create = patched_create

        url = p.start_container(image)
        assert url.startswith("https://")
        _assert_exec_called_with_fragment(p._sandbox, "uvicorn myapp:app")

    def test_no_yaml_no_dockerfile_cmd_raises(self, tmp_path):
        """When neither openenv.yaml nor Dockerfile CMD is available, raises."""
        df = tmp_path / "Dockerfile"
        df.write_text("FROM python:3.11\nRUN pip install flask\n")
        image = DaytonaProvider.image_from_dockerfile(str(df))
        p = DaytonaProvider(api_key="k")

        def _exec(cmd, **kw):
            return ""  # nothing found anywhere

        original_create = p._daytona.create

        def patched_create(params=None, **kwargs):
            sandbox = original_create(params, **kwargs)
            sandbox.process.exec = MagicMock(side_effect=_exec)
            return sandbox

        p._daytona.create = patched_create

        with pytest.raises(ValueError, match="Could not find openenv.yaml"):
            p.start_container(image)
