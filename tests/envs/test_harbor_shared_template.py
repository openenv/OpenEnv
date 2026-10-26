"""One template per distinct environment, not one per task.

Harbor builds a template per task because it names the alias from `task.short_name`. For a suite whose
tasks share one image that is thousands of identical builds, and worse: `alias_exists()` goes true when
a build STARTS, so a GRPO group racing a task's first visit 404s on `tag 'default' does not exist`.

These tests assert on the `environment_name` Harbor's own constructor RECEIVES, because that is the
value the alias is built from — asserting on our wrapper would only prove the wrapper ran.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import httpx
import pytest
from openenv.harbor import shared_template


@pytest.fixture(autouse=True)
def _reset():
    """`enable_shared_templates` is idempotent by design, so tests must clear the latch."""
    shared_template._applied = False
    yield
    shared_template._applied = False


@pytest.fixture
def recorder(monkeypatch):
    """Replace Harbor's E2B constructor with a recorder, then let the patch wrap the recorder."""
    e2b = pytest.importorskip("harbor.environments.e2b")
    seen: list[str] = []

    def fake_init(self, *args, **kwargs):
        seen.append(kwargs.get("environment_name") if kwargs else args[1])

    monkeypatch.setattr(e2b.E2BEnvironment, "__init__", fake_init, raising=True)
    # Restore the production method after enable_shared_templates wraps it.
    monkeypatch.setattr(
        e2b.E2BEnvironment,
        "_does_template_exist",
        e2b.E2BEnvironment._does_template_exist,
    )
    return e2b, seen


def test_the_shared_name_reaches_harbors_constructor(recorder, monkeypatch):
    e2b, seen = recorder
    monkeypatch.delenv("HARBOR_SHARED_ENV_NAME", raising=False)
    assert shared_template.enable_shared_templates("oe-dataagent") is True
    e2b.E2BEnvironment(object(), environment_name="0000_555_555434_qa_3")
    assert seen == ["oe-dataagent"], "the per-task name still reached Harbor"


def test_it_reads_the_env_var(recorder, monkeypatch):
    e2b, seen = recorder
    monkeypatch.setenv("HARBOR_SHARED_ENV_NAME", "from-env")
    assert shared_template.enable_shared_templates() is True
    e2b.E2BEnvironment(object(), environment_name="0000_650_650548_qa_2")
    assert seen == ["from-env"]


def test_unset_is_a_no_op(recorder, monkeypatch):
    """Default behaviour must be untouched: no variable, no renaming."""
    e2b, seen = recorder
    monkeypatch.delenv("HARBOR_SHARED_ENV_NAME", raising=False)
    assert shared_template.enable_shared_templates() is False
    e2b.E2BEnvironment(object(), environment_name="0000_555_555434_qa_3")
    assert seen == ["0000_555_555434_qa_3"]


def test_a_positional_name_is_still_rewritten(recorder, monkeypatch):
    """Harbor passes it by keyword today; a future positional call must not silently opt out."""
    e2b, seen = recorder
    monkeypatch.delenv("HARBOR_SHARED_ENV_NAME", raising=False)
    shared_template.enable_shared_templates("oe-dataagent")
    e2b.E2BEnvironment(object(), "0000_555_555434_qa_3", "session")
    assert seen == ["oe-dataagent"]


@pytest.mark.asyncio
async def test_read_only_lookup_recovers_closed_http2_connection(recorder, monkeypatch):
    e2b, _ = recorder
    lookup = AsyncMock(
        side_effect=[
            httpx.LocalProtocolError(
                "Invalid input ConnectionInputs.RECV_PING in state ConnectionState.CLOSED"
            ),
            True,
        ]
    )
    sleep = AsyncMock()
    create = e2b.E2BEnvironment._create_sandbox
    monkeypatch.setattr(e2b.E2BEnvironment, "_does_template_exist", lookup)
    monkeypatch.setattr(shared_template.asyncio, "sleep", sleep)
    shared_template.enable_shared_templates("test-shared")
    wrapped = e2b.E2BEnvironment._does_template_exist
    shared_template.enable_shared_templates("test-shared")
    assert e2b.E2BEnvironment._does_template_exist is wrapped
    env = e2b.E2BEnvironment(object(), environment_name="task")
    assert await env._does_template_exist() is True
    assert lookup.await_count == 2
    sleep.assert_awaited_once_with(1)
    assert e2b.E2BEnvironment._create_sandbox is create


@pytest.mark.asyncio
async def test_lookup_transport_retries_are_bounded(recorder, monkeypatch):
    e2b, _ = recorder
    failure = httpx.ReadTimeout("lookup timed out")
    lookup = AsyncMock(side_effect=failure)
    sleep = AsyncMock()
    monkeypatch.setattr(e2b.E2BEnvironment, "_does_template_exist", lookup)
    monkeypatch.setattr(shared_template.asyncio, "sleep", sleep)
    shared_template.enable_shared_templates("test-shared")
    with pytest.raises(httpx.ReadTimeout) as caught:
        await e2b.E2BEnvironment(
            object(), environment_name="task"
        )._does_template_exist()
    assert caught.value is failure
    assert lookup.await_count == 3
    assert [call.args for call in sleep.await_args_list] == [(1,), (2,)]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "outcome",
    [
        False,
        httpx.HTTPStatusError(
            "unauthorized",
            request=httpx.Request("GET", "https://example.test"),
            response=httpx.Response(401),
        ),
        asyncio.CancelledError(),
    ],
)
async def test_lookup_does_not_retry_missing_alias_permanent_error_or_cancellation(
    recorder, monkeypatch, outcome
):
    e2b, _ = recorder
    lookup = (
        AsyncMock(side_effect=outcome)
        if isinstance(outcome, BaseException)
        else AsyncMock(return_value=outcome)
    )
    sleep = AsyncMock()
    monkeypatch.setattr(e2b.E2BEnvironment, "_does_template_exist", lookup)
    monkeypatch.setattr(shared_template.asyncio, "sleep", sleep)
    shared_template.enable_shared_templates("test-shared")
    env = e2b.E2BEnvironment(object(), environment_name="task")
    if isinstance(outcome, BaseException):
        with pytest.raises(type(outcome)):
            await env._does_template_exist()
    else:
        assert await env._does_template_exist() is False
    assert lookup.await_count == 1
    sleep.assert_not_awaited()
