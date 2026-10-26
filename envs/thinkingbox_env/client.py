"""Provide the typed OpenEnv client for ThinkingBox benchmark episodes.

The client only handles public WebSocket actions and observations; private
benchmark state, user context, and grading remain inside the server.
"""

from typing import Any

from openenv.core import EnvClient, ListToolsAction
from openenv.core.client_types import StepResult

from .models import (
    _FinishAction,
    CallToolAction,
    SubmitMessageAction,
    SubmittedToolCall,
    ThinkingBoxAction,
    ThinkingBoxObservation,
    ThinkingBoxState,
)


# A turn can include long agent, user-simulator, and judge operations, so the
# client timeout intentionally exceeds individual server and proxy timeouts.
DEFAULT_MESSAGE_TIMEOUT_S = 1800.0


class ThinkingBoxEnv(
    EnvClient[ThinkingBoxAction, ThinkingBoxObservation, ThinkingBoxState]
):
    """Connect a trusted harness to one ThinkingBox environment instance.

    Args:
        base_url (`str`, *optional*):
            OpenEnv server URL.
        message_timeout_s (`float`, *optional*, defaults to `1800.0`):
            Timeout for each WebSocket operation.
        kwargs (`dict`, *optional*):
            Additional arguments forwarded to [`~openenv.core.EnvClient`].

    Examples:

    ```python
    async with ThinkingBoxEnv("http://127.0.0.1:8000") as env:
        result = await env.reset("file.py:test_name")
    ```
    """

    def __init__(
        self,
        base_url: str | None = None,
        *,
        message_timeout_s: float = DEFAULT_MESSAGE_TIMEOUT_S,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            base_url=base_url,
            message_timeout_s=message_timeout_s,
            **kwargs,
        )

    def reset(
        self,
        test_uid: str,
        *,
        dataset: str | None = None,
        agent: str | None = None,
        config: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Start the task selected by its canonical UID.

        Args:
            test_uid (`str`):
                UID from the resolved release manifest.
            dataset (`str`, *optional*):
                Server-visible executable data root or `dataset/` path.
            agent (`str`, *optional*):
                Native ThinkingBox agent definition. When omitted, the server
                uses `OPENENV_TB_AGENT`.
            config (`str`, *optional*):
                Server-visible native ThinkingBox configuration path.
            kwargs (`dict`, *optional*):
                Additional OpenEnv reset arguments, including accepted but
                intentionally unused `seed`.

        Returns:
            A client result containing [`ThinkingBoxObservation`].
        """
        reset_kwargs: dict[str, Any] = {"test_uid": test_uid, **kwargs}
        if dataset is not None:
            reset_kwargs["dataset"] = dataset
        if agent is not None:
            reset_kwargs["agent"] = agent
        if config is not None:
            reset_kwargs["config"] = config
        return super().reset(**reset_kwargs)

    def list_tools(self) -> Any:
        """Return the tools permitted for the active scenario.

        Returns:
            A client result whose observation contains public tool schemas.
        """
        return self.step(ListToolsAction())

    def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        *,
        call_id: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Execute one permitted tool call as an agent action.

        Args:
            name (`str`):
                Tool name.
            arguments (`dict`, *optional*):
                Provider-supplied tool arguments.
            call_id (`str`, *optional*):
                Provider tool-call identifier.
            kwargs (`dict`, *optional*):
                Additional arguments merged into `arguments`.

        Returns:
            A client result containing the tool response or terminal outcome.
        """
        merged = dict(arguments or {})
        merged.update(kwargs)
        return self.step(
            CallToolAction(tool_name=name, arguments=merged, call_id=call_id)
        )

    def call_tools(self, calls: list[SubmittedToolCall]) -> Any:
        """Execute one native parallel tool batch as one agent action.

        Args:
            calls (`list[SubmittedToolCall]`):
                Ordered provider tool calls with unique call identifiers.

        Returns:
            A client result containing ordered batch results or a terminal
            outcome.
        """
        return self.step(CallToolAction(parallel_tool_calls=calls))

    def submit_message(
        self,
        content: str | None,
        *,
        terminal_tool_calls: list[SubmittedToolCall] | None = None,
        tool_calls_before_content: bool = False,
    ) -> Any:
        """Submit assistant-visible text or a terminal provider turn.

        Args:
            content (`str`, *optional*):
                Assistant text visible to the user and grader.
            terminal_tool_calls (`list[SubmittedToolCall]`, *optional*):
                Unexecuted calls from a provider turn that ended the episode.
            tool_calls_before_content (`bool`, *optional*, defaults to `False`):
                Whether provider ordering placed terminal calls before text.

        Returns:
            A client result containing the simulated-user reply or terminal
            evaluation.
        """
        return self.step(
            SubmitMessageAction(
                content=content,
                terminal_tool_calls=terminal_tool_calls or [],
                tool_calls_before_content=tool_calls_before_content,
            )
        )

    def finish(self, reason: str = "harness") -> Any:
        """Finish from trusted harness code without advertising a model tool.

        Args:
            reason (`str`, *optional*, defaults to `"harness"`):
                Trusted stop reason recorded in the terminal observation.

        Returns:
            A client result containing the final native evaluation.
        """
        return self.step(_FinishAction(reason=reason))

    def _step_payload(self, action: ThinkingBoxAction) -> dict[str, Any]:
        return action.model_dump(mode="json")

    def _parse_result(
        self, payload: dict[str, Any]
    ) -> StepResult[ThinkingBoxObservation]:
        data = dict(payload.get("observation", {}))
        data["reward"] = payload.get("reward")
        data["done"] = payload.get("done", False)
        if payload.get("metadata") is not None:
            data["metadata"] = payload["metadata"]
        observation = ThinkingBoxObservation.model_validate(data)
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
            metadata=payload.get("metadata"),
        )

    def _parse_state(self, payload: dict[str, Any]) -> ThinkingBoxState:
        return ThinkingBoxState.model_validate(payload)
