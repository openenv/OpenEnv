"""The verifier-informed expert.

The expert is a strong, frontier chat model that the small agent can consult on
demand through the ``ask_expert`` tool. Before a task starts, the expert reads the
task's Python verifier and distils it into a short list of success criteria; when the
agent asks for help, the expert combines those criteria with the live tool schemas to
return a concrete plan.

This privileged, verifier-informed signal is a *training-time curriculum* (a teacher
with tool knowledge), not an environment capability. The expert lives entirely in the
client: it is never registered as an MCP tool and never reaches the environment
server, and it reads verifier code from the public dataset (see [`awm_data`]) rather
than from any server module. Honest, expert-free evaluation is reported separately by
the benchmark.
"""

from __future__ import annotations

from typing import Any

import awm_data
from llm import chat_complete
from prompts import EXPERT_SYSTEM_PROMPT, VERIFIER_ANALYSIS_PROMPT


class VerifierInformedExpert:
    """A frontier advisor that turns verifier code into agent guidance.

    Args:
        client:
            An `AsyncOpenAI` / `AsyncAzureOpenAI` instance hosting the expert model.
        model (`str`):
            Deployment or model name for the expert.
        temperature (`float`, *optional*):
            Sampling temperature. Left unset by default so reasoning models (which
            only accept their default) work without special-casing.
        analysis_max_tokens (`int`, *optional*, defaults to `2000`):
            Token budget for distilling the verifier into success criteria. Generous
            so reasoning models have room to think before answering.
        plan_max_tokens (`int`, *optional*, defaults to `4000`):
            Token budget for a guidance plan.
    """

    def __init__(
        self,
        client: Any,
        model: str,
        *,
        temperature: float | None = None,
        analysis_max_tokens: int = 2000,
        plan_max_tokens: int = 4000,
    ) -> None:
        self._client = client
        self._model = model
        self._temperature = temperature
        self._analysis_max_tokens = analysis_max_tokens
        self._plan_max_tokens = plan_max_tokens

    def verifier_code(self, scenario: str, task_idx: int) -> str | None:
        """Return the Python verifier source for a task, or `None` if absent."""
        return awm_data.get_verifier_code(scenario, task_idx)

    async def analyze(self, task: str, scenario: str, task_idx: int) -> str:
        """Distil the task's verifier into a short list of success criteria.

        Returns an empty string when no verifier is available or the call fails.
        """
        code = self.verifier_code(scenario, task_idx)
        if not code:
            return ""
        prompt = VERIFIER_ANALYSIS_PROMPT.format(code=code, task=task)
        try:
            return await chat_complete(
                self._client,
                self._model,
                [{"role": "user", "content": prompt}],
                max_tokens=self._analysis_max_tokens,
                temperature=self._temperature,
            )
        except Exception:
            return ""

    async def advise(
        self,
        task: str,
        *,
        tool_schemas: str = "",
        context: str = "",
        success_criteria: str = "",
    ) -> str:
        """Produce a step-by-step plan for the agent.

        Args:
            task (`str`):
                The task the agent is solving.
            tool_schemas (`str`, *optional*):
                Formatted tool list with parameter schemas.
            context (`str`, *optional*):
                What the agent has tried so far, including errors.
            success_criteria (`str`, *optional*):
                The distilled verifier requirements from [`analyze`].

        Returns:
            `str`: The expert's plan, or a safe fallback message on failure.
        """
        sections = [f"TASK: {task}"]
        if tool_schemas:
            sections.append(
                f"AVAILABLE TOOLS (with parameter schemas):\n{tool_schemas}"
            )
        if success_criteria:
            sections.append(
                f"SUCCESS CRITERIA (what the checker requires):\n{success_criteria}"
            )
        sections.append(f"AGENT CONTEXT: {context or 'The agent has just started.'}")
        sections.append(
            "Provide a precise, ordered plan with exact tool names and argument "
            "values that satisfies every success criterion."
        )
        prompt = "\n\n".join(sections)
        try:
            return await chat_complete(
                self._client,
                self._model,
                [
                    {"role": "system", "content": EXPERT_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=self._plan_max_tokens,
                temperature=self._temperature,
            )
        except Exception:
            return "Expert unavailable; proceed with your best judgement."
