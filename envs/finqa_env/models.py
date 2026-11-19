# envs/finqa_env/models.py
"""
State types for the FinQA environment.

FinQA is a financial question-answering benchmark that evaluates LLMs on their
ability to answer complex financial questions using tool calls (SQL queries,
calculations, etc.) on SEC 10-K filing data.

This environment uses the MCP protocol for tool interactions. Use
``CallToolAction`` and ``ListToolsAction`` from ``openenv.core.env_server.mcp_types``
to interact with the environment.
"""

from openenv.core.env_server import State
from openenv.core.env_server.types import ResetRequest
from pydantic import Field


# Tool names - defined statically to avoid circular imports
AVAILABLE_TOOLS = ["get_descriptions", "get_table_info", "sql_query", "submit_answer"]


class FinQAResetRequest(ResetRequest):
    """
    Reset parameters that select a task, published on `/schema`.

    A hint for clients: requests are not validated against it.
    """

    split: str = Field(default="test", description="Task split")
    index: int | None = Field(
        default=None,
        description="Row of the split to load; without it, the seeded order picks",
    )


class FinQAState(State):
    """
    Internal environment state for tracking the current episode.

    All fields are set during reset() and are essential for episode tracking.

    Attributes:
        current_question: The question being asked
        current_company: The company the question is about
        ground_truth: The expected answer for reward computation
        question_id: Identifier for the current question
        # Inherited from State: episode_id, step_count
    """

    current_question: str = ""
    current_company: str = ""
    ground_truth: str = ""
    question_id: str = ""
