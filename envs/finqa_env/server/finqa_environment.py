# envs/finqa_env/server/finqa_environment.py
"""
FinQA Environment Implementation.

A financial question-answering environment that evaluates LLMs on their ability
to answer complex financial questions using tool calls on SEC 10-K filing data.
"""

import logging
import os
import random
import uuid
from typing import Any, Dict, List, Optional

import pandas as pd
from fastmcp import FastMCP
from openenv.core.env_server.mcp_environment import MCPEnvironment
from openenv.core.env_server.mcp_types import CallToolAction
from openenv.core.env_server.types import Action, Observation

from ..models import AVAILABLE_TOOLS, FinQAState
from .rewards import compute_reward
from .tools import FinQATools

logger = logging.getLogger(__name__)


class FinQAEnvironment(MCPEnvironment):
    """
    Financial QA environment for RL training.

    Evaluates agents on their ability to answer financial questions by:
    - Exploring available tables for a company
    - Querying table metadata and executing SQL queries
    - Performing calculations
    - Submitting final answers

    Args:
        data_path: Path to the data directory containing benchmark_questions/ and input_companies/
        max_steps: Maximum number of tool calls per episode (default: 50)
        task: Task name - currently only 'finqa' supported (default: 'finqa')
    """

    def __init__(
        self,
        data_path: str = "./data",
        max_steps: int = 50,
        task: str = "finqa",
    ):
        # Create MCP server and define tools inline
        mcp = FastMCP("finqa_env")

        self.data_path = data_path
        self.max_steps = max_steps
        self.task = task

        assert task == "finqa", "Only finqa task is supported"

        self.questions = self._load_questions()
        logger.info(f"Loaded {len(self.questions)} questions for task '{task}'")

        self._finqa_tools = FinQATools(data_path)

        # Register tools with FastMCP
        @mcp.tool
        def get_descriptions(company_name: str) -> str:
            """
            Get a list of available table names for a company.

            Args:
                company_name: The name of the company

            Returns:
                JSON list of table names
            """
            return self._finqa_tools.get_descriptions(company_name)

        @mcp.tool
        def get_table_info(company_name: str, table_name: str) -> str:
            """
            Get table metadata: description, columns, types, unique values.

            Args:
                company_name: The name of the company
                table_name: The name of the table

            Returns:
                JSON string with table metadata
            """
            return self._finqa_tools.get_table_info(company_name, table_name)

        @mcp.tool
        def sql_query(company_name: str, table_name: str, query: str) -> str:
            """
            Execute a SQL query on a table. Select * not allowed.

            Filters are required: WHERE, HAVING, IN, NOT IN, EXISTS, NOT EXISTS,
            ANY, SOME, ALL, LIKE, NOT LIKE, BETWEEN, NOT BETWEEN, IS NULL,
            IS NOT NULL, CASE, FILTER.

            Args:
                company_name: The name of the company
                table_name: The name of the table
                query: SQL query to execute (must include filters)

            Returns:
                JSON string with query results
            """
            return self._finqa_tools.sql_query(company_name, table_name, query)

        @mcp.tool
        def submit_answer(answer: str) -> str:
            """
            Submit a final answer for the question.

            Args:
                answer: The final answer to submit

            Returns:
                Confirmation message
            """
            return self._finqa_tools.submit_answer(answer)

        # Pass the MCP server to the base class
        super().__init__(mcp)

        # Shuffle dataset for sequential selection. The env owns its RNG so
        # that reset(seed=...) can make the question order reproducible.
        self._rng = random.Random()
        self._start_new_pass()

        self._state = FinQAState()
        self._history: List[Dict[str, Any]] = []

    def _load_questions(self) -> List[Dict[str, Any]]:
        """Load questions from the benchmark CSV."""
        csv_path = os.path.join(
            self.data_path, "benchmark_questions", f"{self.task}.csv"
        )

        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"Benchmark file not found: {csv_path}")

        df = pd.read_csv(csv_path)

        questions = []
        for _, row in df.iterrows():
            questions.append(
                {
                    "id": str(row.get("id", "")),
                    "user_query": row["user_query"],
                    "company": row["company"],
                    "question": row["question"],
                    "answer": row["answer"],
                    "question_type": row.get("question_type", ""),
                    "explanation": row.get("explanation", ""),
                }
            )

        return questions

    def _start_new_pass(self) -> None:
        """Shuffle the full dataset into a new order and start from the top."""
        self._shuffled_questions = self.questions.copy()
        self._rng.shuffle(self._shuffled_questions)
        self._question_index = 0

    def _get_next_question(self) -> Dict[str, Any]:
        """Get the next question using sequential shuffle selection."""
        if self._question_index >= len(self._shuffled_questions):
            self._start_new_pass()

        question = self._shuffled_questions[self._question_index]
        self._question_index += 1
        return question

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Observation:
        """
        Reset the environment for a new episode.

        Args:
            seed: If given, reseed the question order. The same seed always
                yields the same question, and the same sequence of questions
                on the unseeded resets that follow.
            episode_id: Optional episode identifier (generated if omitted)

        Returns:
            Initial observation with the question
        """
        if seed is not None:
            self._rng = random.Random(seed)
            self._start_new_pass()

        question = self._get_next_question()
        self._state = FinQAState(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
            current_question=question["user_query"],
            current_company=question["company"],
            ground_truth=question["answer"],
            question_id=question["id"],
        )
        self._history = []

        logger.info(
            f"Reset episode {self._state.episode_id} with question: {question['question'][:200]}..."
        )

        return Observation(
            done=False,
            reward=0.0,
            metadata={
                "question": question["user_query"],
                "company": question["company"],
                "tool_result": "",
                "history": [],
                "step_count": 0,
                "available_tools": AVAILABLE_TOOLS.copy(),
            },
        )

    def _step_impl(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        """
        Handle non-MCP actions. Returns an error since this env is MCP-only.
        """
        return Observation(
            done=False,
            reward=0.0,
            metadata={
                "error": f"Unknown action type: {type(action).__name__}. "
                "Use ListToolsAction or CallToolAction for MCP interactions."
            },
        )

    def step(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        """
        Execute a step in the environment.

        Delegates to base class for MCP actions. Handles submit_answer
        reward computation and max-step termination.
        """
        self._state.step_count += 1

        # Let the base class handle MCP actions
        obs = super().step(action, timeout_s=timeout_s, **kwargs)

        # Check if submit_answer was called
        if isinstance(action, CallToolAction) and action.tool_name == "submit_answer":
            submitted_answer = action.arguments.get("answer", "")
            reward = compute_reward(submitted_answer, self._state.ground_truth)
            logger.info(
                f"Episode {self._state.episode_id} ended: "
                f"submitted='{submitted_answer}', truth='{self._state.ground_truth}', reward={reward}"
            )
            return Observation(
                done=True,
                reward=reward,
                metadata={
                    **obs.metadata,
                    "ground_truth": self._state.ground_truth,
                    "submitted_answer": submitted_answer,
                },
            )

        # Check for max steps
        if self._state.step_count >= self.max_steps:
            logger.info(
                f"Episode {self._state.episode_id} terminated: max steps reached"
            )
            return Observation(
                done=True,
                reward=0.0,
                metadata={
                    **obs.metadata,
                    "error": f"Max steps ({self.max_steps}) reached without submitting answer.",
                },
            )

        return obs

    async def step_async(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        """
        Async step used by the WebSocket handler.

        Mirrors step() logic (step count, submit_answer reward, max-step
        termination) but delegates to MCPEnvironment.step_async to avoid
        the run_async_safely event-loop deadlock on the WebSocket path.
        """
        self._state.step_count += 1

        obs = await super().step_async(action, timeout_s=timeout_s, **kwargs)

        if isinstance(action, CallToolAction) and action.tool_name == "submit_answer":
            submitted_answer = action.arguments.get("answer", "")
            reward = compute_reward(submitted_answer, self._state.ground_truth)
            logger.info(
                f"Episode {self._state.episode_id} ended: "
                f"submitted='{submitted_answer}', truth='{self._state.ground_truth}', reward={reward}"
            )
            return Observation(
                done=True,
                reward=reward,
                metadata={
                    **obs.metadata,
                    "ground_truth": self._state.ground_truth,
                    "submitted_answer": submitted_answer,
                },
            )

        if self._state.step_count >= self.max_steps:
            logger.info(
                f"Episode {self._state.episode_id} terminated: max steps reached"
            )
            return Observation(
                done=True,
                reward=0.0,
                metadata={
                    **obs.metadata,
                    "error": f"Max steps ({self.max_steps}) reached without submitting answer.",
                },
            )

        return obs

    @property
    def state(self) -> FinQAState:
        """Get the current environment state."""
        return self._state
