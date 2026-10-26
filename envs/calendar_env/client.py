# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Calendar Environment HTTP client.

This client uses the OpenEnv HTTP endpoints exposed by the calendar server:
/reset, /step, and /state. It also includes helpers for database seeding.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import string
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Generic, List, Optional, TYPE_CHECKING, TypeVar

import httpx

if TYPE_CHECKING:
    from openenv.core.llm_client import LLMClient

try:
    from .models import CalendarAction, CalendarObservation
except ImportError:
    import sys
    from pathlib import Path

    CURRENT_DIR = Path(__file__).resolve().parent
    if str(CURRENT_DIR) not in sys.path:
        sys.path.insert(0, str(CURRENT_DIR))
    from models import CalendarAction, CalendarObservation

ObsT = TypeVar("ObsT")


@dataclass
class StepResult(Generic[ObsT]):
    observation: ObsT
    reward: Optional[float] = None
    done: bool = False
    metadata: Optional[Dict[str, Any]] = None


logger = logging.getLogger(__name__)
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "response_output"


def _generate_database_id() -> str:
    timestamp = int(time.time() * 1000)
    suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=9))
    return f"db_{timestamp}_{suffix}"


class CalendarEnv:
    """HTTP client for the Calendar environment."""

    def __init__(
        self,
        base_url: str,
        database_id: str = "default",
        access_token: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        timeout_s: float = 60.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.database_id = database_id
        self.access_token = access_token
        self.context = context or {}
        self.timeout_s = timeout_s
        self._client: Optional[httpx.Client] = None

    def __enter__(self) -> "CalendarEnv":
        self._ensure_client()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def _ensure_client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=self.timeout_s)
        return self._client

    def _headers(
        self,
        database_id: Optional[str] = None,
        access_token: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, str]:
        headers: Dict[str, str] = {"Content-Type": "application/json"}

        resolved_db = database_id if database_id is not None else self.database_id
        if resolved_db:
            headers["x-database-id"] = resolved_db

        resolved_token = access_token if access_token is not None else self.access_token
        if resolved_token:
            headers["x-access-token"] = resolved_token

        merged_context = dict(self.context)
        if context:
            merged_context.update(context)

        for key, value in merged_context.items():
            if value is None:
                continue
            header_key = str(key).strip().lower().replace("_", "-")
            if not header_key.startswith("x-"):
                header_key = f"x-{header_key}"
            if header_key in headers:
                continue
            headers[header_key] = str(value)

        return headers

    def _parse_step_result(
        self, payload: Dict[str, Any]
    ) -> StepResult[CalendarObservation]:
        obs_data = payload.get("observation", {})
        observation = CalendarObservation(
            success=obs_data.get("success", True),
            error_message=obs_data.get("error_message"),
            tools_list=obs_data.get("tools_list"),
            tool_result=obs_data.get("tool_result"),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=payload.get("metadata", obs_data.get("metadata", {})),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
            metadata=payload.get("metadata"),
        )

    def reset(
        self,
        database_id: Optional[str] = None,
        sql_content: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> StepResult[CalendarObservation]:
        payload: Dict[str, Any] = {}
        if database_id:
            payload["database_id"] = database_id
        if sql_content:
            payload["sql_content"] = sql_content

        client = self._ensure_client()
        response = client.post(
            f"{self.base_url}/reset",
            json=payload if payload else None,
            headers=self._headers(database_id=database_id, context=context),
        )
        response.raise_for_status()
        return self._parse_step_result(response.json())

    def reset_with_sql_file(
        self,
        sql_file_path: str | Path,
        database_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> StepResult[CalendarObservation]:
        sql_path = Path(sql_file_path).expanduser().resolve()
        if not sql_path.exists():
            raise FileNotFoundError(f"SQL file not found: {sql_path}")
        sql_content = sql_path.read_text(encoding="utf-8")
        return self.reset(
            database_id=database_id, sql_content=sql_content, context=context
        )

    def step(
        self,
        action: CalendarAction | Dict[str, Any],
        database_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> StepResult[CalendarObservation]:
        if isinstance(action, CalendarAction):
            action_payload = action.model_dump(exclude_none=True)
        elif isinstance(action, dict):
            action_payload = action
        else:
            raise TypeError("action must be CalendarAction or dict")

        client = self._ensure_client()
        response = client.post(
            f"{self.base_url}/step",
            json=action_payload,
            headers=self._headers(database_id=database_id, context=context),
        )
        response.raise_for_status()
        return self._parse_step_result(response.json())

    def list_tools(
        self,
        database_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        result = self.step(
            CalendarAction(action_type="ListToolsAction"),
            database_id=database_id,
            context=context,
        )
        return result.observation.tools_list or []

    def call_tool(
        self,
        tool_name: str,
        arguments: Optional[Dict[str, Any]] = None,
        database_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> CalendarObservation:
        result = self.step(
            CalendarAction(
                action_type="ToolCallAction",
                tool_name=tool_name,
                arguments=arguments or {},
            ),
            database_id=database_id,
            context=context,
        )
        return result.observation

    def state(
        self,
        verify_queries: Optional[List[str]] = None,
        database_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        params = (
            [("verify_queries", q) for q in verify_queries] if verify_queries else None
        )
        client = self._ensure_client()
        response = client.get(
            f"{self.base_url}/state",
            params=params,
            headers=self._headers(database_id=database_id, context=context),
        )
        response.raise_for_status()
        return response.json()

    def get_sample_sql(self) -> str:
        client = self._ensure_client()
        response = client.get(f"{self.base_url}/api/sample-data")
        response.raise_for_status()
        data = response.json()
        sql_content = data.get("sql_content")
        if not sql_content:
            raise ValueError("sample-data response did not include sql_content")
        return sql_content

    def seed_database(
        self,
        sql_content: str,
        database_id: Optional[str] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> str:
        db_id = database_id or _generate_database_id()
        payload = {
            "database_id": db_id,
            "sql_content": sql_content,
            "name": name or f"Calendar DB {db_id}",
            "description": description or "Seeded database",
        }
        client = self._ensure_client()
        response = client.post(f"{self.base_url}/api/seed-database", json=payload)
        response.raise_for_status()
        return db_id

    def seed_database_from_file(
        self,
        sql_file_path: str | Path,
        database_id: Optional[str] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> str:
        sql_path = Path(sql_file_path).expanduser().resolve()
        if not sql_path.exists():
            raise FileNotFoundError(f"SQL file not found: {sql_path}")
        sql_content = sql_path.read_text(encoding="utf-8")
        return self.seed_database(
            sql_content=sql_content,
            database_id=database_id,
            name=name,
            description=description,
        )

    def seed_database_from_sample(
        self,
        database_id: Optional[str] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> str:
        sql_content = self.get_sample_sql()
        return self.seed_database(
            sql_content=sql_content,
            database_id=database_id,
            name=name,
            description=description,
        )


@dataclass
class ScenarioConfig:
    gym_enviornment_url: str
    seed_database_file: str
    system_prompt: str
    user_prompt: str
    llm_model: str
    llm_provider: str
    llm_api_key: str
    database_id: str
    access_token: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    execution_mode: str = "openenv"
    expected_tools: List[str] = field(default_factory=list)
    restricted_tools: List[str] = field(default_factory=list)
    verifiers: List[Dict[str, Any]] = field(default_factory=list)
    number_of_runs: int = 1
    reset_database_between_runs: bool = True
    temperature: float = 0.0
    max_tokens: int = 4096
    max_iterations: int = 20
    seed_mode: str = "reset"
    output_dir: Path = DEFAULT_OUTPUT_DIR


class VerifierEngine:
    def __init__(
        self,
        client: CalendarEnv,
        llm_client: "LLMClient",
        execution_mode: str = "openenv",
    ):
        self.client = client
        self.llm_client = llm_client
        self.execution_mode = execution_mode

    async def execute_verifier(
        self,
        verifier: Dict[str, Any],
        model_response: Dict[str, Any],
        database_id: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        verifier_type = verifier.get("verifier_type")
        if verifier_type == "database_state":
            return await self._execute_database_state_verifier(
                verifier.get("validation_config", {}),
                database_id,
                context,
            )
        if verifier_type == "response_check":
            return await self._execute_response_check_verifier(
                verifier.get("validation_config", {}),
                model_response,
                database_id,
                context,
            )
        if verifier_type == "tool_execution":
            return await self._execute_tool_execution_verifier(
                verifier.get("validation_config", {}),
                model_response,
            )
        return {"passed": False, "error": f"Unsupported verifier type: {verifier_type}"}

    async def _execute_database_state_verifier(
        self,
        validation_config: Dict[str, Any],
        database_id: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        sql_query = validation_config.get("query")
        expected_value = validation_config.get("expected_value")
        comparison_type = validation_config.get("comparison_type", "equals")

        if not sql_query:
            return {"passed": False, "error": "No SQL query provided"}

        result = self._execute_sql_query(sql_query, database_id, context)
        if not result["success"]:
            return {
                "passed": False,
                "error": result.get("error", "SQL query failed"),
                "query": sql_query,
            }

        actual_value = self._extract_value_from_sql_result(result)
        comparison_result = self._compare_values(
            actual_value, expected_value, comparison_type
        )

        return {
            "passed": comparison_result["passed"],
            "expected": expected_value,
            "actual": actual_value,
            "comparison_type": comparison_type,
            "query": sql_query,
            "details": comparison_result.get("details"),
        }

    async def _execute_response_check_verifier(
        self,
        validation_config: Dict[str, Any],
        model_response: Dict[str, Any],
        database_id: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        sql_query = validation_config.get("sql_query")
        comparison_prompt = validation_config.get("comparison_prompt")
        minimum_comparison_value = validation_config.get("minimum_comparison_value", 7)

        if not sql_query or not comparison_prompt:
            return {"passed": False, "error": "Missing sql_query or comparison_prompt"}

        sql_result = self._execute_sql_query(sql_query, database_id, context)
        if not sql_result["success"]:
            return {
                "passed": False,
                "error": sql_result.get("error", "SQL query failed"),
            }

        llm_response_text = self._extract_llm_content(model_response)
        return await self._compare_with_llm(
            sql_result,
            llm_response_text,
            comparison_prompt,
            minimum_comparison_value,
        )

    async def _execute_tool_execution_verifier(
        self, validation_config: Dict[str, Any], model_response: Dict[str, Any]
    ) -> Dict[str, Any]:
        expected_tools = validation_config.get("expected_tools", [])
        minimum_tool_calls = validation_config.get("minimum_tool_calls", 1)

        tools_called = []
        if model_response.get("tool_calls"):
            tools_called = [tc.get("name") for tc in model_response["tool_calls"]]

        missing_tools = [tool for tool in expected_tools if tool not in tools_called]
        passed = len(missing_tools) == 0 and len(tools_called) >= minimum_tool_calls

        return {
            "passed": passed,
            "expected_tools": expected_tools,
            "tools_called": tools_called,
            "missing_tools": missing_tools,
            "minimum_tool_calls": minimum_tool_calls,
            "actual_tool_calls": len(tools_called),
        }

    def _execute_sql_query(
        self,
        query: str,
        database_id: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        try:
            state_data = self.client.state(
                verify_queries=[query],
                database_id=database_id,
                context=context,
            )
            verification_results = state_data.get("verification_results", [])
            if not verification_results:
                return {"success": False, "error": "No verification results returned"}

            query_result = verification_results[0]
            if "error" in query_result:
                return {
                    "success": False,
                    "error": query_result.get("error", "Query failed"),
                }

            return {"success": True, "result": query_result.get("result", [])}
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    def _extract_value_from_sql_result(self, sql_result: Dict[str, Any]) -> Any:
        result_data = sql_result.get("result", [])
        if isinstance(result_data, list) and result_data:
            first_row = result_data[0]
            if isinstance(first_row, dict) and len(first_row) == 1:
                return next(iter(first_row.values()))
        return result_data

    def _compare_values(
        self, actual: Any, expected: Any, comparison_type: str
    ) -> Dict[str, Any]:
        try:
            if comparison_type == "equals":
                passed = actual == expected
            elif comparison_type == "greater_than":
                passed = actual > expected
            elif comparison_type == "less_than":
                passed = actual < expected
            elif comparison_type == "contains":
                passed = expected in str(actual)
            else:
                return {
                    "passed": False,
                    "details": f"Unknown comparison type: {comparison_type}",
                }

            return {
                "passed": passed,
                "details": f"Comparison {comparison_type}: {actual} vs {expected}",
            }
        except Exception as exc:
            return {"passed": False, "details": f"Comparison error: {exc}"}

    def _extract_llm_content(self, model_response: Dict[str, Any]) -> str:
        for key in ("content", "text", "response"):
            if key in model_response:
                return str(model_response[key])
        return str(model_response)

    async def _compare_with_llm(
        self,
        sql_result: Dict[str, Any],
        llm_response: str,
        comparison_prompt: str,
        minimum_score: int,
    ) -> Dict[str, Any]:
        system_prompt = (
            "You are an evaluator comparing database results with an assistant response. "
            "Return only JSON in this format: "
            '{"score": <number 1-10>, "reasoning": "<short explanation>"}'
        )

        sql_result_str = json.dumps(sql_result.get("result", {}), indent=2)
        user_prompt = (
            f"Database result:\n{sql_result_str}\n\n"
            f"Assistant response:\n{llm_response}\n\n"
            f"Comparison task:\n{comparison_prompt}"
        )

        response = await self.llm_client.complete_with_tools(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            [],
        )

        try:
            response_text = response.content
            if "```json" in response_text:
                response_text = (
                    response_text.split("```json")[1].split("```")[0].strip()
                )
            judge_result = json.loads(response_text)
            score = int(judge_result.get("score", 0))
            return {
                "passed": score >= minimum_score,
                "score": score,
                "reasoning": judge_result.get("reasoning", ""),
            }
        except Exception as exc:
            return {"passed": False, "error": f"Judge parsing failed: {exc}"}


class ScenarioRunner:
    def __init__(self, config: ScenarioConfig):
        from openenv.core.llm_client import create_llm_client

        self.config = config
        self.client = CalendarEnv(
            base_url=config.gym_enviornment_url,
            database_id=config.database_id,
            access_token=config.access_token,
            context=config.context,
        )
        self.llm_client = create_llm_client(
            config.llm_provider,
            config.llm_model,
            config.llm_api_key,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )
        self.verifier_engine = VerifierEngine(
            self.client, self.llm_client, config.execution_mode
        )
        self.available_tools: List[Dict[str, Any]] = []
        self._seed_sql_content: Optional[str] = None

    def _prepare_database(self) -> None:
        sql_content = None
        seed_file = self.config.seed_database_file

        if seed_file:
            sql_path = Path(seed_file).expanduser().resolve()
            if not sql_path.exists():
                raise FileNotFoundError(f"SQL file not found: {sql_path}")
            sql_content = sql_path.read_text(encoding="utf-8")
            self._seed_sql_content = sql_content

        if self.config.seed_mode == "api":
            if sql_content is None:
                sql_content = self.client.get_sample_sql()
                self._seed_sql_content = sql_content
            self.client.seed_database(
                sql_content=sql_content,
                database_id=self.config.database_id,
            )
            self.client.reset(
                database_id=self.config.database_id, sql_content=sql_content
            )
        else:
            self.client.reset(
                database_id=self.config.database_id, sql_content=sql_content
            )

    async def initialize(self) -> None:
        self._prepare_database()
        self.available_tools = self.client.list_tools()
        if self.config.restricted_tools:
            self.available_tools = [
                tool
                for tool in self.available_tools
                if tool["name"] not in self.config.restricted_tools
            ]

    async def execute_benchmark(self) -> Dict[str, Any]:
        await self.initialize()
        runs: List[Dict[str, Any]] = []

        for run_number in range(1, self.config.number_of_runs + 1):
            if run_number > 1 and self.config.reset_database_between_runs:
                self.client.reset(
                    database_id=self.config.database_id,
                    sql_content=self._seed_sql_content,
                )

            run_result = await self.execute_single_run(run_number)
            runs.append(run_result)

        statistics = self._calculate_statistics(runs)
        return {
            "benchmark_config": {
                "execution_mode": self.config.execution_mode,
                "model": f"{self.config.llm_provider}/{self.config.llm_model}",
                "number_of_runs": self.config.number_of_runs,
                "user_prompt": self.config.user_prompt,
                "database_id": self.config.database_id,
                "seed_database_file": self.config.seed_database_file or "",
            },
            "runs": runs,
            "statistics": statistics,
        }

    async def execute_single_run(self, run_number: int) -> Dict[str, Any]:
        start_time = datetime.now(timezone.utc)
        task_result = await self._execute_task()
        verification_results = await self._run_verifiers(task_result)

        execution_time_ms = int(
            (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        )
        overall_success = all(v.get("passed") for v in verification_results.values())

        total_verifiers = len(verification_results)
        passed_verifiers = sum(
            1 for v in verification_results.values() if v.get("passed", False)
        )

        return {
            "run_number": run_number,
            "started_at": start_time.isoformat(),
            "execution_time_ms": execution_time_ms,
            "model_response": task_result.get("final_response"),
            "conversation_flow": task_result.get("conversation_flow", []),
            "tools_used": task_result.get("tools_used", []),
            "tool_results": task_result.get("tool_results", []),
            "verification_results": verification_results,
            "verification_summary": {
                "total": total_verifiers,
                "passed": passed_verifiers,
                "failed": total_verifiers - passed_verifiers,
                "pass_rate": passed_verifiers / total_verifiers
                if total_verifiers > 0
                else 0.0,
            },
            "overall_success": overall_success,
        }

    async def _execute_task(self) -> Dict[str, Any]:
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.config.system_prompt},
            {"role": "user", "content": self.config.user_prompt},
        ]

        conversation_flow: List[Dict[str, Any]] = []
        tools_used: List[str] = []
        tool_results: List[Dict[str, Any]] = []

        for iteration in range(self.config.max_iterations):
            response = await self.llm_client.complete_with_tools(
                messages, self.available_tools
            )
            messages.append(response.to_message_dict())

            tool_calls = [
                {"id": tc.id, "name": tc.name, "args": tc.args}
                for tc in response.tool_calls
            ]
            conversation_flow.append(
                {
                    "type": "ai_message",
                    "content": response.content,
                    "tool_calls": tool_calls,
                }
            )

            if not tool_calls:
                break

            for tool_call in tool_calls:
                tool_name = tool_call.get("name")
                tool_args = tool_call.get("args", {})

                observation = self.client.call_tool(tool_name, arguments=tool_args)
                tool_result = {
                    "success": observation.success,
                    "result": observation.tool_result,
                    "error_message": observation.error_message,
                    "metadata": observation.metadata,
                }

                if tool_name and tool_name not in tools_used:
                    tools_used.append(tool_name)

                tool_results.append(
                    {
                        "tool_name": tool_name,
                        "arguments": tool_args,
                        "result": tool_result,
                    }
                )

                messages.append(
                    {
                        "role": "tool",
                        "content": json.dumps(tool_result.get("result", {})),
                        "tool_call_id": tool_call.get("id", ""),
                    }
                )

                conversation_flow.append(
                    {
                        "type": "tool_result",
                        "tool_name": tool_name,
                        "result": tool_result,
                    }
                )

        # Find the last assistant message (not a tool result).
        final_response = ""
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                final_response = msg.get("content", "")
                break
        return {
            "final_response": final_response,
            "conversation_flow": conversation_flow,
            "tools_used": tools_used,
            "tool_results": tool_results,
        }

    async def _run_verifiers(self, task_result: Dict[str, Any]) -> Dict[str, Any]:
        verification_results: Dict[str, Any] = {}
        model_response = {
            "content": task_result.get("final_response", ""),
            "tool_calls": [
                {"name": tr["tool_name"], "args": tr["arguments"]}
                for tr in task_result.get("tool_results", [])
            ],
        }

        for i, verifier in enumerate(self.config.verifiers):
            verifier_name = verifier.get("name") or f"verifier_{i + 1}"
            verification_results[
                verifier_name
            ] = await self.verifier_engine.execute_verifier(
                verifier, model_response, self.config.database_id, self.config.context
            )

        return verification_results

    def _calculate_statistics(self, runs: List[Dict[str, Any]]) -> Dict[str, Any]:
        successful_runs = [r for r in runs if r.get("overall_success")]
        total_runs = len(runs)
        overall_success_rate = (
            len(successful_runs) / total_runs if total_runs > 0 else 0.0
        )
        pass_at_1 = 1.0 if runs and runs[0].get("overall_success") else 0.0

        total_verifiers_count = 0
        passed_verifiers_count = 0
        verifier_pass_rates: Dict[str, Dict[str, int]] = {}

        for run in runs:
            summary = run.get("verification_summary")
            if summary:
                total_verifiers_count += summary.get("total", 0)
                passed_verifiers_count += summary.get("passed", 0)

            for verifier_name, result in run.get("verification_results", {}).items():
                if verifier_name not in verifier_pass_rates:
                    verifier_pass_rates[verifier_name] = {"passed": 0, "total": 0}
                verifier_pass_rates[verifier_name]["total"] += 1
                if result.get("passed", False):
                    verifier_pass_rates[verifier_name]["passed"] += 1

        verifier_stats = {}
        for verifier_name, counts in verifier_pass_rates.items():
            total = counts["total"]
            passed = counts["passed"]
            verifier_stats[verifier_name] = {
                "passed": passed,
                "total": total,
                "pass_rate": passed / total if total > 0 else 0.0,
            }

        verifier_level_pass_rate = (
            passed_verifiers_count / total_verifiers_count
            if total_verifiers_count > 0
            else 0.0
        )

        execution_times = [
            r.get("execution_time_ms", 0) for r in runs if "execution_time_ms" in r
        ]
        mean_time = (
            sum(execution_times) / len(execution_times) if execution_times else 0.0
        )

        tool_counts: Dict[str, int] = {}
        for run in runs:
            for tool in run.get("tools_used", []):
                tool_counts[tool] = tool_counts.get(tool, 0) + 1

        return {
            "total_runs": total_runs,
            "successful_runs": len(successful_runs),
            "overall_success_rate": overall_success_rate,
            "pass_at_1": pass_at_1,
            "verifier_level_pass_rate": verifier_level_pass_rate,
            "total_verifiers_checked": total_verifiers_count,
            "total_verifiers_passed": passed_verifiers_count,
            "individual_verifier_stats": verifier_stats,
            "mean_execution_time_ms": mean_time,
            "tool_usage": tool_counts,
        }


def _resolve_api_key(provider: str, config_key: Optional[str]) -> str:
    if config_key:
        return config_key

    provider = provider.lower()
    env_map = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "google": "GOOGLE_API_KEY",
    }
    key = os.getenv(env_map.get(provider, "LLM_API_KEY")) or os.getenv("LLM_API_KEY")
    if not key:
        raise ValueError("LLM API key is required for scenario runs")
    return key


def _load_scenario_config(
    config_path: str,
    base_url_override: Optional[str],
    seed_sql_override: Optional[str],
    database_id_override: Optional[str],
    auto_db: bool,
    seed_mode_override: Optional[str],
    output_dir: Optional[str],
    access_token_override: Optional[str],
) -> ScenarioConfig:
    path = Path(config_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Scenario config not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        config_data = json.load(file)

    config_data = {k: v for k, v in config_data.items() if not str(k).startswith("_")}

    base_url = (
        base_url_override
        or config_data.get("gym_enviornment_url")
        or config_data.get("gym_environment_url")
        or config_data.get("base_url")
    )
    if not base_url:
        raise ValueError("Missing gym_enviornment_url in scenario config")

    seed_database_file = (
        seed_sql_override
        if seed_sql_override is not None
        else config_data.get("seed_database_file", "")
    )
    seed_mode = seed_mode_override or config_data.get("seed_mode", "reset")
    if seed_mode not in {"reset", "api"}:
        raise ValueError(f"Invalid seed_mode: {seed_mode}")

    required_fields = ["system_prompt", "user_prompt", "llm_provider", "llm_model"]
    missing = [field for field in required_fields if not config_data.get(field)]
    if missing:
        raise ValueError(
            f"Missing required fields in scenario config: {', '.join(missing)}"
        )

    execution_mode = config_data.get("execution_mode", "openenv")
    if execution_mode != "openenv":
        raise ValueError("Only execution_mode 'openenv' is supported by this client")

    database_id = database_id_override or config_data.get("database_id")
    if auto_db:
        database_id = _generate_database_id()
    if not database_id:
        database_id = "default"

    access_token = access_token_override or config_data.get("access_token")

    llm_provider = config_data.get("llm_provider")
    llm_model = config_data.get("llm_model")
    llm_api_key = _resolve_api_key(llm_provider, config_data.get("llm_api_key"))

    output_path = (
        Path(output_dir).expanduser().resolve() if output_dir else DEFAULT_OUTPUT_DIR
    )
    if not output_path.is_absolute():
        output_path = Path(__file__).resolve().parent / output_path

    return ScenarioConfig(
        gym_enviornment_url=base_url,
        seed_database_file=seed_database_file or "",
        system_prompt=config_data.get("system_prompt", ""),
        user_prompt=config_data.get("user_prompt", ""),
        llm_model=llm_model,
        llm_provider=llm_provider,
        llm_api_key=llm_api_key,
        database_id=database_id,
        access_token=access_token,
        context=config_data.get("context", {}),
        execution_mode=execution_mode,
        expected_tools=config_data.get("expected_tools", []) or [],
        restricted_tools=config_data.get("restricted_tools", []) or [],
        verifiers=config_data.get("verifiers", []) or [],
        number_of_runs=config_data.get("number_of_runs", 1),
        reset_database_between_runs=config_data.get(
            "reset_database_between_runs", True
        ),
        temperature=config_data.get("temperature", 0.0),
        max_tokens=config_data.get("max_tokens", 4096),
        max_iterations=config_data.get("max_iterations", 20),
        seed_mode=seed_mode,
        output_dir=output_path,
    )


def _write_scenario_output(result: Dict[str, Any], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = (
        f"benchmark_results_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    )
    output_path = output_dir / filename
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2, default=str)
    return output_path


def _parse_json_arg(value: str) -> Dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON value: {value}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("JSON value must be an object")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Calendar environment HTTP client")
    parser.add_argument("--base-url", default="http://localhost:8004")
    parser.add_argument("--database-id", default="default")
    parser.add_argument("--auto-db", action="store_true")
    parser.add_argument("--access-token", default=None)
    parser.add_argument("--context", default="{}")
    parser.add_argument("--seed-sql", default=None)
    parser.add_argument("--seed-mode", choices=["reset", "api"], default=None)
    parser.add_argument("--tool", default=None)
    parser.add_argument("--args", default="{}")
    parser.add_argument("--scenario", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    context = _parse_json_arg(args.context)

    if args.scenario:
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )
        config = _load_scenario_config(
            config_path=args.scenario,
            base_url_override=args.base_url,
            seed_sql_override=args.seed_sql,
            database_id_override=args.database_id,
            auto_db=args.auto_db,
            seed_mode_override=args.seed_mode,
            output_dir=args.output_dir,
            access_token_override=args.access_token,
        )

        async def _run() -> None:
            runner = ScenarioRunner(config)
            result = await runner.execute_benchmark()
            output_path = _write_scenario_output(result, config.output_dir)
            logger.info("Scenario results saved to %s", output_path)

        asyncio.run(_run())
        return

    database_id = _generate_database_id() if args.auto_db else args.database_id

    with CalendarEnv(
        base_url=args.base_url,
        database_id=database_id,
        access_token=args.access_token,
        context=context,
    ) as client:
        seed_mode = args.seed_mode or "reset"
        if args.seed_sql:
            if seed_mode == "api":
                client.seed_database_from_file(
                    sql_file_path=args.seed_sql,
                    database_id=database_id,
                )
                client.reset_with_sql_file(
                    sql_file_path=args.seed_sql,
                    database_id=database_id,
                )
            else:
                client.reset_with_sql_file(
                    sql_file_path=args.seed_sql,
                    database_id=database_id,
                )
        else:
            client.reset(database_id=database_id)

        tools = client.list_tools()
        print(f"tools: {len(tools)}")

        if args.tool:
            tool_args = _parse_json_arg(args.args)
            observation = client.call_tool(args.tool, arguments=tool_args)
            print(observation.model_dump())


if __name__ == "__main__":
    main()
