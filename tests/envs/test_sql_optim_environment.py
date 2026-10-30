# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests for the SQL Query Optimization environment.

Covers the server-side environment (reset/step/state driven against real
DuckDB execution), the wire models, and the client's payload/parse helpers.
The client's transport is exercised separately by the WebSocket integration
tests; here we unit-test the pure parsing methods without a live server.
"""

import os
import sys
import time
import tracemalloc

import pytest

# Add the repo root so `envs.*` imports resolve.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from envs.sql_optim_env import (  # noqa: E402
    SQLOptimAction,
    SQLOptimEnv,
    SQLOptimObservation,
    SQLOptimState,
)
from envs.sql_optim_env.server.sql_optim_environment import (  # noqa: E402
    DEFAULT_TASK_ID,
    SQLOptimEnvironment,
)
from envs.sql_optim_env.server.tasks import TASKS  # noqa: E402


@pytest.fixture(scope="module")
def env():
    """A single environment instance (DuckDB tables are built once)."""
    return SQLOptimEnvironment()


@pytest.fixture(scope="module")
def executor():
    """The shared DuckDB query executor (tables built once)."""
    from envs.sql_optim_env.server.executor import get_executor

    return get_executor()


def _good_action() -> SQLOptimAction:
    """A result-preserving task-1 rewrite: sargable predicates, projection kept."""
    return SQLOptimAction(
        suggestions=[
            {
                "issue_type": "select_star",
                "line": 1,
                "description": "SELECT * is wasteful",
                "severity": "high",
                "fix": "project columns",
            },
            {
                "issue_type": "non_sargable_cast",
                "line": 3,
                "description": "CAST blocks pruning",
                "severity": "high",
                "fix": "compare int",
            },
            {
                "issue_type": "function_on_date_column",
                "line": 4,
                "description": "year() non-sargable",
                "severity": "medium",
                "fix": "date range",
            },
        ],
        optimized_query=(
            "SELECT * FROM orders WHERE customer_id = 5000 "
            "AND created_at >= DATE '2024-01-01' AND created_at < DATE '2025-01-01';"
        ),
        summary="Made the CAST and year() predicates sargable; flagged SELECT *.",
        estimated_improvement="~10x faster",
        approved=False,
    )


class TestReset:
    def test_reset_returns_task_observation(self, env):
        obs = env.reset(task_id="task_1_basic_antipatterns")
        assert isinstance(obs, SQLOptimObservation)
        assert obs.task_id == "task_1_basic_antipatterns"
        assert obs.step_count == 0
        assert obs.reward is None
        assert obs.done is False
        assert obs.sql_query  # a query is presented

    def test_reset_defaults_to_first_task(self, env):
        obs = env.reset()
        assert obs.task_id == DEFAULT_TASK_ID

    def test_reset_unknown_task_raises(self, env):
        with pytest.raises(ValueError, match="Unknown task_id"):
            env.reset(task_id="does_not_exist")


class TestStep:
    def test_step_rewards_a_faster_correct_rewrite(self, env):
        env.reset(task_id="task_1_basic_antipatterns")
        obs = env.step(_good_action())
        assert isinstance(obs.reward, float)
        assert obs.reward > 0.0
        assert "reward_breakdown" in obs.metadata
        assert obs.metadata["execution"]["speedup"] > 1.0  # real DuckDB speedup

    def test_step_accumulates_issue_context(self, env):
        env.reset(task_id="task_1_basic_antipatterns")
        obs = env.step(_good_action())
        assert set(obs.issues_found_so_far) >= {"select_star", "non_sargable_cast"}

    def test_episode_terminates_at_max_steps(self, env):
        env.reset(task_id="task_1_basic_antipatterns")
        max_steps = TASKS["task_1_basic_antipatterns"]["max_steps"]
        empty = SQLOptimAction(optimized_query="")  # no reward, forces max_steps
        done = False
        for _ in range(max_steps):
            done = env.step(empty).done
        assert done is True

    def test_step_after_done_raises(self, env):
        env.reset(task_id="task_1_basic_antipatterns")
        max_steps = TASKS["task_1_basic_antipatterns"]["max_steps"]
        empty = SQLOptimAction(optimized_query="")
        for _ in range(max_steps):
            env.step(empty)
        with pytest.raises(RuntimeError, match="Episode finished"):
            env.step(empty)


class TestState:
    def test_state_tracks_episode(self, env):
        env.reset(task_id="task_2_correlated_subqueries")
        state = env.state
        assert isinstance(state, SQLOptimState)
        assert state.step_count == 0
        assert state.episode_done is False
        assert state.cumulative_reward == 0.0


class TestModels:
    def test_action_roundtrips(self):
        action = _good_action()
        dumped = action.model_dump()
        assert dumped["optimized_query"].startswith("SELECT *")
        assert SQLOptimAction(**dumped) == action

    def test_observation_carries_base_fields(self):
        obs = SQLOptimObservation(task_id="t", reward=0.5, done=True)
        assert obs.reward == 0.5 and obs.done is True


class TestClientParsing:
    """The client's pure helpers, without opening a WebSocket."""

    def _client(self):
        return SQLOptimEnv(base_url="http://localhost:8000")

    def test_step_payload(self):
        payload = self._client()._step_payload(_good_action())
        assert payload["optimized_query"].startswith("SELECT *")
        assert payload["approved"] is False

    def test_parse_result(self):
        client = self._client()
        result = client._parse_result(
            {
                "observation": {"task_id": "t", "sql_query": "SELECT 1"},
                "reward": 0.42,
                "done": True,
                "metadata": {"feedback": "ok"},
            }
        )
        assert isinstance(result.observation, SQLOptimObservation)
        assert result.reward == 0.42
        assert result.done is True
        assert result.metadata == {"feedback": "ok"}

    def test_parse_state(self):
        state = self._client()._parse_state({"task_id": "t", "step_count": 3})
        assert isinstance(state, SQLOptimState)
        assert state.task_id == "t" and state.step_count == 3


class TestExecutionSafetyAndCorrectness:
    """Regression tests for the execution-engine review findings.

    Cover DB isolation, correctness-check robustness, single-compare-per-step, stale
    execution on empty rewrites, and the task-definition corrections.
    """

    def test_mutating_rewrite_is_rejected_and_db_intact(self, executor):
        """A DROP/DELETE rewrite is rejected and never mutates the shared DB."""
        before = executor.table_stats["orders"]
        for bad in (
            "DROP TABLE orders",
            "DELETE FROM orders",
            "SELECT 1; DROP TABLE orders",
        ):
            res = executor.compare("SELECT 1", bad)
            assert res["optimized_error"], f"not rejected: {bad}"
            assert res["results_match"] is False
        assert executor.table_stats["orders"] == before  # nothing persisted

    def test_read_only_rewrite_still_runs(self, executor):
        """A legitimate SELECT rewrite executes normally."""
        res = executor.compare(
            "SELECT COUNT(*) FROM orders", "SELECT COUNT(*) FROM orders"
        )
        assert res["optimized_error"] is None
        assert res["results_match"] is True

    def test_trailing_semicolon_large_result_checksum(self, executor):
        """A `;`-terminated query over >50k rows is still compared correctly."""
        q = "SELECT * FROM events;"  # 1M rows, trailing ';'
        res = executor.compare(q, q)
        assert res["results_match"] is True
        assert res["original_rows"] == 1_000_000

    def test_task3_equality_rewrite_is_result_preserving(self, executor):
        """Task 3's intended equality rewrite returns exactly the same rows.

        (Speedup is not asserted — wall-clock timing is too noisy for a stable
        threshold; result-preservation is the property that matters here.)
        """
        original = TASKS["task_3_wildcard_scan"]["sql_query"]
        rewrite = original.replace(
            "WHERE event_type LIKE '%purchase%'\n   OR event_type LIKE '%buy%';",
            "WHERE event_type = 'purchase';",
        )
        assert rewrite != original  # the replacement actually happened
        res = executor.compare(original, rewrite)
        assert res["results_match"] is True
        assert res["optimized_error"] is None

    def test_task5_is_deterministic(self, executor):
        """Task 5's windows are deterministic, so a result-preserving rewrite can match."""
        q = TASKS["task_5_window_functions"]["sql_query"]
        assert executor.compare(q, q)["results_match"] is True

    def test_task4_subqueries_labeled_uncorrelated(self):
        """Task 4 no longer mislabels its uncorrelated scalar subqueries as correlated."""
        types = [
            g["type"] for g in TASKS["task_4_implicit_join"]["ground_truth_issues"]
        ]
        assert "uncorrelated_scalar_subquery_avg" in types
        assert "uncorrelated_scalar_subquery_max" in types
        blob = str(TASKS["task_4_implicit_join"]).lower()
        assert "uncorrelated" in blob
        assert "per group" not in blob  # the "recomputed per group" mislabel is gone

    def test_step_runs_compare_once(self, monkeypatch):
        """`step` reuses the grader's comparison instead of running DuckDB twice."""
        from envs.sql_optim_env.server import executor as ex_mod

        ex = ex_mod.get_executor()
        real = ex.compare
        calls = {"n": 0}

        def counting(*a, **k):
            calls["n"] += 1
            return real(*a, **k)

        monkeypatch.setattr(ex, "compare", counting)
        e = SQLOptimEnvironment()
        e.reset(task_id="task_1_basic_antipatterns")
        e.step(_good_action())
        assert calls["n"] == 1  # was 2 before the dedup fix

    def test_empty_rewrite_does_not_report_stale_execution(self):
        """An empty rewrite clears `execution` instead of keeping the prior step's."""
        e = SQLOptimEnvironment()
        e.reset(task_id="task_1_basic_antipatterns")
        obs1 = e.step(
            SQLOptimAction(
                suggestions=[{"issue_type": "select_star", "severity": "low"}],
                optimized_query="SELECT id FROM orders WHERE customer_id = 5000",
                summary="partial",
                approved=False,
            )
        )
        assert obs1.metadata["execution"] is not None
        assert obs1.done is False
        obs2 = e.step(
            SQLOptimAction(optimized_query="", summary="no rewrite", approved=False)
        )
        assert obs2.metadata["execution"] is None  # not the stale step-1 comparison

    def test_valid_rewrites_run_including_comments_and_literals(self, executor):
        """Valid read-only rewrites execute — no structural pre-check false-rejects.

        The read-only connection is the only gate, so rewrites using function
        names (`REPLACE`), string literals containing `set`/`;`, or a leading SQL
        comment all run instead of being rejected up front.
        """
        for good in (
            "SELECT REPLACE(status, 'a', 'b') FROM orders LIMIT 1",
            "SELECT COUNT(*) FROM orders WHERE status = 'set'",
            "SELECT COUNT(*) FROM orders WHERE status = 'a;b'",
            "-- optimized\nSELECT COUNT(*) FROM orders",
        ):
            res = executor.compare("SELECT COUNT(*) FROM orders", good)
            assert res["optimized_error"] is None, f"falsely rejected: {good!r}"

    def test_issue_detection_ignores_the_rewrite_sql(self):
        """Echoing the slow SQL earns no issue-detection credit without analysis.

        Anti-pattern tokens live in the query itself; scoring detection off the
        rewrite would let an agent farm the slice by copying the original query.
        """
        from envs.sql_optim_env.server.graders import grade

        original = TASKS["task_1_basic_antipatterns"]["sql_query"]
        # Rewrite echoes the original SQL, but the analysis fields are empty.
        action = SQLOptimAction(
            suggestions=[], optimized_query=original, summary="", approved=False
        )
        reward = grade(TASKS["task_1_basic_antipatterns"], action)
        assert reward.breakdown["issue_detection"] == 0.0

    def test_read_only_connection_rejects_writes_at_engine_level(self, executor):
        """Even a DML-in-CTE form is rejected — the connection is truly read-only."""
        before = executor.table_stats["orders"]
        res = executor.compare("SELECT 1", "WITH d AS (SELECT 1) DELETE FROM orders")
        assert res["optimized_error"]  # engine (or structural gate) rejects it
        assert executor.table_stats["orders"] == before

    def test_task1_result_preserving_rewrite_matches(self, executor):
        """Task 1's intended rewrite (sargable predicates, SELECT * kept) matches."""
        original = TASKS["task_1_basic_antipatterns"]["sql_query"]
        rewrite = (
            "SELECT * FROM orders WHERE customer_id = 5000 "
            "AND created_at >= DATE '2024-01-01' AND created_at < DATE '2025-01-01'"
        )
        res = executor.compare(original, rewrite)
        assert res["results_match"] is True

    def test_task2_last_order_is_deterministic(self, executor):
        """Task 2's `last_order_amount` subquery has a unique tie-breaker."""
        q = TASKS["task_2_correlated_subqueries"]["sql_query"]
        assert executor.compare(q, q)["results_match"] is True

    def test_speedup_requires_correctness(self):
        """A fast but wrong rewrite earns no speedup credit (only when results match)."""
        from envs.sql_optim_env.server.graders import grade

        task = TASKS["task_5_window_functions"]
        # Trivially fast, but wrong results.
        wrong = SQLOptimAction(optimized_query="SELECT 1", summary="x" * 60)
        r_wrong = grade(task, wrong)
        assert r_wrong.breakdown["execution_speedup"] == 0.0
        assert r_wrong.execution["results_match"] is False

    def test_concurrent_compare_is_safe(self, executor):
        """Concurrent access to the shared connection does not corrupt or crash."""
        import threading

        errors = []

        def worker():
            try:
                for _ in range(3):
                    executor.compare(
                        "SELECT COUNT(*) FROM orders",
                        "SELECT COUNT(*) FROM orders WHERE status = 'completed'",
                    )
                    assert executor.table_stats["orders"] == 500_000
            except Exception as exc:  # pragma: no cover - failure path
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors


class TestSandboxAndCorrectnessCredit:
    """The agent-authored query cannot reach the filesystem or fake a match."""

    def test_agent_sql_cannot_touch_the_filesystem(self, executor, tmp_path):
        """`read_only` alone left DuckDB's file/extension escape hatches open."""
        written = tmp_path / "exfil.csv"
        escapes = [
            # Writes
            f"COPY (SELECT * FROM users) TO '{written}'",
            f"COPY (SELECT 1) TO '{written}' (FORMAT CSV)",
            f"EXPORT DATABASE '{tmp_path / 'dump'}'",
            # Reads
            "SELECT * FROM read_csv('/etc/passwd', header=false, "
            "columns={'line': 'VARCHAR'})",
            "SELECT * FROM read_parquet('/etc/passwd')",
            "SELECT * FROM read_text('/etc/hostname')",
            "SELECT * FROM read_blob('/etc/hostname')",
            "SELECT * FROM glob('/*')",
            "COPY users FROM '/etc/passwd'",
            # Mounting and extensions
            f"ATTACH '{tmp_path / 'side.db'}' AS side",
            "INSTALL httpfs",
            "LOAD '/tmp/evil.duckdb_extension'",
        ]
        for escape in escapes:
            res = executor.compare("SELECT COUNT(*) FROM users", escape)
            assert res["optimized_error"], f"not blocked: {escape}"
            assert res["results_match"] is False
        assert not written.exists()  # nothing was exfiltrated
        assert not (tmp_path / "dump").exists()

    def test_sandbox_also_holds_on_the_timing_path(self, executor, tmp_path):
        """`_engine_ms` re-runs the query under `EXPLAIN ANALYZE` — still sandboxed."""
        written = tmp_path / "profiled.csv"
        for escape in (
            f"COPY (SELECT 1) TO '{written}'",
            "SELECT * FROM read_text('/etc/hostname')",
        ):
            with pytest.raises(Exception):
                executor.conn.execute(f"EXPLAIN ANALYZE {escape}").fetchall()
        assert not written.exists()

    def test_agent_sql_cannot_re_enable_external_access(self, executor):
        """A rewrite cannot `SET`/`PRAGMA` its way back out of the sandbox."""
        for escape in (
            "SET enable_external_access=true",
            "SET GLOBAL enable_external_access=true",
            "SET extension_directory='/tmp/pwn'",
            "PRAGMA enable_external_access=true",
            "PRAGMA lock_configuration=false",
        ):
            res = executor.compare("SELECT 1", escape)
            assert res["optimized_error"], f"not blocked: {escape}"
            assert res["results_match"] is False
        # Still sandboxed afterwards.
        assert executor.compare("SELECT 1", "SELECT * FROM read_text('/etc/hostname')")[
            "optimized_error"
        ]

    def test_trailing_comment_does_not_break_the_large_result_checksum(self, executor):
        """A rewrite ending in `--` still gets a real correctness verdict.

        The correctness check wraps the query; on one line, a trailing comment
        swallowed the closing paren. When the old checksum failed that way,
        equal row counts were reported as a match — full correctness credit for
        a rewrite returning different rows on every one of a million of them.
        """
        differing = executor.compare(
            "SELECT id FROM events -- keep",
            "SELECT id + 1 AS id FROM events -- keep",
        )
        assert differing["original_rows"] == differing["optimized_rows"] == 1_000_000
        assert differing["results_match"] is False  # was True

        identical = executor.compare(
            "SELECT id FROM events -- keep", "SELECT id FROM events -- keep"
        )
        assert identical["results_match"] is True

    def test_unverifiable_large_result_earns_no_correctness_credit(
        self, executor, monkeypatch
    ):
        """A comparison that errors is a mismatch, and is not retried another way.

        Equal row counts are not evidence of equal data, so a failed comparison
        must not fall back to them, or to any other strategy.
        """
        from envs.sql_optim_env.server import executor as ex_mod

        calls = []

        def broken(original, optimized):
            calls.append((original, optimized))
            return "SELECT * FRM nowhere"

        monkeypatch.setattr(ex_mod, "_symmetric_difference_sql", broken)
        res = executor.compare("SELECT id FROM events", "SELECT id FROM events")
        assert res["original_rows"] == res["optimized_rows"] == 1_000_000
        assert res["original_error"] is None and res["optimized_error"] is None
        assert res["results_match"] is False
        assert len(calls) == 1

        monkeypatch.undo()  # the executor fixture is module-scoped: restore it
        res = executor.compare("SELECT id FROM events", "SELECT id FROM events")
        assert res["results_match"] is True


class TestResourceLimits:
    """A runaway rewrite cannot wedge the shared connection or the server.

    The agent chooses the SQL, and `FROM events a, events b` is a trillion-row
    cross join. DuckDB has no statement timeout, so the executor arms its own
    deadline and caps memory and spill space.
    """

    RUNAWAY = "SELECT COUNT(*) FROM events a, events b"

    @pytest.fixture
    def short_deadline(self, monkeypatch):
        """Shrink the deadline so these tests take a second, not fifteen."""
        from envs.sql_optim_env.server import executor as ex_mod

        monkeypatch.setattr(ex_mod, "_QUERY_TIMEOUT_S", 1.0, raising=False)

    def test_runaway_query_is_cancelled_and_reported(self, executor, short_deadline):
        """It comes back as an error instead of running for hours."""
        start = time.perf_counter()
        res = executor.compare("SELECT COUNT(*) FROM orders", self.RUNAWAY)
        elapsed = time.perf_counter() - start

        assert elapsed < 30  # generous; unbounded before, ~1s after
        assert "Query cancelled" in (res["optimized_error"] or "")
        assert res["results_match"] is False

        # The connection survives the cancellation.
        assert executor.table_stats["orders"] == 500_000
        assert executor.compare("SELECT 1", "SELECT 1")["results_match"] is True

    def test_timeout_is_not_defeated_by_the_row_count_fallback(
        self, executor, short_deadline
    ):
        """A cancelled count must not fall through to an unbounded drain.

        `_row_count` retries a streaming drain when the subquery wrap fails.
        Treating a timeout as that kind of failure restarted the same runaway
        query with the timer already spent, so the deadline bought nothing.
        """
        start = time.perf_counter()
        count, error = executor._row_count(self.RUNAWAY)
        elapsed = time.perf_counter() - start

        assert count is None
        assert "Query cancelled" in (error or "")
        assert elapsed < 30

    def test_deadline_releases_the_shared_execution_lock(
        self, executor, short_deadline
    ):
        """One session's runaway query cannot block the others indefinitely."""
        import threading

        waited = {}

        def hog():
            executor._run(self.RUNAWAY)

        def waiter():
            start = time.perf_counter()
            executor._run("SELECT 1")
            waited["s"] = time.perf_counter() - start

        hogger = threading.Thread(target=hog, daemon=True)
        hogger.start()
        second = threading.Thread(target=waiter, daemon=True)
        second.start()
        second.join(60)

        assert "s" in waited, "second session never got the lock"
        assert waited["s"] < 30
        hogger.join(30)

    def test_explain_returns_a_plan_and_survives_the_deadline(
        self, executor, short_deadline
    ):
        """`explain()` still returns its plan; a timeout is reported, not raised.

        Nothing else in the suite covered this method, which is how a dead
        `return` slipped past a full green run.
        """
        plan = executor.explain("SELECT COUNT(*) FROM orders;")
        assert isinstance(plan, str)
        assert plan and "EXPLAIN error" not in plan

        bad = executor.explain("SELECT * FRM orders")
        assert bad.startswith("EXPLAIN error:")

    def test_table_stats_are_unaffected_by_the_deadline(self, executor):
        """The trusted stats queries keep working (and are not timed out)."""
        assert executor.table_stats == {
            "users": 10_000,
            "orders": 500_000,
            "products": 1_000,
            "events": 1_000_000,
        }

    def test_memory_and_spill_leave_runtime_headroom(self, executor):
        """Include tmpfs-backed spill and the real seed DB in the memory budget."""
        from pathlib import Path

        import yaml

        manifest = (
            Path(__file__).resolve().parents[2] / "envs/sql_optim_env/openenv.yaml"
        )
        resources = yaml.safe_load(manifest.read_text())["validation"]["resources"]
        mib = 1024 * 1024
        limits = {}
        for option in ("memory_limit", "max_temp_directory_size"):
            with executor._exec_lock:
                value = executor.conn.execute(
                    f"SELECT current_setting('{option}')"
                ).fetchone()[0]
            amount, unit = value.split()
            limits[option] = float(amount) * {"MiB": mib, "GiB": 1024 * mib}[unit]

        # docker-local puts /tmp on tmpfs and reserves up to 64 MiB for /dev/shm.
        database_bytes = sum(
            p.stat().st_size for p in Path(executor._dir).rglob("*") if p.is_file()
        )
        disk_bytes = resources["disk_mb"] * mib
        shm_bytes = min(64 * mib, disk_bytes // 2)
        writable_bytes = limits["max_temp_directory_size"] + database_bytes + shm_bytes
        assert database_bytes > 0
        assert writable_bytes <= disk_bytes / 2
        # Reserve at least half for Python, unmanaged DuckDB allocations and server.
        assert (
            limits["memory_limit"] + writable_bytes <= resources["memory_mb"] * mib / 2
        )


class TestLargeResultMeasurement:
    """Large result sets are measured without being pulled into Python.

    Timing comes from DuckDB's profiler and correctness from the in-engine
    comparison, so a million-row query neither buys a million Python tuples nor
    reports the cost of building them as its execution time.
    """

    def test_large_result_is_not_materialized(self, executor):
        """A 1M-row query yields an exact count and no materialized rows."""
        ms, count, rows, err = executor._run("SELECT * FROM events")
        assert err is None
        assert rows is None  # never held in Python
        assert count == 1_000_000  # still exact, computed inside DuckDB
        assert ms < 90_000  # a real measurement, not the error sentinel

    def test_small_result_is_still_compared_row_by_row(self, executor):
        """Below the cap, rows are materialized and still compared precisely."""
        ms, count, rows, err = executor._run("SELECT * FROM products")
        assert err is None
        assert count == 1_000
        assert rows is not None and len(rows) == 1_000

        # Precise comparison, not count-only: same rows in a different order
        # match, while an equal-sized result with different values does not.
        reordered = executor.compare(
            "SELECT id, price FROM products ORDER BY id",
            "SELECT id, price FROM products ORDER BY id DESC",
        )
        assert reordered["results_match"] is True
        altered = executor.compare(
            "SELECT id, price FROM products",
            "SELECT id, price + 1 AS price FROM products",
        )
        assert altered["results_match"] is False
        assert altered["original_rows"] == altered["optimized_rows"] == 1_000

    def test_timing_excludes_python_materialization(self, executor):
        """The reported time tracks DuckDB execution, not tuple construction."""
        query = "SELECT * FROM events"
        start = time.perf_counter()
        with executor._exec_lock:
            fetched = executor.conn.execute(query).fetchall()
        fetchall_ms = (time.perf_counter() - start) * 1000.0
        assert len(fetched) == 1_000_000
        del fetched

        reported_ms = executor._run(query)[0]
        # Scanning the table costs a few percent of what turning it into a
        # million Python tuples does, so this bound has an order of magnitude of
        # headroom for a slow CI machine while still failing outright if
        # `fetchall()` timing comes back.
        assert reported_ms < fetchall_ms * 0.25

    def test_peak_memory_is_bounded_for_large_results(self, executor):
        """Comparing a 500k-row query does not buffer the whole result set."""
        tracemalloc.start()
        try:
            result = executor.compare("SELECT * FROM orders", "SELECT * FROM orders")
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()

        assert result["results_match"] is True
        assert result["original_rows"] == 500_000
        # Materializing 500k rows twice cost ~430 MB before; the cap keeps this
        # to a couple of batches regardless of result size.
        assert peak < 100 * 1024 * 1024

    def test_timing_is_floored_at_the_profiler_resolution(self, executor):
        """A query faster than the profiler can express still gets a usable time.

        `EXPLAIN ANALYZE` prints seconds to four decimals, so it reports 0.0000s
        for a query answered from table metadata. Guards the invariant rather
        than reproducing a past failure: every reported time stays at or above
        the floor, so the speedup ratio can never divide by zero.
        """
        from envs.sql_optim_env.server.executor import _MIN_MEASURABLE_MS

        for query in (
            "SELECT COUNT(*) FROM events",  # answered from metadata
            "SELECT 1",
            "SELECT * FROM products WHERE id = -1",  # empty result
        ):
            reported_ms = executor._run(query)[0]
            assert reported_ms >= _MIN_MEASURABLE_MS, query
            result = executor.compare(query, query)
            assert result["optimized_ms"] >= _MIN_MEASURABLE_MS
            assert result["speedup"] > 0
            assert result["results_match"] is True


class TestSessionIsolation:
    """Agent SQL must not leave objects behind on the executor's connection.

    A read-only DuckDB connection rejects persistent writes, but it still
    accepts session-scoped DDL: `CREATE TEMP TABLE users AS ...` succeeds and
    shadows the base table for every later query on that same connection. On a
    long-lived server that poisons all subsequent episodes, and it is directly
    reward-hacking: shrink `orders` to one row and the "optimized" rewrite is
    both instant and, because the original runs against the same shadowed
    table, still "correct".
    """

    def test_temp_table_cannot_shadow_a_base_table(self, executor):
        baseline = executor.table_stats["users"]

        executor.compare(
            "SELECT COUNT(*) FROM users",
            "CREATE TEMP TABLE users AS SELECT 1 AS id, 'HIJACKED' AS name",
        )

        assert executor.table_stats["users"] == baseline
        rows, error = executor._probe("SELECT COUNT(*) FROM users")
        assert error is None
        assert rows[0][0] == baseline

    def test_a_shadowed_table_cannot_fake_a_later_episode(self, executor):
        """The reward-hacking path: poison once, then grade a later rewrite.

        The row count is hardcoded rather than read back from the executor: a
        leaked shadow table would corrupt that reading too, and the assertion
        would pass against the poisoned value.
        """
        true_rows = 10_000  # `users` seed size, see `_build_tables`

        executor.compare(
            "SELECT COUNT(*) FROM users",
            "CREATE TEMP TABLE users AS SELECT 1 AS id, 'HIJACKED' AS name",
        )
        result = executor.compare("SELECT * FROM users", "SELECT * FROM users")

        assert result["original_error"] is None
        assert result["original_rows"] == true_rows
        assert result["optimized_rows"] == true_rows

    def test_ddl_submitted_as_a_rewrite_is_never_credited(self, executor):
        """DDL is not an error any more, and does not need to be.

        Before sessions were isolated, DDL surfaced as an error only by
        accident: the temp table survived the first timing run, so the second
        collided with it. Now each run gets a clean catalog and the statement
        simply succeeds. That is harmless, because the grader gates speedup
        credit on `results_match`, and a `CREATE` returns a row count rather
        than the query's rows.
        """
        info = executor.compare(
            "SELECT COUNT(*) FROM users",
            "CREATE TEMP TABLE scratch_pad AS SELECT 1",
        )
        assert not info["results_match"]

    def test_the_same_temp_name_can_be_reused_across_queries(self, executor):
        """No residue: a name is free again once its session ends."""
        for _ in range(3):
            info = executor.compare(
                "SELECT COUNT(*) FROM users",
                "CREATE TEMP TABLE scratch_pad AS SELECT 1",
            )
            assert info["optimized_error"] is None


class TestSingleStatement:
    """Agent SQL must be one statement, checked by DuckDB's own parser.

    `_engine_ms` prefixes `EXPLAIN ANALYZE ` onto the agent's query. A second
    statement after a `;` survives that prefix, so DuckDB runs the real query
    as its own statement and the plan fetch returns the whole result set
    instead of a plan, bypassing `_MAX_MATERIALIZED_ROWS`. Timing out does not
    help: `interrupt()` cannot stop rows already being converted into Python
    objects, and the lock is held throughout.

    The check uses `duckdb.extract_statements` rather than scanning for `;`,
    so comments and semicolons inside string literals do not false-reject the
    way an earlier keyword pre-check did.
    """

    @pytest.mark.parametrize("rows", [0, 2, 50_002])
    @pytest.mark.parametrize(
        "suffix",
        [
            "; -- trailing comment",
            "; /* trailing block */",
            " /* before */; /* outer /* nested ; */ comment */ -- end",
            "; -- first terminator\n; /* second */;",
        ],
    )
    def test_commented_terminators_preserve_correctness(self, executor, rows, suffix):
        query = f"SELECT i AS x FROM range({rows}) t(i)"
        result = executor.compare(query + suffix, query + suffix)
        assert result["original_error"] is None
        assert result["optimized_error"] is None
        assert result["original_rows"] == result["optimized_rows"] == rows
        assert result["results_match"] is True
        assert executor.compare(query, query + suffix)["results_match"] is True
        assert executor.compare(query, query + " WHERE false" + suffix)[
            "results_match"
        ] is (rows == 0)

    @pytest.mark.parametrize(
        "query",
        [
            "SELECT 'Ä; -- /* quoted */' AS \"é;--\"",
            "SELECT 'it''s; /* quoted */' AS \"a\"\";b\"",
            "SELECT $$; -- dollar quoted /* text */$$ AS x",
            "SELECT $tag$é; -- text$tag$ AS x",
            "/* leading ; */ SELECT 1 /* embedded ; */ AS x -- final ;",
        ],
    )
    def test_terminator_normalization_preserves_sql_text(self, executor, query):
        from envs.sql_optim_env.server.executor import _strip_terminators

        suffix = "\n; /* trailing ; */ -- final ;"
        assert _strip_terminators(query + suffix) == query + suffix.replace(
            "\n;", "\n ", 1
        )
        assert executor.compare(query, query + suffix)["results_match"] is True

    @pytest.mark.parametrize(
        "query",
        [
            "SELECT 1 AS x; /* comment */ SELECT 1 AS x; -- end",
            "SELECT 1 AS x; -- comment\nSELECT 1 AS x; /* end */",
            "SELECT 'unterminated; -- comment",
            "SELECT 1 AS x; /* unterminated",
            "; /* comment only */ -- end",
        ],
    )
    def test_normalization_does_not_hide_invalid_statements(self, executor, query):
        result = executor.compare("SELECT 1 AS x", query)
        assert result["optimized_error"] is not None
        assert result["results_match"] is False

    def test_a_second_statement_is_rejected(self, executor):
        info = executor.compare(
            "SELECT id FROM events WHERE user_id = 42",
            "SELECT 1; SELECT * FROM events",
        )
        assert info["optimized_error"] is not None
        assert not info["results_match"]

    def test_rejection_is_cheap(self, executor):
        """Rejecting must not first materialize the hidden result set."""
        import tracemalloc

        tracemalloc.start()
        executor.compare(
            "SELECT id FROM events WHERE user_id = 42",
            "SELECT 1; SELECT * FROM events",
        )
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        assert peak < 50_000_000, f"peaked at {peak / 1e6:.0f} MB"

    @pytest.mark.parametrize(
        "query",
        [
            "-- leading comment\nSELECT id FROM events WHERE user_id = 42",
            "/* block */ SELECT id FROM events WHERE user_id = 42",
            "SELECT id, ';' AS semi FROM events WHERE user_id = 42",
            "SELECT id FROM events WHERE user_id = 42;",
        ],
    )
    def test_legitimate_rewrites_are_not_rejected(self, executor, query):
        """Comments, string-literal semicolons and a trailing `;` stay valid."""
        info = executor.compare("SELECT id FROM events WHERE user_id = 42", query)

        assert info["optimized_error"] is None


@pytest.mark.parametrize("value", [1, 2])
def test_large_result_checksum_preserves_duplicate_multiplicity(executor, value):
    original = "SELECT 1 AS x FROM range(50002)"
    rewritten = f"SELECT {value} AS x FROM range(50002)"
    result = executor.compare(original, rewritten)
    assert result["original_error"] is None
    assert result["optimized_error"] is None
    assert result["results_match"] is (value == 1)


def test_expired_deadline_finishes_before_releasing_execution_lock(
    executor, monkeypatch
):
    import threading

    from envs.sql_optim_env.server import executor as module

    fired = threading.Event()
    finish_interrupt = threading.Event()
    released = threading.Event()

    def delayed_interrupt():
        fired.set()
        finish_interrupt.wait(5)

    def run():
        with executor._exec_lock, executor._deadline():
            assert fired.wait(2)
        released.set()

    monkeypatch.setattr(module, "_QUERY_TIMEOUT_S", 0.01)
    monkeypatch.setattr(executor, "_interrupt_active", delayed_interrupt)
    worker = threading.Thread(target=run)
    worker.start()
    try:
        assert fired.wait(2)
        assert not released.wait(0.1)
    finally:
        finish_interrupt.set()
        worker.join(2)
    assert released.is_set()


class TestExactComparison:
    """`results_match` follows one exact equality rule at every result size.

    Same output schema by position (column count, ASCII case-insensitive names,
    identical DuckDB types), then the same rows as a multiset under DuckDB's own
    value equality, with no float tolerance. The old comparator compared row
    strings below 50k rows and a row-hash fingerprint above, which disagreed
    with each other: `1 AS x` vs `'1' AS x` was rejected for 2 rows and
    accepted for 50,002, and renaming the column did the reverse.
    """

    RUNAWAY = "SELECT COUNT(*) FROM events a, events b"

    @pytest.fixture
    def deadline(self, monkeypatch):
        """Set the executor deadline to a given number of seconds."""
        from envs.sql_optim_env.server import executor as ex_mod

        def set_deadline(seconds):
            monkeypatch.setattr(ex_mod, "_QUERY_TIMEOUT_S", seconds)

        return set_deadline

    @pytest.mark.parametrize("rows", [2, 50_002])
    @pytest.mark.parametrize(
        "original, rewritten, expected",
        [
            # Lossy coercions a bare set operation would accept.
            ("9007199254740993::BIGINT AS x", "9007199254740992::DOUBLE AS x", False),
            ("true AS x", "1 AS x", False),
            ("1 AS x", "'1' AS x", False),
            # Names: compared by position, ASCII-only case folding.
            ("1 AS x", "1 AS y", False),
            ("1 AS X", "1 AS x", True),
            ('1 AS "Ä"', '1 AS "ä"', False),
            ("1 AS x, 2 AS x", "1 AS x, 2 AS x_1", False),
            ("1 AS x, 2 AS x", "1 AS x, 2 AS x", True),
            ("1 AS x, 2 AS y", "2 AS y, 1 AS x", False),
            # Types: DuckDB's canonical type strings, compared exactly.
            ("1::INT4 AS x", "1::INTEGER AS x", True),
            ("[1]::INT4[] AS x", "[1]::INTEGER[] AS x", True),
            (
                "{'A': 1}::STRUCT(A INTEGER) AS x",
                "{'a': 1}::STRUCT(a INTEGER) AS x",
                False,
            ),
            ("1.5::DECIMAL(5,1) AS x", "1.50::DECIMAL(5,2) AS x", False),
            ("1.5::DECIMAL(5,1) AS x", "1.5::DECIMAL(6,1) AS x", False),
            # Values: engine equality, no float tolerance.
            ("-0.0::DOUBLE AS x", "0.0::DOUBLE AS x", True),
            ("'nan'::DOUBLE AS x", "'nan'::DOUBLE AS x", True),
            ("NULL::INTEGER AS x", "NULL::INTEGER AS x", True),
            ("0.1::DOUBLE + 0.2::DOUBLE AS x", "0.3::DOUBLE AS x", False),
            # Engine equality, not representation identity.
            ("INTERVAL '1 month' AS x", "INTERVAL '30 days' AS x", True),
        ],
    )
    def test_equality_rule_is_the_same_on_both_sides_of_50k(
        self, executor, rows, original, rewritten, expected
    ):
        res = executor.compare(
            f"SELECT {original} FROM range({rows})",
            f"SELECT {rewritten} FROM range({rows})",
        )
        assert res["original_error"] is None and res["optimized_error"] is None
        assert res["original_rows"] == res["optimized_rows"] == rows
        assert res["results_match"] is expected

    @pytest.mark.parametrize(
        "rewritten, expected",
        [
            ("SELECT 1 AS x WHERE false", True),
            ("SELECT 1 AS X WHERE false", True),
            ("SELECT '1' AS x WHERE false", False),
            ("SELECT 1 AS y WHERE false", False),
            ("SELECT 1 AS x, 2 AS y WHERE false", False),
        ],
    )
    def test_empty_results_still_need_matching_schemas(
        self, executor, rewritten, expected
    ):
        res = executor.compare("SELECT 1 AS x WHERE false", rewritten)
        assert res["original_rows"] == res["optimized_rows"] == 0
        assert res["results_match"] is expected

    @pytest.mark.parametrize(
        "original, rewritten, expected",
        [
            (
                "SELECT x FROM (VALUES (1), (1), (2)) t(x)",
                "SELECT x FROM (VALUES (2), (1), (1)) t(x)",
                True,
            ),
            (
                "SELECT x FROM (VALUES (1), (1), (2)) t(x)",
                "SELECT x FROM (VALUES (1), (2), (2)) t(x)",
                False,
            ),
            # Above 50k: the same multiset in a different order, then the same
            # values with one row moved from the 0s to the 1s.
            (
                "SELECT i % 2 AS x FROM range(50002) t(i)",
                "SELECT (i + 1) % 2 AS x FROM range(50002) t(i)",
                True,
            ),
            (
                "SELECT i % 2 AS x FROM range(50002) t(i)",
                "SELECT CASE WHEN i = 0 THEN 1 ELSE i % 2 END AS x "
                "FROM range(50002) t(i)",
                False,
            ),
        ],
    )
    def test_duplicate_multiplicity_counts_and_order_does_not(
        self, executor, original, rewritten, expected
    ):
        assert executor.compare(original, rewritten)["results_match"] is expected

    def test_each_operand_is_materialized_once(self, executor):
        """Both `EXCEPT ALL` arms read one evaluation of each query."""
        from envs.sql_optim_env.server.executor import _symmetric_difference_sql

        sql = _symmetric_difference_sql("SELECT id FROM users", "SELECT id FROM orders")
        with executor._exec_lock, executor._session() as cur:
            plan = "\n".join(
                str(r[1]) for r in cur.execute(f"EXPLAIN {sql}").fetchall()
            )
        # One base-table scan per query, read by two CTE scans each.
        assert plan.count("SEQ_SCAN") == 2
        assert plan.count("CTE_SCAN") == 4

    def test_a_rewrite_cannot_read_the_comparison_ctes(self, executor):
        """Naming the original's CTE fails on its own, so it never matches."""
        rewrite = "SELECT * FROM __sql_optim_original"
        res = executor.compare("SELECT id FROM users", rewrite)
        assert res["optimized_error"] is not None
        assert res["results_match"] is False
        assert executor._results_equal("SELECT id FROM users", rewrite) is False

    def test_multiple_statements_are_rejected_before_comparing(self, executor):
        assert (
            executor._results_equal("SELECT 1 AS x", "SELECT 1 AS x; SELECT 2") is False
        )

    def test_expired_deadline_keeps_interrupting_between_statements(
        self, executor, deadline
    ):
        """An interrupt on an idle connection is not remembered, so it is re-sent.

        With a single timer, a deadline that expired between two statements let
        the second one run unbounded.
        """
        import duckdb

        deadline(0.01)
        start = time.perf_counter()
        with executor._exec_lock, executor._session() as cur:
            with executor._deadline() as expired:
                cur.execute("DESCRIBE SELECT 1").fetchall()
                time.sleep(0.05)  # idle while the deadline passes
                assert expired()
                with pytest.raises(duckdb.InterruptException):
                    cur.execute(self.RUNAWAY).fetchone()
        assert time.perf_counter() - start < 30

    def test_interruption_during_materialization_then_recovery(
        self, executor, deadline
    ):
        """A runaway comparison is cancelled, and the next session works."""
        deadline(1.0)
        start = time.perf_counter()
        assert executor._results_equal(self.RUNAWAY, self.RUNAWAY) is False
        assert time.perf_counter() - start < 30

        assert executor._results_equal("SELECT 1 AS x", "SELECT 1 AS x") is True
        assert executor.table_stats["events"] == 1_000_000

    def test_expiry_between_stages_stops_the_comparison(
        self, executor, deadline, monkeypatch
    ):
        """Schema inspection outlives the deadline: no comparison, no credit."""
        from envs.sql_optim_env.server import executor as ex_mod

        compared = []
        real_schema = ex_mod.QueryExecutor._output_schema

        def slow_schema(cur, query):
            schema = real_schema(cur, query)
            time.sleep(0.3)
            return schema

        def recording(original, optimized):
            compared.append(True)
            return "SELECT 0"

        deadline(0.1)
        monkeypatch.setattr(executor, "_output_schema", slow_schema)
        monkeypatch.setattr(ex_mod, "_symmetric_difference_sql", recording)
        assert executor._results_equal("SELECT 1 AS x", "SELECT 1 AS x") is False
        assert compared == []

    def test_expiry_during_comparison_earns_no_credit(
        self, executor, deadline, monkeypatch
    ):
        """A comparison that completes after the deadline is not trusted."""
        from envs.sql_optim_env.server import executor as ex_mod

        real_sql = ex_mod._symmetric_difference_sql

        def slow_sql(original, optimized):
            time.sleep(0.3)  # the deadline passes after the last pre-stage check
            return real_sql(original, optimized)

        deadline(0.1)
        # No interrupt lands, so the statement itself completes with a match.
        monkeypatch.setattr(executor, "_interrupt_active", lambda: None)
        monkeypatch.setattr(ex_mod, "_symmetric_difference_sql", slow_sql)
        assert executor._results_equal("SELECT 1 AS x", "SELECT 1 AS x") is False

        deadline(15.0)
        monkeypatch.setattr(ex_mod, "_symmetric_difference_sql", real_sql)
        assert executor._results_equal("SELECT 1 AS x", "SELECT 1 AS x") is True
