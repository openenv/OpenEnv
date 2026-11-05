"""
executor.py — DuckDB In-Memory SQL Execution Engine
=====================================================
The core innovation of this environment: instead of keyword-matching
heuristics, we ACTUALLY execute both the original and optimized queries
against realistic synthetic data and measure real performance differences.

Tables populated:
  users    — 10,000 rows
  orders   — 500,000 rows
  products —  1,000 rows
  events   — 1,000,000 rows
"""

import atexit
import contextlib
import os
import re
import shutil
import string
import tempfile
import threading
import time
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import duckdb

_instance: Optional["QueryExecutor"] = None
_lock = threading.Lock()

# Result sets larger than this are never pulled into Python: the row count is
# computed inside DuckDB instead. `SELECT * FROM events` is a million rows, and
# materializing that (three times, for the timing median) both dominated the
# measurement and held the whole result in memory. Correctness does not depend
# on this cap: `compare()` checks every result size inside DuckDB the same way.
_MAX_MATERIALIZED_ROWS = 50_000
# An `EXPLAIN ANALYZE` plan is a couple of rows; this only exists so the plan
# fetch can never become an unbounded materialization of agent-chosen data.
_MAX_PLAN_ROWS = 2_000
_FETCH_BATCH = 10_000
_TIMING_RUNS = 3
# `EXPLAIN ANALYZE` prints seconds with four decimals, so 0.1 ms is the smallest
# value it can express. Used as a floor so a sub-resolution query can never make
# the speedup ratio divide by zero.
_MIN_MEASURABLE_MS = 0.1
_TOTAL_TIME_RE = re.compile(r"Total Time:\s*([0-9]*\.?[0-9]+)\s*s")

# Opening the connection read-only stops the agent from changing the *database*,
# but on its own it leaves DuckDB's filesystem escape hatches open: `COPY ... TO`
# writes arbitrary files, `read_csv`/`read_text`/`glob` read them, `ATTACH`
# mounts other databases and `INSTALL`/`LOAD` pull in extensions. Since the query
# under measurement is agent-authored, external access is disabled outright, and
# the configuration is locked so a rewrite cannot `SET` its way back out (DuckDB
# also refuses to re-enable external access on a running database).
#
# The same reasoning applies to resources. The agent chooses the SQL, so a
# rewrite like `FROM events a, events b` is a trillion-row cross join: without
# a ceiling it can exhaust memory, fill the spill directory, and hold the
# execution lock for hours while every other session waits behind it. DuckDB has
# no statement-timeout option. These caps cover buffer-managed memory and spill
# only; Python conversion and other allocations still need process isolation.
# Keep headroom in the 1024 MiB manifest budget, including RAM-backed /tmp in
# docker-local validation. `_deadline()` interrupts engine work, not Python.
_DUCKDB_CONFIG = {
    "threads": "2",
    "enable_external_access": "false",
    "lock_configuration": "true",
    "memory_limit": "256MiB",
    "max_temp_directory_size": "128MiB",
}
# Generous next to the real tasks (the slowest is ~0.6s) and short enough that a
# runaway rewrite cannot wedge the server.
_QUERY_TIMEOUT_S = 15.0
_TIMEOUT_ERROR = (
    f"Query cancelled: exceeded the {_QUERY_TIMEOUT_S:.0f}s execution limit"
)
# An interrupt sent to an idle connection is not remembered, so once a deadline
# has expired it is re-sent at this interval until the deadline is released.
_INTERRUPT_RETRY_S = 0.01
# Output column names compare case-insensitively the way DuckDB identifiers do,
# which folds ASCII letters only (`X` matches `x`, `Ä` does not match `ä`).
# `str.lower()` and `str.casefold()` fold far more than that.
_ASCII_LOWER = str.maketrans(string.ascii_uppercase, string.ascii_lowercase)


def _strip_terminators(query: str) -> str:
    """Remove trailing terminator tokens without changing comments or quoted SQL.

    DuckDB's tokenizer skips comments and reports UTF-8 byte offsets. Replace
    only trailing semicolon tokens with spaces so queries remain embeddable
    even when comments follow the terminator. Interior statement separators
    stay intact for the single-statement validation to reject.
    """
    query = query.strip()
    try:
        tokens = duckdb.tokenize(query)
    except Exception:
        # Leave malformed SQL for the normal validation/error reporting path.
        return query
    sql = bytearray(query, "utf-8")
    for position, token_type in reversed(tokens):
        if token_type != duckdb.token_type.operator or sql[position] != ord(";"):
            break
        sql[position] = ord(" ")
    return sql.decode("utf-8")


def _as_subquery(query: str, select: str) -> str:
    """Build ``SELECT <select> FROM (<query>) t``.

    The query goes on its own line: a rewrite ending in a ``--`` comment would
    otherwise swallow the closing parenthesis and turn a valid query into a
    parse error, which used to cost the checksum its result.
    """
    return f"SELECT {select} FROM (\n{query}\n) t"


def _symmetric_difference_sql(original: str, optimized: str) -> str:
    """Build a query counting the rows that differ between two queries.

    Rows are compared as a multiset: order is ignored, duplicate multiplicity
    counts. Each operand is a ``MATERIALIZED`` CTE, so it is evaluated once and
    both ``EXCEPT ALL`` arms read the same result. The CTE renames duplicate
    output columns (``x, x`` becomes ``x, x_1``), but set operations match
    columns by position, so that does not change the outcome.

    Set operations do cast mismatched column types to a common type, which
    would make ``true`` equal ``1``, so callers must first check that both
    output schemas are identical. Each query sits on its own lines so that a
    trailing ``--`` comment cannot swallow the wrapper. The CTE names are not
    visible to a query run on its own, so a rewrite that refers to them fails
    the earlier checks instead of reading the original's result.
    """
    return (
        f"WITH __sql_optim_original AS MATERIALIZED (\n{original}\n),\n"
        f"__sql_optim_optimized AS MATERIALIZED (\n{optimized}\n)\n"
        "SELECT COUNT(*) FROM (\n"
        "    (FROM __sql_optim_original EXCEPT ALL FROM __sql_optim_optimized)\n"
        "    UNION ALL\n"
        "    (FROM __sql_optim_optimized EXCEPT ALL FROM __sql_optim_original)\n"
        ") differing"
    )


def _single_statement_error(query: str) -> Optional[str]:
    """
    Return an error string if *query* is not exactly one SQL statement.

    Agent SQL gets `EXPLAIN ANALYZE ` prefixed onto it for timing. A prefix only
    binds to the first statement, so `SELECT 1; SELECT * FROM events` leaves the
    real query running as a statement of its own and the plan fetch returns its
    whole result set, past `_MAX_MATERIALIZED_ROWS`. The deadline does not save
    us there, because `interrupt()` cannot stop rows that are already being
    converted into Python objects.

    DuckDB's own parser decides what counts as a statement, so a `;` inside a
    string literal, a leading `--` comment and a trailing terminator all stay
    valid. That is the part an earlier keyword-scanning pre-check got wrong.

    Args:
        query (`str`):
            The SQL to inspect.

    Returns:
        `str` or `None`: an error message, or `None` when the query is a single
        statement. A query DuckDB cannot parse is reported as its parse error.
    """
    try:
        statements = duckdb.extract_statements(query)
    except Exception as exc:
        return str(exc)
    if len(statements) != 1:
        return (
            f"Expected a single SQL statement, got {len(statements)}. "
            "Submit one query, without a second statement after `;`."
        )
    return None


class QueryExecutor:
    """
    Runs SQL against an in-memory DuckDB database with realistic
    synthetic data.  Provides execution timing, result correctness
    checks, and EXPLAIN plans — all used by the reward function.
    """

    def __init__(self) -> None:
        # Build the tables once on a writable connection to a temp file, then
        # reopen the database read-only. Every query — trusted task SQL and
        # agent rewrites alike — runs against the read-only database, so the
        # DuckDB engine itself rejects any *persistent* write (no keyword
        # heuristics, and DML inside a CTE cannot slip through on any DuckDB
        # version) and, per `_DUCKDB_CONFIG`, cannot reach the filesystem
        # either.
        #
        # Read-only is not sufficient on its own: DuckDB still accepts
        # session-scoped DDL, so `CREATE TEMP TABLE users AS ...` succeeds and
        # shadows a base table for every later query on the same connection.
        # Queries therefore run on a short-lived cursor (`_session`), whose
        # temporary catalog dies with it, rather than directly on `self.conn`.
        # A lock serializes access because a single DuckDB connection is not
        # safe for concurrent use by the server's worker pool.
        self._dir = tempfile.mkdtemp(prefix="sql_optim_")
        self._path = os.path.join(self._dir, "sql_optim.duckdb")
        builder = duckdb.connect(self._path, config=_DUCKDB_CONFIG)
        self._build_tables(builder)
        builder.close()

        self.conn = duckdb.connect(self._path, read_only=True, config=_DUCKDB_CONFIG)
        # Innermost-last stack of the cursors currently executing, so a deadline
        # armed at any nesting level cancels whatever is actually running.
        self._sessions: List[duckdb.DuckDBPyConnection] = []
        # Reentrant so `_run` can hold the lock across a whole measurement while
        # the helpers it calls still take it individually.
        self._exec_lock = threading.RLock()
        atexit.register(self._cleanup)

    def _cleanup(self) -> None:
        """Close the connection and remove the temp database directory."""
        try:
            self.conn.close()
        except Exception:
            pass
        shutil.rmtree(self._dir, ignore_errors=True)

    # ── Schema Setup ─────────────────────────────────────────────────────

    def _build_tables(self, conn: "duckdb.DuckDBPyConnection") -> None:
        """Create and populate all four synthetic tables on *conn*."""

        # users — 10k rows
        conn.execute("""
            CREATE TABLE users AS
            SELECT
                i                                                      AS id,
                'u' || i || '@mail.com'                                AS email,
                CASE i % 3
                    WHEN 0 THEN 'premium'
                    WHEN 1 THEN 'free'
                    ELSE 'enterprise' END                              AS tier,
                CASE i % 5
                    WHEN 0 THEN 'US'   WHEN 1 THEN 'EU'
                    WHEN 2 THEN 'IN'   WHEN 3 THEN 'UK'
                    ELSE 'AU' END                                      AS region,
                CASE i % 2 WHEN 0 THEN 'premium' ELSE 'basic' END     AS plan,
                DATE '2020-01-01' + CAST(i AS INTEGER)                 AS created_at
            FROM generate_series(1, 10000) t(i)
        """)

        # orders — 500k rows
        conn.execute("""
            CREATE TABLE orders AS
            SELECT
                i                                                      AS id,
                1 + (i % 10000)                                        AS customer_id,
                (i % 100) + 1                                          AS product_id,
                CASE i % 4
                    WHEN 0 THEN 'completed'  WHEN 1 THEN 'pending'
                    WHEN 2 THEN 'cancelled'  ELSE 'shipped' END        AS status,
                ROUND((i % 1000) * 1.5 + 49.99, 2)                   AS total,
                DATE '2023-01-01' + CAST(i % 730 AS INTEGER)          AS created_at
            FROM generate_series(1, 500000) t(i)
        """)

        # products — 1k rows
        conn.execute("""
            CREATE TABLE products AS
            SELECT
                i                                                      AS id,
                'Product_' || i                                        AS name,
                CASE i % 5
                    WHEN 0 THEN 'Electronics'  WHEN 1 THEN 'Clothing'
                    WHEN 2 THEN 'Food'         WHEN 3 THEN 'Books'
                    ELSE 'Sports' END                                  AS category,
                ROUND((i % 500) + 9.99, 2)                            AS price
            FROM generate_series(1, 1000) t(i)
        """)

        # events — 1M rows
        conn.execute("""
            CREATE TABLE events AS
            SELECT
                i                                                      AS id,
                1 + (i % 10000)                                        AS user_id,
                'sess_' || (i % 50000)                                 AS session_id,
                CASE i % 6
                    WHEN 0 THEN 'purchase'  WHEN 1 THEN 'view'
                    WHEN 2 THEN 'click'     WHEN 3 THEN 'signup'
                    WHEN 4 THEN 'logout'    ELSE 'search' END          AS event_type,
                DATE '2024-01-01' + CAST(i % 365 AS INTEGER)          AS occurred_at
            FROM generate_series(1, 1000000) t(i)
        """)

    # ── Execution helpers ─────────────────────────────────────────────────

    @contextlib.contextmanager
    def _session(self) -> Iterator[duckdb.DuckDBPyConnection]:
        """Yield a short-lived cursor to run one statement on.

        `cursor()` opens another connection to the same database with its own
        temporary catalog, so session DDL an agent smuggles in
        (`CREATE TEMP TABLE orders AS SELECT 1`, which a read-only database
        still permits) is discarded when the cursor closes instead of shadowing
        a base table for every later episode in the process.

        Registered on `_sessions` while it runs so `_deadline` can cancel it.
        """
        cursor = self.conn.cursor()
        self._sessions.append(cursor)
        try:
            yield cursor
        finally:
            self._sessions.pop()
            try:
                cursor.close()
            except Exception:
                pass

    def _interrupt_active(self) -> None:
        """Cancel the innermost running session, if any.

        Resolved when the timer fires rather than when it is armed, so a
        deadline set around a whole measurement still cancels the individual
        statement that a helper is running underneath it.
        """
        # Read once: the owning thread can pop the last session between an
        # emptiness check and the index.
        try:
            target = self._sessions[-1]
        except IndexError:
            target = self.conn
        try:
            target.interrupt()
        except Exception:
            pass

    @contextlib.contextmanager
    def _deadline(self) -> Iterator[Callable[[], bool]]:
        """Cancel whatever is running on the connection after the time limit.

        DuckDB has no statement timeout, but ``interrupt()`` cancels the running
        query from another thread and leaves the connection usable, raising
        ``duckdb.InterruptException`` in the thread that issued it.

        Armed by every helper that executes agent SQL *and* around the whole of
        ``_run``, so a helper is bounded whoever calls it while the measurement
        as a whole is bounded too. Nesting is harmless: each level stops its
        own watchdog, and the earliest one to expire wins.

        Must be entered while already holding ``_exec_lock`` — every call site
        does, as ``with self._exec_lock, self._deadline():``. That ordering is
        what makes the unavoidable cancel race benign: the watchdog can fire in
        the window between the statement finishing and the context exiting. It
        is joined before the lock is released, so an interrupt already in
        flight cannot target a later session. With the lock held, no other
        session can have a query in flight, so the stray ``interrupt()`` lands on
        an idle connection and does nothing.

        Callers must not treat the interrupt as a reason to try a different
        strategy: the deadline has already expired.

        An interrupt that reaches an idle connection is not remembered, so a
        single one is not enough for a caller that runs several statements:
        expiring in the gap between two of them would let the next one run
        unbounded. Once expired, the watchdog therefore keeps re-sending the
        interrupt until the context exits. It also yields an ``expired()``
        check, which multi-statement callers use before each stage and before
        trusting a result, since a statement can complete just as the deadline
        passes.
        """
        expires = time.monotonic() + _QUERY_TIMEOUT_S
        released = threading.Event()

        def watchdog() -> None:
            if released.wait(_QUERY_TIMEOUT_S):
                return
            while not released.is_set():
                self._interrupt_active()
                released.wait(_INTERRUPT_RETRY_S)

        thread = threading.Thread(target=watchdog, daemon=True)
        thread.start()
        try:
            yield lambda: time.monotonic() >= expires
        finally:
            released.set()
            thread.join()

    def _probe(self, query: str) -> Tuple[Optional[List], Optional[str]]:
        """
        Fetch at most ``_MAX_MATERIALIZED_ROWS`` rows of *query*.

        Returns (rows, error). ``rows`` is None when the result set is larger
        than the cap — the caller then works from the in-engine row count
        instead of holding the full result in Python.

        This is also the authoritative error check: the query runs unwrapped on
        the read-only connection, so a write (DDL/DML, inside a CTE, or after a
        ``;``) still surfaces as a DuckDB error exactly as before. Multi-statement
        input is rejected here, before anything executes, because ``_run`` stops
        on this error and so never reaches the plan fetch it would abuse.
        """
        error = _single_statement_error(query)
        if error is not None:
            return None, error

        try:
            with self._exec_lock, self._session() as cur, self._deadline():
                # The leftover rows of an over-cap result are released when the
                # session closes, so nothing stays pending on `self.conn`.
                head = cur.execute(query).fetchmany(_MAX_MATERIALIZED_ROWS + 1)
        except duckdb.InterruptException:
            return None, _TIMEOUT_ERROR
        except Exception as exc:
            return None, str(exc)
        if len(head) > _MAX_MATERIALIZED_ROWS:
            return None, None
        return head, None

    def _row_count(self, query: str) -> Tuple[Optional[int], Optional[str]]:
        """
        Exact row count for *query*, computed inside DuckDB — no rows cross into
        Python. Falls back to a streaming drain (bounded to one batch of memory)
        for the rare query that cannot be wrapped in a subquery, but never after
        a timeout: the drain would be at least as expensive as the query that
        just ran out of time.
        """
        try:
            with self._exec_lock, self._session() as cur, self._deadline():
                return (
                    cur.execute(_as_subquery(query, "COUNT(*)")).fetchone()[0],
                    None,
                )
        except duckdb.InterruptException:
            return None, _TIMEOUT_ERROR
        except Exception:
            pass

        try:
            count = 0
            with self._exec_lock, self._session() as cur, self._deadline():
                result = cur.execute(query)
                while True:
                    batch = result.fetchmany(_FETCH_BATCH)
                    if not batch:
                        break
                    count += len(batch)
            return count, None
        except duckdb.InterruptException:
            return None, _TIMEOUT_ERROR
        except Exception as exc:
            return None, str(exc)

    def _engine_ms(self, query: str, runs: int) -> Optional[float]:
        """
        Median DuckDB-side execution time in ms, or None if unavailable.

        ``EXPLAIN ANALYZE`` really executes the query and discards the result
        inside the engine, so this measures execution alone: no rows are
        converted to Python objects and nothing is buffered client-side. Returns
        None when the profile footer cannot be parsed (a DuckDB version whose
        output differs), which makes the caller fall back to wall-clock timing.
        A timeout propagates instead, so the caller does not answer it by
        running the query yet again.
        """
        timings: List[float] = []
        for _ in range(runs):
            try:
                with self._exec_lock, self._session() as cur, self._deadline():
                    # Bounded on purpose: a plan is tiny, so anything larger is
                    # not a plan and must not be pulled into Python wholesale.
                    rows = cur.execute(f"EXPLAIN ANALYZE {query}").fetchmany(
                        _MAX_PLAN_ROWS
                    )
            except duckdb.InterruptException:
                raise
            except Exception:
                return None
            match = _TOTAL_TIME_RE.search("\n".join(str(r[-1]) for r in rows))
            if match is None:
                return None
            timings.append(float(match.group(1)) * 1000.0)

        timings.sort()
        return round(max(timings[len(timings) // 2], _MIN_MEASURABLE_MS), 3)

    def _wall_ms(self, query: str, runs: int) -> Tuple[float, Optional[str]]:
        """
        Wall-clock fallback timing: time a streaming drain that discards rows.

        Includes client-side conversion cost, so it is less faithful than
        ``_engine_ms``, but memory stays bounded to a single batch instead of
        the whole result set.
        """
        timings: List[float] = []
        for _ in range(runs):
            try:
                with self._exec_lock, self._session() as cur, self._deadline():
                    t0 = time.perf_counter()
                    result = cur.execute(query)
                    while result.fetchmany(_FETCH_BATCH):
                        pass
                    timings.append((time.perf_counter() - t0) * 1000.0)
            except duckdb.InterruptException:
                raise
            except Exception as exc:
                return 99_999.0, str(exc)

        timings.sort()
        return round(max(timings[len(timings) // 2], _MIN_MEASURABLE_MS), 3), None

    def _run(
        self, query: str, runs: int = _TIMING_RUNS
    ) -> Tuple[float, Optional[int], Optional[List], Optional[str]]:
        """
        Execute *query* on the read-only connection and measure it.

        Returns (median_ms, row_count, rows, error_or_None). ``row_count`` is
        exact whenever there is no error; ``rows`` is None for result sets above
        ``_MAX_MATERIALIZED_ROWS``. A write raises and is returned as the error
        string.

        The timing comes from DuckDB's own profiler rather than from wrapping a
        ``fetchall()``, so it reflects query execution instead of the cost of
        building Python tuples — for a 1M-row scan the latter was roughly 30x
        the former and swamped the signal the reward function is built on.

        The whole measurement shares one deadline, so an agent-authored query
        cannot hold the execution lock — and with it every other session — for
        longer than ``_QUERY_TIMEOUT_S``.
        """
        with self._exec_lock, self._deadline():
            try:
                rows, error = self._probe(query)
                if error is not None:
                    return 99_999.0, None, None, error

                if rows is not None:
                    row_count: Optional[int] = len(rows)
                else:
                    row_count, error = self._row_count(query)
                    if error is not None:
                        return 99_999.0, None, None, error

                median_ms = self._engine_ms(query, runs)
                if median_ms is None:
                    median_ms, error = self._wall_ms(query, runs)
                    if error is not None:
                        return 99_999.0, None, None, error
            except duckdb.InterruptException:
                return 99_999.0, None, None, _TIMEOUT_ERROR

        return median_ms, row_count, rows, None

    @staticmethod
    def _output_schema(
        cur: duckdb.DuckDBPyConnection, query: str
    ) -> List[Tuple[str, str]]:
        """Return the output columns of *query* as (name, type) pairs, in order.

        ``DESCRIBE <query>`` binds the query without running it, so it is cheap
        however expensive the query is, and it reports the names the query
        itself produces. Describing a wrapper such as ``SELECT * FROM (<query>)``
        would not do: the wrapper deduplicates ``SELECT 1 x, 2 x`` into
        ``(x, x_1)``, which would make it look identical to ``SELECT 1 x, 2 x_1``.

        Names are folded with ASCII-only case folding. Type strings are kept
        exactly as DuckDB prints them, which already canonicalizes aliases
        (``INT4`` is reported as ``INTEGER``, including inside nested types) but
        keeps the case of struct field names and decimal precision and scale.
        """
        return [
            (name.translate(_ASCII_LOWER), column_type)
            for name, column_type, *_ in cur.execute(f"DESCRIBE {query}").fetchall()
        ]

    def _results_equal(self, original: str, optimized: str) -> bool:
        """Return whether two queries produce the same result.

        The equality rule, the same for every result size, empty results
        included:

        - Same output schema, compared by position: the same number of columns,
          the same names (ASCII case-insensitive) and identical DuckDB types.
        - The same rows as a multiset: row order is ignored, duplicate
          multiplicity is not. Values are compared with DuckDB's own equality,
          with no float tolerance, so ``-0.0`` equals ``0.0``, ``NaN`` equals
          ``NaN`` and ``NULL`` equals ``NULL``. Engine equality is not byte
          identity: ``INTERVAL '1 month'`` equals ``INTERVAL '30 days'``.

        The schema check comes first because it is what keeps the set operation
        from coercing values: without it ``true`` would equal ``1`` and a
        ``BIGINT`` would equal a ``DOUBLE`` that cannot represent it. Only the
        count of differing rows comes back to Python.

        Schema inspection and comparison share one session, lock and deadline,
        and the deadline is checked before each stage and before a match is
        reported. Any error or timeout is a mismatch with no fallback: a result
        that cannot be verified earns no correctness credit. This compares fresh
        executions of both queries, so it does not establish equivalence for
        volatile queries such as ones using ``random()``.
        """
        if _single_statement_error(original) or _single_statement_error(optimized):
            return False
        try:
            with self._exec_lock, self._session() as cur, self._deadline() as expired:
                schemas = []
                for query in (original, optimized):
                    if expired():
                        return False
                    schemas.append(self._output_schema(cur, query))
                if schemas[0] != schemas[1] or expired():
                    return False
                sql = _symmetric_difference_sql(original, optimized)
                differing = cur.execute(sql).fetchone()[0]
                return differing == 0 and not expired()
        except Exception:
            # Includes `duckdb.InterruptException`.
            return False

    # ── Public API ────────────────────────────────────────────────────────

    def compare(self, original: str, optimized: str) -> Dict[str, Any]:
        """
        Execute both queries, measure real timing, check correctness.

        Returns a dict with:
          original_ms, optimized_ms, speedup,
          results_match, original_rows, optimized_rows,
          original_error, optimized_error, verdict
        """
        # Normalize (drop trailing ``;``) so both direct execution and the
        # wrapped correctness check are valid.
        original = _strip_terminators(original)
        optimized = _strip_terminators(optimized)

        orig_ms, orig_count, _, orig_err = self._run(original)

        # The optimized query is agent-authored, but the read-only connection is
        # the guarantee: any write (DDL/DML, in a CTE, or after a `;`) is rejected
        # by the DuckDB engine and surfaces as `optimized_error`. No structural
        # pre-check is needed, which also avoids false rejections of valid
        # rewrites (leading comments, a `;` inside a string literal, etc.).
        opt_ms, opt_count, _, opt_err = self._run(optimized)

        # ── Correctness: do both queries return the same data? ────────
        # Different row counts settle it. Otherwise the exact comparison decides,
        # for empty results too, since those still need matching schemas.
        results_match = (
            orig_err is None
            and opt_err is None
            and orig_count == opt_count
            and self._results_equal(original, optimized)
        )

        # ── Speedup ratio ─────────────────────────────────────────────
        speedup = 1.0
        if opt_ms > 0 and orig_ms < 90_000:
            speedup = round(orig_ms / opt_ms, 3)

        # ── Human-readable verdict ────────────────────────────────────
        if opt_err:
            verdict = f"[FAIL] Optimized query error: {opt_err[:120]}"
        elif results_match and speedup >= 2.0:
            verdict = f"[OK] {speedup:.1f}x faster with correct results"
        elif results_match and speedup >= 1.0:
            verdict = (
                f"[WARN] Correct results but only {speedup:.1f}x speedup -- dig deeper"
            )
        elif not results_match and speedup >= 2.0:
            verdict = (
                f"[WARN] {speedup:.1f}x faster but results don't match -- fix the logic"
            )
        else:
            verdict = f"[FAIL] {speedup:.1f}x -- no meaningful improvement"

        return {
            "original_ms": orig_ms,
            "optimized_ms": opt_ms,
            "speedup": speedup,
            "results_match": results_match,
            "original_rows": orig_count if orig_count is not None else 0,
            "optimized_rows": opt_count if opt_count is not None else 0,
            "original_error": orig_err,
            "optimized_error": opt_err,
            "verdict": verdict,
        }

    def explain(self, query: str) -> str:
        """Return EXPLAIN output for a query.

        Plain ``EXPLAIN`` only plans the query, but it is still agent-authored,
        so it runs under the same deadline as everything else.
        """
        error = _single_statement_error(_strip_terminators(query))
        if error is not None:
            return f"EXPLAIN error: {error}"

        try:
            with self._exec_lock, self._session() as cur, self._deadline():
                rows = cur.execute(f"EXPLAIN {_strip_terminators(query)}").fetchall()
            return "\n".join(str(r[1]) for r in rows)
        except duckdb.InterruptException:
            return f"EXPLAIN error: {_TIMEOUT_ERROR}"
        except Exception as exc:
            return f"EXPLAIN error: {exc}"

    @property
    def table_stats(self) -> Dict[str, int]:
        tables = ["users", "orders", "products", "events"]
        # No deadline: these are fixed, trusted queries, so arming one would only
        # add a window in which a stray interrupt could cancel them.
        with self._exec_lock, self._session() as cur:
            return {
                t: cur.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                for t in tables
            }


# ── Singleton accessor ────────────────────────────────────────────────────


def get_executor() -> QueryExecutor:
    """Return the process-level singleton (lazy init, thread-safe)."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = QueryExecutor()
    return _instance
