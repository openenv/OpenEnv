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

Agent SQL never runs in the server process. The server builds the database,
then hands every query to a worker process (this file, run as a script) that it
kills when the deadline passes or the worker dies, so a runaway query costs the
worker instead of the server. This file must therefore import only the standard
library and duckdb.
"""

import atexit
import contextlib
import json
import multiprocessing
import os
import re
import shutil
import string
import subprocess
import sys
import tempfile
import threading
import time
from multiprocessing.connection import Connection
from typing import Any, Dict, Iterator, List, Optional, Tuple

import duckdb

_instance: Optional["QueryExecutor"] = None
_lock = threading.Lock()

# An `EXPLAIN ANALYZE` plan is a couple of rows; this only exists so the plan
# fetch can never become an unbounded materialization of agent-chosen data.
_MAX_PLAN_ROWS = 2_000
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
# no statement-timeout option, and `interrupt()` is only checked between units
# of work, so the server kills the worker process when the deadline passes.
# These caps cover buffer-managed memory and spill only. Agent results are
# never converted into Python, in the worker or the server: only counts,
# timings and plans cross the process boundary. Allocations DuckDB does not
# track, such as one huge `repeat()`, are bounded by the worker's address-space
# cap and the container; see `_WORKER_ADDRESS_SPACE_BYTES`.
# Keep headroom in the 1024 MiB manifest budget, including RAM-backed /tmp in
# docker-local validation.
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
_WORKER_FAILED_ERROR = "Query aborted: the query process ran out of memory or crashed"
# Starting the worker is trusted work and is not charged to any query's deadline.
_WORKER_START_TIMEOUT_S = 30.0
# Memory DuckDB does not track (one huge `repeat()`) is bounded by the
# container, so on Linux the worker volunteers as the OOM killer's victim
# (`oom_score_adj`): when the container runs out, the kernel kills the worker
# and the server survives. That holds when the OOM killer picks one process,
# as under Docker's defaults; a runtime that kills a container's processes as a
# group still takes the server down. The address-space cap is a backstop for deployments without
# a container limit: an allocation that would take the worker past 1 GiB of
# address space fails with DuckDB's out-of-memory error. It is not
# tighter because address space is a poor proxy for memory here: the honest
# task workload uses ~400 MiB resident, yet peaks at 580-660 MiB of address
# space in the validation image and up to ~850 MiB on other Linux hosts, and a
# cap below that fails honest comparisons at random.
_WORKER_ADDRESS_SPACE_BYTES = 1024 * 1024 * 1024
_WORKER_ENV = {"MALLOC_ARENA_MAX": "2"}
_TABLES = ("users", "orders", "products", "events")
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
    real query running as a statement of its own and the plan fetch converts its
    rows into Python instead of a plan. That would bypass the rule that agent
    results never become Python objects, and the time spent converting them
    would be measured as query time.

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


# ── Worker side: runs in the query process ───────────────────────────────


@contextlib.contextmanager
def _session(conn: duckdb.DuckDBPyConnection) -> Iterator[duckdb.DuckDBPyConnection]:
    """Yield a short-lived cursor to run one statement on.

    `cursor()` opens another connection to the same database with its own
    temporary catalog, so session DDL an agent smuggles in
    (`CREATE TEMP TABLE orders AS SELECT 1`, which a read-only database
    still permits) is discarded when the cursor closes instead of shadowing
    a base table for every later query the worker runs.
    """
    cursor = conn.cursor()
    try:
        yield cursor
    finally:
        try:
            cursor.close()
        except Exception:
            pass


def _row_count(conn: duckdb.DuckDBPyConnection, query: str) -> int:
    """
    Exact row count for *query*, computed inside DuckDB: no rows are converted
    into Python. A query that cannot be wrapped in a subquery (``CALL``, DDL) is
    an error rather than a reason to fetch its rows.
    """
    with _session(conn) as cur:
        return cur.execute(_as_subquery(query, "COUNT(*)")).fetchone()[0]


def _engine_ms(conn: duckdb.DuckDBPyConnection, query: str, runs: int) -> float:
    """
    Median DuckDB-side execution time of *query* in ms.

    ``EXPLAIN ANALYZE`` really executes the query and discards the result
    inside the engine, so this measures execution alone: no rows are converted
    to Python objects and nothing is buffered client-side. Any failure raises,
    including a profile footer that cannot be parsed (a DuckDB version whose
    output differs): there is no fallback that would fetch the rows instead.
    """
    timings: List[float] = []
    for _ in range(runs):
        with _session(conn) as cur:
            # Bounded on purpose: a plan is tiny, so anything larger is not a
            # plan and must not be pulled into Python wholesale.
            rows = cur.execute(f"EXPLAIN ANALYZE {query}").fetchmany(_MAX_PLAN_ROWS)
        # DuckDB 1.5 prints the statement on one line above the timing box.
        # Skip it, so "Total Time: 0s" in the agent's SQL is not read as its time.
        profile = "\n".join(
            line
            for line in "\n".join(str(r[-1]) for r in rows).split("\n")
            if not line.startswith("EXPLAIN ANALYZE ")
        )
        match = _TOTAL_TIME_RE.search(profile)
        if match is None:
            raise RuntimeError("EXPLAIN ANALYZE output has no total time")
        timings.append(float(match.group(1)) * 1000.0)

    timings.sort()
    return round(max(timings[len(timings) // 2], _MIN_MEASURABLE_MS), 3)


def _output_schema(cur: duckdb.DuckDBPyConnection, query: str) -> List[Tuple[str, str]]:
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


def _same_results(
    conn: duckdb.DuckDBPyConnection, original: str, optimized: str
) -> bool:
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
    count of differing rows comes back to Python. Any error raises; the
    server treats it as a mismatch, with no fallback. This compares fresh
    executions of both queries, so it does not establish equivalence for
    volatile queries such as ones using ``random()``.
    """
    with _session(conn) as cur:
        schemas = [_output_schema(cur, query) for query in (original, optimized)]
        if schemas[0] != schemas[1]:
            return False
        sql = _symmetric_difference_sql(original, optimized)
        return cur.execute(sql).fetchone()[0] == 0


def _handle(conn: duckdb.DuckDBPyConnection, job: Dict[str, Any]) -> Dict[str, Any]:
    """Run one job from the server and return its JSON-serializable reply.

    A failure is a reply, never an exception. ``fatal`` marks an out-of-memory
    failure, after which the server replaces the worker rather than trust a
    process whose allocator has already failed once.
    """
    try:
        op = job["op"]
        if op == "measure":
            rows = _row_count(conn, job["query"])
            ms = _engine_ms(conn, job["query"], job["runs"])
            return {"ok": True, "ms": ms, "rows": rows}
        if op == "equal":
            equal = _same_results(conn, job["original"], job["optimized"])
            return {"ok": True, "equal": equal}
        if op == "explain":
            with _session(conn) as cur:
                rows = cur.execute(f"EXPLAIN {job['query']}").fetchall()
            return {"ok": True, "plan": "\n".join(str(r[1]) for r in rows)}
        if op == "stats":
            with _session(conn) as cur:
                stats = {
                    t: cur.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                    for t in _TABLES
                }
            return {"ok": True, "stats": stats}
        raise ValueError(f"Unknown query worker job: {op!r}")
    except Exception as exc:
        fatal = isinstance(exc, (duckdb.OutOfMemoryException, MemoryError))
        # The agent chooses the error text (`error(repeat(...))`), so cap it.
        return {"ok": False, "error": str(exc)[:2000], "fatal": fatal}


def _confine_worker(address_space_bytes: Optional[int]) -> None:
    """Cap this process's address space and make it the first OOM victim.

    Linux only; elsewhere the worker still runs, just without either guard.
    Raising one's own ``oom_score_adj`` needs no privilege, but ``/proc`` can
    be read-only in some sandboxes, so a failed write is ignored.
    """
    if sys.platform != "linux":
        return
    import resource

    if address_space_bytes:
        # A tighter hard limit the host already set stays: it cannot be raised.
        _, hard = resource.getrlimit(resource.RLIMIT_AS)
        if hard != resource.RLIM_INFINITY:
            address_space_bytes = min(address_space_bytes, hard)
        limit = (address_space_bytes, address_space_bytes)
        resource.setrlimit(resource.RLIMIT_AS, limit)
    with contextlib.suppress(OSError), open("/proc/self/oom_score_adj", "w") as f:
        f.write("1000")


def _exit_when_orphaned(interval_s: float = 0.5) -> None:
    """Exit as soon as the server process is gone, even in the middle of a job.

    Between jobs a closed pipe already ends the worker, but during a job it is
    busy inside DuckDB and would otherwise run a runaway query to completion.
    """
    server = os.getppid()

    def watch() -> None:
        while os.getppid() == server:
            time.sleep(interval_s)
        os._exit(1)

    threading.Thread(target=watch, daemon=True).start()


def _worker_main(fd: int) -> None:
    """Serve jobs from the server over the pipe at *fd* until it closes."""
    _exit_when_orphaned()
    pipe = Connection(fd)
    try:
        setup = json.loads(pipe.recv_bytes())
        _confine_worker(setup["address_space_bytes"])
        conn = duckdb.connect(setup["path"], read_only=True, config=setup["config"])
        conn.execute("SELECT 1").fetchall()
    except Exception as exc:
        error = f"Query process failed to start: {exc}"
        with contextlib.suppress(OSError):
            pipe.send_bytes(json.dumps({"ok": False, "error": error}).encode())
        return
    try:
        pipe.send_bytes(json.dumps({"ok": True}).encode())
        while True:
            job = json.loads(pipe.recv_bytes())
            pipe.send_bytes(json.dumps(_handle(conn, job)).encode())
    except (EOFError, OSError):
        # The server closed its end, or went away: nothing left to serve.
        return


# ── Server side ──────────────────────────────────────────────────────────


class QueryExecutor:
    """
    Runs SQL against an in-memory DuckDB database with realistic
    synthetic data.  Provides execution timing, result correctness
    checks, and EXPLAIN plans — all used by the reward function.
    """

    def __init__(self) -> None:
        # Build the tables once on a writable connection to a temp file. Every
        # query — trusted task SQL and agent rewrites alike — then runs in the
        # worker process on a read-only connection to that file, so the DuckDB
        # engine itself rejects any *persistent* write (no keyword heuristics,
        # and DML inside a CTE cannot slip through on any DuckDB version) and,
        # per `_DUCKDB_CONFIG`, cannot reach the filesystem either.
        #
        # Read-only is not sufficient on its own: DuckDB still accepts
        # session-scoped DDL, so `CREATE TEMP TABLE users AS ...` succeeds and
        # shadows a base table for every later query on the same connection.
        # The worker therefore runs each statement on a short-lived cursor
        # (`_session`), whose temporary catalog dies with it.
        #
        # One worker serves every session, one job at a time: the lock
        # serializes them, and parallel workers would compete for the same
        # CPUs and add noise to the timings the reward is built on.
        self._dir = tempfile.mkdtemp(prefix="sql_optim_")
        self._path = os.path.join(self._dir, "sql_optim.duckdb")
        builder = duckdb.connect(self._path, config=_DUCKDB_CONFIG)
        self._build_tables(builder)
        builder.close()

        self._worker: Optional[subprocess.Popen] = None
        self._pipe: Optional[Connection] = None
        self._spill_dir: Optional[str] = None
        self._exec_lock = threading.Lock()
        # A process forked from this one inherits the executor. It must neither
        # share the worker's pipe nor, at exit, kill the worker or delete the
        # database, so both are tied to the process that created them.
        self._owner_pid = os.getpid()
        self._worker_owner_pid = os.getpid()
        atexit.register(self._cleanup)

    def _cleanup(self) -> None:
        """Stop the worker and remove the temp database directory."""
        self._kill_worker()
        if os.getpid() == self._owner_pid:
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

    # ── Worker process ────────────────────────────────────────────────────

    def _start_worker(self) -> Optional[str]:
        """Make sure a live worker is ready; return an error if none can start.

        Replaces a worker that died since the last job before anything is sent
        to it. Starting one is trusted work, bounded by its own timeout rather
        than any query's deadline.
        """
        if (
            self._worker is not None
            and self._worker_owner_pid == os.getpid()
            and self._worker.poll() is None
        ):
            return None
        self._kill_worker()

        server_end, worker_end = multiprocessing.Pipe()
        fd = worker_end.fileno()
        try:
            # Each worker spills to its own directory, so removing it after a
            # kill cannot touch the files of a worker a forked copy started.
            spill_dir = tempfile.mkdtemp(prefix="spill_", dir=self._dir)
            # `-P` keeps this directory off `sys.path`, so the worker imports
            # only what this file imports. A new session keeps a Ctrl-C meant
            # for the server from reaching the worker mid-job; it exits by
            # itself once the server closes its end of the pipe or dies.
            worker = subprocess.Popen(
                [sys.executable, "-P", os.path.abspath(__file__), str(fd)],
                pass_fds=(fd,),
                start_new_session=True,
                # glibc reads this at startup, so it has to be in the
                # environment rather than set by the worker itself.
                env={**os.environ, **_WORKER_ENV},
            )
        except OSError as exc:
            server_end.close()
            return f"Query process failed to start: {exc}"
        finally:
            worker_end.close()
        self._worker, self._pipe, self._spill_dir = worker, server_end, spill_dir
        self._worker_owner_pid = os.getpid()

        setup = {
            "path": self._path,
            "config": {**_DUCKDB_CONFIG, "temp_directory": self._spill_dir},
            "address_space_bytes": _WORKER_ADDRESS_SPACE_BYTES,
        }
        try:
            server_end.send_bytes(json.dumps(setup).encode())
            if server_end.poll(_WORKER_START_TIMEOUT_S):
                reply = json.loads(server_end.recv_bytes())
            else:
                reply = {"ok": False, "error": "Query process did not start in time"}
        except (EOFError, OSError):
            reply = {"ok": False, "error": _WORKER_FAILED_ERROR}
        except BaseException:
            # The ready message may still arrive and be read as a job's reply.
            self._kill_worker()
            raise
        if not reply["ok"]:
            self._kill_worker()
            return reply["error"]
        return None

    def _kill_worker(self) -> None:
        """Kill and reap the worker, if there is one. Safe to call repeatedly."""
        worker, pipe, spill_dir = self._worker, self._pipe, self._spill_dir
        self._worker = self._pipe = self._spill_dir = None
        if pipe is not None:
            pipe.close()
        # A worker inherited through fork belongs to the parent: let it be.
        if worker is not None and self._worker_owner_pid == os.getpid():
            if worker.poll() is None:
                worker.kill()
            worker.wait()
            # DuckDB removes its spill files only when it shuts down cleanly.
            shutil.rmtree(spill_dir, ignore_errors=True)

    def _call(self, job: Dict[str, Any]) -> Dict[str, Any]:
        """Run *job* in the worker under the query deadline and return its reply.

        Agent SQL only ever runs here, in the worker process. When the deadline
        passes the worker is killed, whatever it is doing: DuckDB checks
        ``interrupt()`` only between units of work, so one long operation inside
        the engine could otherwise run on while holding the lock. A worker
        that dies, or runs out of memory, is replaced the same way. Every
        failure comes back as ``{"ok": False, "error": ...}``, which callers
        treat as an error with no credit.

        Nothing from a killed worker can leak into a later job: the reply is
        only read before the deadline, and the worker that might still send one
        is gone.
        """
        with self._exec_lock:
            error = self._start_worker()
            if error is not None:
                return {"ok": False, "error": error}
            expires = time.monotonic() + _QUERY_TIMEOUT_S
            try:
                self._pipe.send_bytes(json.dumps(job).encode())
                if self._pipe.poll(max(0.0, expires - time.monotonic())):
                    reply = json.loads(self._pipe.recv_bytes())
                else:
                    reply = {"ok": False, "error": _TIMEOUT_ERROR, "fatal": True}
            except (EOFError, OSError):
                reply = {"ok": False, "error": _WORKER_FAILED_ERROR, "fatal": True}
            except BaseException:
                # The job may still be running and will send a reply. Left in
                # place, it would be read as the answer to the next job.
                self._kill_worker()
                raise
            if reply.get("fatal"):
                self._kill_worker()
            return reply

    # ── Execution helpers ─────────────────────────────────────────────────

    def _run(
        self, query: str, runs: int = _TIMING_RUNS
    ) -> Tuple[float, Optional[int], Optional[str]]:
        """
        Execute *query* in the worker and measure it.

        Returns (median_ms, row_count, error_or_None). ``row_count`` is exact
        whenever there is no error, and comes from a ``COUNT(*)`` wrap inside
        DuckDB. No result value is converted into Python: one row can hold a
        whole table (``SELECT list(e) FROM events e``). Any failure, including a
        query the wrap cannot hold (``CALL``, DDL, a write), is returned as the
        error string.

        The timing comes from DuckDB's own profiler rather than from wrapping a
        ``fetchall()``, so it reflects query execution instead of the cost of
        building Python tuples — for a 1M-row scan the latter was roughly 30x
        the former and swamped the signal the reward function is built on.

        The whole measurement shares one deadline, so an agent-authored query
        cannot hold the execution lock — and with it every other session — for
        longer than ``_QUERY_TIMEOUT_S``.
        """
        error = _single_statement_error(query)
        if error is not None:
            return 99_999.0, None, error

        reply = self._call({"op": "measure", "query": query, "runs": runs})
        if not reply["ok"]:
            return 99_999.0, None, reply["error"]
        return reply["ms"], reply["rows"], None

    def _results_equal(self, original: str, optimized: str) -> bool:
        """Return whether two queries produce the same result.

        See `_same_results` for the equality rule. Any error, timeout or worker
        failure is a mismatch with no fallback: a result that cannot be
        verified earns no correctness credit.
        """
        if _single_statement_error(original) or _single_statement_error(optimized):
            return False
        reply = self._call(
            {"op": "equal", "original": original, "optimized": optimized}
        )
        return bool(reply["ok"] and reply["equal"])

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

        orig_ms, orig_count, orig_err = self._run(original)

        # The optimized query is agent-authored, but the read-only connection is
        # the guarantee: any write (DDL/DML, in a CTE, or after a `;`) is rejected
        # by the DuckDB engine and surfaces as `optimized_error`. No structural
        # pre-check is needed, which also avoids false rejections of valid
        # rewrites (leading comments, a `;` inside a string literal, etc.).
        opt_ms, opt_count, opt_err = self._run(optimized)

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
        so it runs in the worker under the same deadline as everything else.
        """
        query = _strip_terminators(query)
        error = _single_statement_error(query)
        if error is not None:
            return f"EXPLAIN error: {error}"

        reply = self._call({"op": "explain", "query": query})
        if not reply["ok"]:
            return f"EXPLAIN error: {reply['error']}"
        return reply["plan"]

    @property
    def table_stats(self) -> Dict[str, int]:
        reply = self._call({"op": "stats"})
        if not reply["ok"]:
            raise RuntimeError(reply["error"])
        return reply["stats"]


# ── Singleton accessor ────────────────────────────────────────────────────


def get_executor() -> QueryExecutor:
    """Return the process-level singleton (lazy init, thread-safe)."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = QueryExecutor()
    return _instance


if __name__ == "__main__":
    _worker_main(int(sys.argv[1]))
