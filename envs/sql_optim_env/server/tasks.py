"""
tasks.py — SQL Query Optimization Tasks
========================================
Five tasks of increasing difficulty, each with a DuckDB-executable
"bad" query (stored in sql_query) that agents must analyze and rewrite.

All queries run against the executor's synthetic tables:
  users    (10,000 rows)  — id, email, tier, region, plan, created_at
  orders   (500,000 rows) — id, customer_id, product_id, status, total, created_at
  products (1,000 rows)   — id, name, category, price
  events   (1,000,000 rows) — id, user_id, session_id, event_type, occurred_at
"""

from typing import Any, Dict

TASKS: Dict[str, Dict[str, Any]] = {
    # ─────────────────────────────────────────────────────────────────
    # TASK 1 — EASY: Basic Anti-pattern Detection
    # ─────────────────────────────────────────────────────────────────
    "task_1_basic_antipatterns": {
        "task_id": "task_1_basic_antipatterns",
        "task_name": "Basic SQL Anti-pattern Detection",
        "task_description": (
            "Analyze the SQL query below for common anti-patterns that destroy performance. "
            "Your rewrite must return the SAME rows and columns: replace the non-sargable "
            "predicates — `CAST(customer_id AS VARCHAR) = '5000'` with `customer_id = 5000`, "
            "and `year(created_at) = 2024` with a `created_at >= '2024-01-01' AND "
            "created_at < '2025-01-01'` range — so DuckDB can prune. "
            "Separately identify — but do NOT remove, since it changes the output — the "
            "SELECT * that fetches every column from 500k rows. "
            "For each issue report: issue_type, line, description, severity, and a concrete fix. "
            "Your optimized_query is EXECUTED against real data; speedup is credited only "
            "when the results still match."
        ),
        "difficulty": "easy",
        "dialect": "duckdb/postgresql",
        "max_steps": 3,
        "schema_info": (
            "Table: orders (500,000 rows)\n"
            "  id INT, customer_id INT, product_id INT,\n"
            "  status VARCHAR, total DECIMAL, created_at DATE\n\n"
            "No indexes defined (DuckDB uses columnar min/max pruning when columns "
            "are not wrapped in functions).\n"
            "Scan cost: ~500k rows × all columns with SELECT *"
        ),
        "sql_query": (
            "SELECT *\n"
            "FROM orders\n"
            "WHERE CAST(customer_id AS VARCHAR) = '5000'\n"
            "  AND year(created_at) = 2024;"
        ),
        "ground_truth_issues": [
            {
                "type": "select_star",
                "line": 1,
                "keywords": [
                    "select *",
                    "star",
                    "all columns",
                    "unnecessary columns",
                    "column projection",
                    "specify columns",
                    "bandwidth",
                ],
            },
            {
                "type": "non_sargable_cast",
                "line": 3,
                "keywords": [
                    "cast",
                    "varchar",
                    "type cast",
                    "type conversion",
                    "non-sargable",
                    "sargable",
                    "integer comparison",
                    "string comparison",
                    "prevents",
                    "pruning",
                ],
            },
            {
                "type": "function_on_date_column",
                "line": 4,
                "keywords": [
                    "year(",
                    "function on column",
                    "non-sargable",
                    "date range",
                    "between",
                    "extract",
                    "full scan",
                    "date filter",
                ],
            },
        ],
        "approved_expected": False,
    },
    # ─────────────────────────────────────────────────────────────────
    # TASK 2 — MEDIUM: N+1 Correlated Subqueries
    # ─────────────────────────────────────────────────────────────────
    "task_2_correlated_subqueries": {
        "task_id": "task_2_correlated_subqueries",
        "task_name": "N+1 Correlated Subquery Elimination",
        "task_description": (
            "The query below uses three correlated scalar subqueries — each one scans "
            "the entire orders table (500k rows) once per premium user (~3,300 users). "
            "That's ~10 million row reads just for aggregation. "
            "Identify the N+1 pattern, explain why each subquery is harmful, "
            "and rewrite the query as a single aggregation JOIN. "
            "Your optimized_query will be executed; results must match the original."
        ),
        "difficulty": "medium",
        "dialect": "duckdb/postgresql",
        "max_steps": 4,
        "schema_info": (
            "Table: users (10,000 rows)\n"
            "  id INT, email VARCHAR, tier VARCHAR, region VARCHAR,\n"
            "  plan VARCHAR, created_at DATE\n\n"
            "Table: orders (500,000 rows)\n"
            "  id INT, customer_id INT, product_id INT,\n"
            "  status VARCHAR, total DECIMAL, created_at DATE\n\n"
            "Premium users: ~3,300  |  Orders per user avg: 50\n"
            "Worst-case scans: 3 subqueries × 3,300 users × 500k rows = ~5B row reads"
        ),
        "sql_query": (
            "SELECT\n"
            "    u.email,\n"
            "    u.region,\n"
            "    (SELECT COUNT(*)\n"
            "     FROM orders o\n"
            "     WHERE o.customer_id = u.id AND o.status = 'completed') AS completed_orders,\n"
            "    (SELECT SUM(o.total)\n"
            "     FROM orders o\n"
            "     WHERE o.customer_id = u.id\n"
            "       AND o.created_at >= DATE '2024-01-01') AS ytd_spend,\n"
            "    (SELECT total\n"
            "     FROM orders o\n"
            "     WHERE o.customer_id = u.id\n"
            "     ORDER BY created_at DESC, id DESC LIMIT 1) AS last_order_amount\n"
            "FROM users u\n"
            "WHERE u.tier = 'premium';"
        ),
        "ground_truth_issues": [
            {
                "type": "correlated_subquery_count",
                "line": 4,
                "keywords": [
                    "correlated",
                    "subquery",
                    "per row",
                    "n+1",
                    "each user",
                    "repeated scan",
                    "join",
                    "aggregation",
                ],
            },
            {
                "type": "correlated_subquery_sum",
                "line": 7,
                "keywords": [
                    "correlated",
                    "subquery",
                    "per row",
                    "n+1",
                    "each user",
                    "repeated scan",
                    "join",
                    "group by",
                ],
            },
            {
                "type": "correlated_subquery_limit",
                "line": 11,
                "keywords": [
                    "correlated",
                    "subquery",
                    "limit 1",
                    "order by",
                    "lateral",
                    "row_number",
                    "rank",
                    "window function",
                    "per row",
                ],
            },
            {
                "type": "missing_aggregation_join",
                "line": 16,
                "keywords": [
                    "left join",
                    "group by",
                    "aggreg",
                    "single pass",
                    "coalesce",
                    "join aggregat",
                ],
            },
        ],
        "approved_expected": False,
    },
    # ─────────────────────────────────────────────────────────────────
    # TASK 3 — MEDIUM-HARD: Wildcard LIKE Full Scan
    # ─────────────────────────────────────────────────────────────────
    "task_3_wildcard_scan": {
        "task_id": "task_3_wildcard_scan",
        "task_name": "Wildcard LIKE & Projection Optimization",
        "task_description": (
            "The query scans all 1,000,000 events rows with leading-wildcard LIKE "
            "patterns that disable columnar zone-map pruning, plus a redundant OR branch "
            "that never matches any row. "
            "Your rewrite must return the SAME rows and columns: replace the "
            "leading-wildcard filter with exact equality (`event_type = 'purchase'`), "
            "which is provably equivalent here and lets DuckDB prune. "
            "Separately identify — but do NOT apply, since they would change the output — "
            "the SELECT * on a million-row table and the derived columns computed for "
            "every row."
        ),
        "difficulty": "medium-hard",
        "dialect": "duckdb/postgresql",
        "max_steps": 4,
        "schema_info": (
            "Table: events (1,000,000 rows)\n"
            "  id INT, user_id INT, session_id VARCHAR,\n"
            "  event_type VARCHAR, occurred_at DATE\n\n"
            "Distinct event_type values: purchase, view, click, signup, logout, search\n"
            "Only 'purchase' contains the substring 'purchase' and no event_type contains\n"
            "'buy', so the filter is equivalent to `event_type = 'purchase'`.\n"
            "Leading-wildcard LIKE on 1M rows forces a full scan; exact equality enables\n"
            "columnar zone-map pruning."
        ),
        "sql_query": (
            "SELECT\n"
            "    *,\n"
            "    CAST(id AS VARCHAR) || '_' || event_type  AS event_key,\n"
            "    upper(event_type)                          AS event_type_upper\n"
            "FROM events\n"
            "WHERE event_type LIKE '%purchase%'\n"
            "   OR event_type LIKE '%buy%';"
        ),
        "ground_truth_issues": [
            {
                "type": "leading_wildcard_like",
                "line": 6,
                "keywords": [
                    "leading wildcard",
                    "like '%",
                    "full scan",
                    "exact match",
                    "equality",
                    "pruning disabled",
                    "wildcard",
                    "zone map",
                ],
            },
            {
                "type": "or_expands_to_full_scan",
                "line": 7,
                "keywords": [
                    "or condition",
                    "union",
                    "separate queries",
                    "or expands",
                    "full scan",
                    "like '%buy%'",
                    "redundant",
                ],
            },
            {
                "type": "select_star_large_table",
                "line": 2,
                "keywords": [
                    "select *",
                    "1 million",
                    "all columns",
                    "projection",
                    "column pruning",
                    "unnecessary",
                    "bandwidth",
                ],
            },
            {
                "type": "pre_filter_computed_columns",
                "line": 3,
                "keywords": [
                    "computed column",
                    "derived",
                    "upper(",
                    "cast(",
                    "concatenat",
                    "before filter",
                    "pre-filter",
                    "push down",
                    "CTE",
                ],
            },
        ],
        "approved_expected": False,
    },
    # ─────────────────────────────────────────────────────────────────
    # TASK 4 — HARD: Implicit Cross Join + Repeated Scalar Subqueries
    # ─────────────────────────────────────────────────────────────────
    "task_4_implicit_join": {
        "task_id": "task_4_implicit_join",
        "task_name": "Implicit Cross Join & Scalar Subquery Elimination",
        "task_description": (
            "This query uses comma-separated FROM (implicit cross join syntax) and "
            "two uncorrelated scalar subqueries that compute global constants over the "
            "orders table (neither references the outer query). "
            "Identify: implicit cross join risk (comma in FROM clause), "
            "two uncorrelated scalar subqueries computing global stats that are clearer "
            "hoisted into a CTE, "
            "and the GROUP BY without an explicit JOIN. "
            "Rewrite using explicit INNER JOIN and a CTE for the global stats "
            "so the intent is explicit and they are computed once."
        ),
        "difficulty": "hard",
        "dialect": "duckdb/postgresql",
        "max_steps": 5,
        "schema_info": (
            "Table: users (10,000 rows)  — id, email, tier, region, plan, created_at\n"
            "Table: orders (500,000 rows) — id, customer_id, product_id, status, total, created_at\n\n"
            "Join: users.id = orders.customer_id\n"
            "Implicit join (comma syntax) risk: if WHERE predicate is missing,\n"
            "produces a Cartesian product of 10k × 500k = 5 BILLION rows.\n"
            "Scalar subqueries: uncorrelated global aggregates over all 500k orders;\n"
            "clearer and guaranteed single-eval when hoisted into a CTE."
        ),
        "sql_query": (
            "SELECT\n"
            "    u.region,\n"
            "    u.plan,\n"
            "    COUNT(*)      AS total_orders,\n"
            "    SUM(o.total)  AS revenue,\n"
            "    (SELECT AVG(total) FROM orders)                                    AS global_avg,\n"
            "    (SELECT MAX(total) FROM orders WHERE status = 'completed')         AS max_deal\n"
            "FROM users u, orders o\n"
            "WHERE u.id = o.customer_id\n"
            "  AND o.status IN ('completed', 'shipped')\n"
            "GROUP BY u.region, u.plan;"
        ),
        "ground_truth_issues": [
            {
                "type": "implicit_cross_join",
                "line": 8,
                "keywords": [
                    "implicit",
                    "cross join",
                    "comma join",
                    "explicit join",
                    "inner join",
                    "cartesian",
                    "comma in from",
                ],
            },
            {
                "type": "uncorrelated_scalar_subquery_avg",
                "line": 6,
                "keywords": [
                    "scalar subquery",
                    "uncorrelated",
                    "global constant",
                    "hoist",
                    "cte",
                    "with clause",
                    "pre-compute",
                    "global avg",
                ],
            },
            {
                "type": "uncorrelated_scalar_subquery_max",
                "line": 7,
                "keywords": [
                    "scalar subquery",
                    "uncorrelated",
                    "global constant",
                    "max deal",
                    "cte",
                    "pre-compute",
                    "compute once",
                    "constant",
                ],
            },
            {
                "type": "missing_explicit_join",
                "line": 8,
                "keywords": [
                    "inner join",
                    "explicit",
                    "on clause",
                    "join condition",
                    "readable",
                    "maintainable",
                    "ansi sql",
                ],
            },
        ],
        "approved_expected": False,
    },
    # ─────────────────────────────────────────────────────────────────
    # TASK 5 — EXPERT: Window Function Over Entire 1M-Row Table
    # ─────────────────────────────────────────────────────────────────
    "task_5_window_functions": {
        "task_id": "task_5_window_functions",
        "task_name": "Window Function & Full-Scan Audit",
        "task_description": (
            "Five window functions are computed over ALL 1,000,000 events rows. Three of "
            "them share `PARTITION BY user_id` and can be consolidated so that partition is "
            "built once instead of repeatedly. "
            "Your rewrite must return identical rows and columns: keep all output columns "
            "but consolidate the window functions that share `PARTITION BY user_id` "
            "(result-preserving). "
            "Separately identify — but do NOT apply, since they change the result — the "
            "absence of pre-filtering (a WHERE would drop rows) and the global "
            "RANK() OVER (ORDER BY occurred_at) that sorts the entire table."
        ),
        "difficulty": "expert",
        "dialect": "duckdb/postgresql",
        "max_steps": 5,
        "schema_info": (
            "Table: events (1,000,000 rows)\n"
            "  id INT, user_id INT, session_id VARCHAR,\n"
            "  event_type VARCHAR, occurred_at DATE\n\n"
            "Window function cost: each OVER() = full sort/hash pass over 1M rows\n"
            "5 window functions = 5 full passes before any filtering\n"
            "Global RANK(): sorts all 1M rows globally — most expensive operation\n"
            "Filtering to 'purchase' events first reduces dataset to ~167k rows (1/6)"
        ),
        "sql_query": (
            "SELECT\n"
            "    user_id,\n"
            "    event_type,\n"
            "    occurred_at,\n"
            "    COUNT(*) OVER (PARTITION BY user_id)                                AS total_user_events,\n"
            "    COUNT(*) OVER (PARTITION BY user_id, event_type)                   AS type_count,\n"
            "    ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY occurred_at DESC, id) AS recency_rank,\n"
            "    RANK() OVER (ORDER BY occurred_at DESC, id)                        AS global_rank,\n"
            "    SUM(CASE WHEN event_type = 'purchase' THEN 1 ELSE 0 END)\n"
            "        OVER (PARTITION BY user_id)                                    AS user_purchases\n"
            "FROM events;"
        ),
        "ground_truth_issues": [
            {
                "type": "no_pre_filter",
                "line": 11,
                "keywords": [
                    "no where",
                    "no filter",
                    "full table",
                    "1 million",
                    "all rows",
                    "pre-filter",
                    "filter first",
                    "cte",
                    "with clause",
                ],
            },
            {
                "type": "global_rank_no_partition",
                "line": 8,
                "keywords": [
                    "rank() over",
                    "global rank",
                    "no partition",
                    "entire table",
                    "full sort",
                    "expensive",
                    "global ordering",
                    "remove",
                ],
            },
            {
                "type": "redundant_window_functions",
                "line": 5,
                "keywords": [
                    "5 window",
                    "multiple over",
                    "redundant",
                    "merge",
                    "combine",
                    "single pass",
                    "same partition",
                    "consolidate",
                ],
            },
            {
                "type": "count_vs_conditional_sum",
                "line": 9,
                "keywords": [
                    "case when",
                    "sum case",
                    "count filter",
                    "filter clause",
                    "count(*) filter",
                    "simpler",
                    "merge with",
                ],
            },
        ],
        "approved_expected": False,
    },
}


def get_task_list():
    return [
        {
            "task_id": t["task_id"],
            "task_name": t["task_name"],
            "difficulty": t["difficulty"],
            "max_steps": t["max_steps"],
            "description": t["task_description"],
            "action_schema": {
                "suggestions": "List of {issue_type, line, description, severity, fix}",
                "optimized_query": "str — complete rewritten SQL (will be EXECUTED for real timing)",
                "summary": "str — overall performance analysis",
                "estimated_improvement": "str — expected speedup (e.g. '10x faster')",
                "approved": "bool — True if already optimal",
            },
        }
        for t in TASKS.values()
    ]
