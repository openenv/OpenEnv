# envs/finqa_env/server/tools.py
"""
Tool implementations for the FinQA environment.

Ported from FinQABenchmark with simplifications:
- Removed LangChain dependencies
- Added submit_answer tool for episode termination
"""

import json
import os
import re
import sqlite3
from typing import Any, Dict, List, Tuple

import pandas as pd


class FinQATools:
    """
    Tool implementations for financial QA tasks.

    Args:
        data_path: Path to the data directory containing benchmark_questions/ and input_companies/
    """

    def __init__(self, data_path: str):
        self.data_path = data_path
        self.companies_path = os.path.join(data_path, "input_companies")
        self._tables_cleaned = None

    @property
    def tables_cleaned(self) -> Dict:
        """Lazy load the cleaned tables metadata."""
        if self._tables_cleaned is None:
            tables_path = os.path.join(
                self.companies_path, "tables_cleaned_all_companies.json"
            )
            with open(tables_path, "r") as f:
                self._tables_cleaned = json.load(f)
        return self._tables_cleaned

    def get_available_companies(self) -> List[str]:
        """Get list of available company names."""
        return [
            d
            for d in os.listdir(self.companies_path)
            if os.path.isdir(os.path.join(self.companies_path, d))
        ]

    def _resolve_within(self, *parts: str) -> str | None:
        """
        Join agent-supplied path components under ``companies_path`` and confirm the
        result stays inside it.

        The tool arguments (``company_name``, ``table_name``) are supplied by the agent
        over the MCP boundary. Joining them into a filesystem path without a containment
        check allows ``..`` traversal to read files outside the data directory. This
        rejects any such escape.

        Args:
            parts (`str`):
                Path components to join beneath ``companies_path``.

        Returns:
            `str` or `None`: the resolved absolute path if it stays within
            ``companies_path``, otherwise `None`.
        """
        root = os.path.realpath(self.companies_path)
        candidate = os.path.realpath(os.path.join(self.companies_path, *parts))
        try:
            if candidate != root and os.path.commonpath([root, candidate]) == root:
                return candidate
        except ValueError:
            # Raised when paths cannot be compared (e.g. different drives on Windows).
            pass
        return None

    def execute_tool(
        self, tool_name: str, tool_args: Dict[str, Any]
    ) -> Tuple[str, bool]:
        """
        Execute a tool and return its result.

        Args:
            tool_name: Name of the tool to execute
            tool_args: Arguments for the tool

        Returns:
            Tuple of (result_string, is_final_answer)
        """
        if tool_name == "get_descriptions":
            return self.get_descriptions(**tool_args), False
        elif tool_name == "get_table_info":
            return self.get_table_info(**tool_args), False
        elif tool_name == "sql_query":
            return self.sql_query(**tool_args), False
        elif tool_name == "submit_answer":
            return self.submit_answer(**tool_args), True
        else:
            return f"Error: Unknown tool '{tool_name}'", False

    def get_descriptions(self, company_name: str) -> str:
        """
        Get a list of available table names for a company.

        Args:
            company_name: The name of the company

        Returns:
            JSON list of table names
        """
        company_path = self._resolve_within(company_name)

        if company_path is None or not os.path.isdir(company_path):
            available = self.get_available_companies()
            return (
                f"Error: '{company_name}' not found. Available companies: {available}"
            )

        # Get all JSON files (tables) for this company
        tables = []
        for f in os.listdir(company_path):
            if f.endswith(".json"):
                tables.append(f.replace(".json", ""))

        return json.dumps(tables)

    def get_table_info(self, company_name: str, table_name: str) -> str:
        """
        Get table metadata: description, columns, types, unique values.

        Args:
            company_name: The name of the company
            table_name: The name of the table

        Returns:
            JSON string with table metadata (description, columns, dtypes, unique values)
        """
        company_path = self._resolve_within(company_name)

        if company_path is None or not os.path.isdir(company_path):
            available = self.get_available_companies()
            return (
                f"Error: '{company_name}' not found. Available companies: {available}"
            )

        # Clean table name (remove .json or .txt if present)
        cleaned_table_name = table_name.replace(".json", "").replace(".txt", "")
        table_key = f"{company_name}/{cleaned_table_name}"

        if table_key not in self.tables_cleaned:
            return f"Error: Table '{table_name}' not found for company '{company_name}'"

        table_info = self.tables_cleaned[table_key].copy()

        # Load the actual table to get column info
        cleaned_table = pd.DataFrame(json.loads(table_info["table"]))

        # Drop numeric columns (keep only structure columns for querying hints)
        cols_to_drop = []
        for col in cleaned_table.columns.tolist()[1:]:  # Skip first column
            vals = cleaned_table[col].tolist()[1:]
            cleaned_vals = [
                "".join(char for char in str(x) if char.isalnum()).strip() for x in vals
            ]
            all_numeric = all(v.isnumeric() or len(v) == 0 for v in cleaned_vals)
            if all_numeric:
                cols_to_drop.append(col)

        table_info["column_dtypes"] = {
            col: str(cleaned_table[col].dtype) for col in cleaned_table.columns.tolist()
        }

        # Only show unique values for non-numeric columns
        cleaned_table_filtered = cleaned_table.drop(cols_to_drop, axis=1)
        table_info["unique_vals_per_col"] = {
            col: list(cleaned_table_filtered[col].unique())
            for col in cleaned_table_filtered.columns.tolist()
        }

        # Remove the raw table data from response
        del table_info["table"]

        return json.dumps(table_info, indent=0).replace("\n", "")

    def sql_query(self, company_name: str, table_name: str, query: str) -> str:
        """
        Execute a SQL query on a table. Select * not allowed (too inefficient).

        Filters are required to query: WHERE, HAVING, IN, NOT IN, EXISTS, NOT EXISTS, ANY, SOME, ALL, LIKE, NOT LIKE, BETWEEN, NOT BETWEEN, IS NULL, IS NOT NULL, CASE, FILTER.

        Args:
            company_name: The name of the company
            table_name: The name of the table
            query: SQL query to execute (must include filters)

        Returns:
            JSON string with query results
        """
        # Validate query has filters (prevent full table scans)
        if "select *" in query.lower():
            return "Error: SELECT * is not allowed (too inefficient)"

        sql_filters = [
            "WHERE",
            "HAVING",
            "IN",
            "NOT IN",
            "EXISTS",
            "NOT EXISTS",
            "ANY",
            "SOME",
            "ALL",
            "LIKE",
            "NOT LIKE",
            "BETWEEN",
            "NOT BETWEEN",
            "IS NULL",
            "IS NOT NULL",
            "CASE",
            "FILTER",
        ]

        query_upper = re.sub(r"(\r|\n|\t)+", " ", query).upper()
        pattern = (
            r"(?<!\w|\[)("
            + "|".join([re.escape(f) for f in sql_filters])
            + r")(?!\w|\])"
        )

        has_filter = (
            any(f" {filt} " in query_upper for filt in sql_filters)
            or len(re.findall(pattern, query_upper)) > 0
        )

        if not has_filter:
            return "Error: Query must include filters (WHERE, HAVING, etc.)"

        # Clean table name
        cleaned_table_name = table_name.replace(".txt", "").replace(".json", "")
        table_path = self._resolve_within(company_name, f"{cleaned_table_name}.json")

        if table_path is None or not os.path.isfile(table_path):
            return f"Error: Table file not found for '{company_name}/{table_name}'"

        try:
            # Load table and execute query
            conn = sqlite3.connect(":memory:")
            df = pd.read_json(table_path)
            df.to_sql(cleaned_table_name, conn, index=False, if_exists="replace")
            result = pd.read_sql_query(query, conn)
            conn.close()

            return result.to_json(orient="records")
        except Exception as e:
            return f"Error executing query: {str(e)}"

    def submit_answer(self, answer: str) -> str:
        """
        Submit a final answer for the question.

        Args:
            answer: The final answer to submit

        Returns:
            Confirmation message
        """
        return f"Answer submitted: {answer}"
