"""Unit tests for Agent World Model environment modules."""

import json
import os
import sqlite3
import sys
import tempfile

import pytest

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, "src"))

from envs.agent_world_model_env.models import AWMAction, AWMObservation
from envs.agent_world_model_env.server.data_loader import (
    _load_jsonl,
    AWMDataLoader,
    normalize_scenario_name,
)
from envs.agent_world_model_env.server.db_manager import (
    cleanup_session_dir,
    create_database,
    save_snapshot,
)
from envs.agent_world_model_env.server.verifier import (
    _normalize_azure_url,
    execute_code_verifier,
    execute_sql_verifier,
    run_verifier,
)


# ─── normalize_scenario_name ─────────────────────────────────────────


class TestNormalizeScenarioName:
    def test_simple(self):
        assert normalize_scenario_name("e_commerce_33") == "e_commerce_33"

    def test_uppercase(self):
        assert normalize_scenario_name("E_Commerce_33") == "e_commerce_33"

    def test_spaces_and_special(self):
        assert normalize_scenario_name("My App - v2!") == "my_app_v2"

    def test_multiple_underscores(self):
        assert normalize_scenario_name("foo___bar") == "foo_bar"

    def test_leading_trailing(self):
        assert normalize_scenario_name("__test__") == "test"


# ─── _load_jsonl ─────────────────────────────────────────────────────


class TestLoadJsonl:
    def test_basic(self, tmp_path):
        p = tmp_path / "test.jsonl"
        p.write_text('{"a": 1}\n{"b": 2}\n')
        result = _load_jsonl(str(p))
        assert len(result) == 2
        assert result[0] == {"a": 1}
        assert result[1] == {"b": 2}

    def test_empty_lines(self, tmp_path):
        p = tmp_path / "test.jsonl"
        p.write_text('{"a": 1}\n\n{"b": 2}\n\n')
        result = _load_jsonl(str(p))
        assert len(result) == 2


# ─── AWMDataLoader ───────────────────────────────────────────────────


@pytest.fixture
def mock_data_dir(tmp_path):
    """Create a minimal mock data directory with all required files."""

    scenarios = [
        {"name": "test_shop", "description": "A test shopping environment"},
        {"name": "test_bank", "description": "A test banking environment"},
    ]
    tasks = [
        {"scenario": "test_shop", "tasks": ["Buy a widget", "Search for gizmos"]},
        {"scenario": "test_bank", "tasks": ["Check balance"]},
    ]
    envs = [
        {"scenario": "test_shop", "full_code": "# test shop code\nprint('hello')"},
        {"scenario": "test_bank", "full_code": "# test bank code\nprint('bank')"},
    ]
    db_schemas = [
        {
            "scenario": "test_shop",
            "db_schema": {
                "tables": [
                    {
                        "ddl": "CREATE TABLE products (id INTEGER PRIMARY KEY, name TEXT, price REAL)",
                        "indexes": [],
                    }
                ]
            },
        },
        {
            "scenario": "test_bank",
            "db_schema": {
                "tables": [
                    {
                        "ddl": "CREATE TABLE accounts (id INTEGER PRIMARY KEY, balance REAL)",
                        "indexes": [],
                    }
                ]
            },
        },
    ]
    samples = [
        {
            "scenario": "test_shop",
            "sample_data": [
                {
                    "table_name": "products",
                    "insert_statements": [
                        "INSERT INTO products (id, name, price) VALUES (1, 'Widget', 9.99)",
                        "INSERT INTO products (id, name, price) VALUES (2, 'Gizmo', 19.99)",
                    ],
                }
            ],
        },
        {
            "scenario": "test_bank",
            "sample_data": [
                {
                    "table_name": "accounts",
                    "insert_statements": [
                        "INSERT INTO accounts (id, balance) VALUES (1, 1000.00)",
                    ],
                }
            ],
        },
    ]
    # SQL verifiers (gen_verifier.jsonl)
    sql_verifiers = [
        {
            "scenario": "test_shop",
            "task_idx": 0,
            "verification": {
                "code": (
                    "def verify_task(initial_db_path, final_db_path):\n"
                    "    import sqlite3\n"
                    "    conn = sqlite3.connect(final_db_path)\n"
                    "    cursor = conn.cursor()\n"
                    "    cursor.execute('SELECT COUNT(*) FROM products')\n"
                    "    count = cursor.fetchone()[0]\n"
                    "    conn.close()\n"
                    "    return {'product_count': count}\n"
                ),
            },
        },
        {
            "scenario": "test_shop",
            "task_idx": 1,
            "verification": {
                "code": (
                    "def verify_task(initial_db_path, final_db_path):\n"
                    "    import sqlite3\n"
                    "    conn = sqlite3.connect(initial_db_path)\n"
                    "    cursor = conn.cursor()\n"
                    "    cursor.execute('SELECT COUNT(*) FROM products')\n"
                    "    initial_count = cursor.fetchone()[0]\n"
                    "    conn.close()\n"
                    "    conn = sqlite3.connect(final_db_path)\n"
                    "    cursor = conn.cursor()\n"
                    "    cursor.execute('SELECT COUNT(*) FROM products')\n"
                    "    final_count = cursor.fetchone()[0]\n"
                    "    conn.close()\n"
                    "    return {'initial': initial_count, 'final': final_count}\n"
                ),
            },
        },
    ]
    # Code verifiers (gen_verifier.pure_code.jsonl)
    code_verifiers = [
        {
            "scenario": "test_shop",
            "task_idx": 0,
            "verification": {
                "code": (
                    "def verify_task_completion(initial_db_path, final_db_path, final_answer):\n"
                    "    import sqlite3\n"
                    "    conn = sqlite3.connect(final_db_path)\n"
                    "    cursor = conn.cursor()\n"
                    "    cursor.execute('SELECT COUNT(*) FROM products')\n"
                    "    count = cursor.fetchone()[0]\n"
                    "    conn.close()\n"
                    "    if count > 2:\n"
                    "        return {'result': 'complete'}\n"
                    "    return {'result': 'others'}\n"
                ),
            },
        },
        {
            "scenario": "test_shop",
            "task_idx": 1,
            "verification": {
                "code": (
                    "def verify_task_completion(initial_db_path, final_db_path, final_answer):\n"
                    "    return {'result': 'complete'}\n"
                ),
            },
        },
    ]

    for name, data in [
        ("gen_scenario.jsonl", scenarios),
        ("gen_tasks.jsonl", tasks),
        ("gen_envs.jsonl", envs),
        ("gen_db.jsonl", db_schemas),
        ("gen_sample.jsonl", samples),
        ("gen_verifier.jsonl", sql_verifiers),
        ("gen_verifier.pure_code.jsonl", code_verifiers),
    ]:
        with open(f"{tmp_path}/{name}", "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")

    return str(tmp_path)


class TestAWMDataLoader:
    def test_list_scenarios(self, mock_data_dir):
        loader = AWMDataLoader(cache_dir=mock_data_dir)
        scenarios = loader.list_scenarios()
        assert len(scenarios) == 2
        names = {s["name"] for s in scenarios}
        assert "test_shop" in names
        assert "test_bank" in names

    def test_list_scenarios_has_tasks(self, mock_data_dir):
        loader = AWMDataLoader(cache_dir=mock_data_dir)
        scenarios = loader.list_scenarios()
        shop = next(s for s in scenarios if s["name"] == "test_shop")
        assert shop["num_tasks"] == 2
        assert "Buy a widget" in shop["tasks"]

    def test_scenario_exists(self, mock_data_dir):
        loader = AWMDataLoader(cache_dir=mock_data_dir)
        assert loader.scenario_exists("test_shop") is True
        assert loader.scenario_exists("nonexistent") is False

    def test_get_env_code(self, mock_data_dir):
        loader = AWMDataLoader(cache_dir=mock_data_dir)
        code = loader.get_env_code("test_shop")
        assert "test shop code" in code

    def test_get_env_code_not_found(self, mock_data_dir):
        loader = AWMDataLoader(cache_dir=mock_data_dir)
        with pytest.raises(ValueError, match="not found"):
            loader.get_env_code("nonexistent")

    def test_get_tasks(self, mock_data_dir):
        loader = AWMDataLoader(cache_dir=mock_data_dir)
        tasks = loader.get_tasks("test_shop")
        assert len(tasks) == 2
        assert tasks[0] == "Buy a widget"

    def test_get_tasks_empty(self, mock_data_dir):
        loader = AWMDataLoader(cache_dir=mock_data_dir)
        tasks = loader.get_tasks("nonexistent")
        assert tasks == []

    def test_get_verifier(self, mock_data_dir):
        loader = AWMDataLoader(cache_dir=mock_data_dir)
        v = loader.get_verifier("test_shop", 0, "sql")
        assert v is not None
        assert "verification" in v

    def test_get_verifier_not_found(self, mock_data_dir):
        loader = AWMDataLoader(cache_dir=mock_data_dir)
        v = loader.get_verifier("test_shop", 99, "sql")
        assert v is None

    def test_get_db_schema(self, mock_data_dir):
        loader = AWMDataLoader(cache_dir=mock_data_dir)
        schema = loader.get_db_schema("test_shop")
        assert "tables" in schema
        assert len(schema["tables"]) == 1

    def test_get_sample_data(self, mock_data_dir):
        loader = AWMDataLoader(cache_dir=mock_data_dir)
        data = loader.get_sample_data("test_shop")
        assert len(data) == 1
        assert data[0]["table_name"] == "products"


# ─── db_manager ──────────────────────────────────────────────────────


class TestDBManager:
    def test_create_database(self, mock_data_dir):
        loader = AWMDataLoader(cache_dir=mock_data_dir)
        schema = loader.get_db_schema("test_shop")
        sample = loader.get_sample_data("test_shop")

        with tempfile.TemporaryDirectory() as td:
            db_path = f"{td}/test.db"
            create_database(db_path, schema, sample)

            assert os.path.exists(db_path)

            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM products")
            count = cursor.fetchone()[0]
            conn.close()

            assert count == 2

    def test_save_snapshot(self, mock_data_dir):
        loader = AWMDataLoader(cache_dir=mock_data_dir)
        schema = loader.get_db_schema("test_shop")
        sample = loader.get_sample_data("test_shop")

        with tempfile.TemporaryDirectory() as td:
            db_path = f"{td}/test.db"
            snapshot_path = f"{td}/test_initial.db"
            create_database(db_path, schema, sample)
            save_snapshot(db_path, snapshot_path)

            assert os.path.exists(snapshot_path)

            conn = sqlite3.connect(db_path)
            conn.execute(
                "INSERT INTO products (id, name, price) VALUES (3, 'New', 5.0)"
            )
            conn.commit()
            conn.close()

            conn = sqlite3.connect(snapshot_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM products")
            count = cursor.fetchone()[0]
            conn.close()
            assert count == 2

    def test_cleanup_session_dir(self):
        td = tempfile.mkdtemp(prefix="awm_test_")
        with open(f"{td}/test.txt", "w") as f:
            f.write("hello")
        cleanup_session_dir(td)
        assert not os.path.exists(td)

    def test_cleanup_nonexistent_dir(self):
        cleanup_session_dir("/nonexistent/path/awm_cleanup_test")


# ─── verifier ────────────────────────────────────────────────────────


class TestVerifier:
    def _setup_db(self, td, n_products=2):
        schema = {
            "tables": [
                {
                    "ddl": "CREATE TABLE products (id INTEGER PRIMARY KEY, name TEXT, price REAL)",
                    "indexes": [],
                }
            ]
        }
        sample = [
            {
                "table_name": "products",
                "insert_statements": [
                    f"INSERT INTO products (id, name, price) VALUES ({i}, 'Item{i}', {i * 10.0})"
                    for i in range(1, n_products + 1)
                ],
            }
        ]

        initial_path = f"{td}/initial.db"
        final_path = f"{td}/final.db"
        create_database(initial_path, schema, sample)
        create_database(final_path, schema, sample)
        return initial_path, final_path

    def test_code_verifier_complete(self):
        code = (
            "def verify_task_completion(initial_db_path, final_db_path, final_answer):\n"
            "    return {'result': 'complete'}\n"
        )
        with tempfile.TemporaryDirectory() as td:
            initial, final = self._setup_db(td)
            result = execute_code_verifier(
                code, "verify_task_completion", initial, final
            )
            assert result["result"] == "complete"
            assert result["execution_status"] == "success"

    def test_code_verifier_others(self):
        code = (
            "def verify_task_completion(initial_db_path, final_db_path, final_answer):\n"
            "    return {'result': 'others'}\n"
        )
        with tempfile.TemporaryDirectory() as td:
            initial, final = self._setup_db(td)
            result = execute_code_verifier(
                code, "verify_task_completion", initial, final
            )
            assert result["result"] == "others"

    def test_code_verifier_bad_function(self):
        code = "def wrong_name():\n    return {'result': 'complete'}\n"
        with tempfile.TemporaryDirectory() as td:
            initial, final = self._setup_db(td)
            result = execute_code_verifier(
                code, "verify_task_completion", initial, final
            )
            assert result["execution_status"] == "error"
            assert result["result"] == "others"

    def test_code_verifier_db_query(self):
        code = (
            "def verify_task_completion(initial_db_path, final_db_path, final_answer):\n"
            "    import sqlite3\n"
            "    conn = sqlite3.connect(final_db_path)\n"
            "    cursor = conn.cursor()\n"
            "    cursor.execute('SELECT COUNT(*) FROM products')\n"
            "    count = cursor.fetchone()[0]\n"
            "    conn.close()\n"
            "    if count == 2:\n"
            "        return {'result': 'complete'}\n"
            "    return {'result': 'others'}\n"
        )
        with tempfile.TemporaryDirectory() as td:
            initial, final = self._setup_db(td)
            result = execute_code_verifier(
                code, "verify_task_completion", initial, final
            )
            assert result["result"] == "complete"

    def test_code_verifier_2arg_signature(self):
        """Bug #8: verifier with 2-arg signature (no final_answer) should work."""
        code = (
            "def verify_task(initial_db_path, final_db_path):\n"
            "    return {'result': 'complete'}\n"
        )
        with tempfile.TemporaryDirectory() as td:
            initial, final = self._setup_db(td)
            result = execute_code_verifier(code, "verify_task", initial, final)
            assert result["result"] == "complete"
            assert result["execution_status"] == "success"

    def test_sql_verifier(self):
        code = (
            "def verify_task(initial_db_path, final_db_path):\n"
            "    import sqlite3\n"
            "    conn = sqlite3.connect(final_db_path)\n"
            "    cursor = conn.cursor()\n"
            "    cursor.execute('SELECT COUNT(*) FROM products')\n"
            "    count = cursor.fetchone()[0]\n"
            "    conn.close()\n"
            "    return {'product_count': count}\n"
        )
        with tempfile.TemporaryDirectory() as td:
            initial, final = self._setup_db(td)
            result = execute_sql_verifier(code, "verify_task", initial, final)
            assert result["product_count"] == 2

    def test_sql_verifier_missing_db(self):
        code = "def verify_task(initial_db_path, final_db_path):\n    return {}\n"
        result = execute_sql_verifier(
            code, "verify_task", "/nonexistent/a.db", "/nonexistent/b.db"
        )
        assert result["execution_status"] == "error"

    def test_run_verifier_code_mode(self):
        verifier_entry = {
            "verification": {
                "code": (
                    "def verify_task_completion(initial_db_path, final_db_path, final_answer):\n"
                    "    return {'result': 'complete'}\n"
                ),
            },
        }
        with tempfile.TemporaryDirectory() as td:
            initial, final = self._setup_db(td)
            reward_type, result = run_verifier(verifier_entry, "code", initial, final)
            assert reward_type == "complete"

    def test_run_verifier_code_mode_2arg(self):
        """Bug #8: run_verifier in code mode with 2-arg function name."""
        verifier_entry = {
            "verification": {
                "code": (
                    "def verify_task(initial_db_path, final_db_path):\n"
                    "    return {'result': 'complete'}\n"
                ),
            },
        }
        with tempfile.TemporaryDirectory() as td:
            initial, final = self._setup_db(td)
            reward_type, result = run_verifier(verifier_entry, "code", initial, final)
            assert reward_type == "complete"

    def test_run_verifier_no_code(self):
        verifier_entry = {"verification": {"code": ""}}
        with tempfile.TemporaryDirectory() as td:
            initial, final = self._setup_db(td)
            reward_type, result = run_verifier(verifier_entry, "code", initial, final)
            assert reward_type == "judge_error"


# ─── Azure URL normalization ─────────────────────────────────────────


class TestAzureURLNormalization:
    def test_azure_url_without_suffix(self):
        url = "https://sfc-ml-sweden.openai.azure.com/"
        assert (
            _normalize_azure_url(url)
            == "https://sfc-ml-sweden.openai.azure.com/openai/v1"
        )

    def test_azure_url_with_suffix(self):
        url = "https://sfc-ml-sweden.openai.azure.com/openai/v1/"
        assert (
            _normalize_azure_url(url)
            == "https://sfc-ml-sweden.openai.azure.com/openai/v1/"
        )

    def test_non_azure_url_unchanged(self):
        url = "https://api.openai.com/v1"
        assert _normalize_azure_url(url) == url

    def test_azure_url_no_trailing_slash(self):
        url = "https://sfc-ml-sweden.openai.azure.com"
        assert (
            _normalize_azure_url(url)
            == "https://sfc-ml-sweden.openai.azure.com/openai/v1"
        )


# ─── scenario_manager (unit-testable parts) ──────────────────────────


class TestScenarioManagerHelpers:
    def test_get_random_port(self):
        from envs.agent_world_model_env.server.scenario_manager import _get_random_port

        port = _get_random_port()
        assert 1024 <= port <= 65535

    def test_patch_env_code_replaces_create_engine(self):
        from envs.agent_world_model_env.server.scenario_manager import _patch_env_code

        code = (
            "from sqlalchemy import create_engine\n"
            "engine = create_engine('sqlite:///old.db')\n"
            "if __name__ == '__main__':\n"
            "    uvicorn.run(app, host='0.0.0.0', port=8000)\n"
        )
        patched = _patch_env_code(code, "/tmp/new.db", "127.0.0.1", 9999)
        assert "sqlite:////tmp/new.db" in patched
        assert "port=9999" in patched
        assert "FastApiMCP" in patched

    def test_scenario_process_initial_state(self):
        from envs.agent_world_model_env.server.scenario_manager import ScenarioProcess

        proc = ScenarioProcess()
        assert proc.port is None
        assert proc.mcp_url is None
        assert proc.is_running is False


# ─── AWMObservation serialization ────────────────────────────────────


class TestAWMObservation:
    def test_top_level_fields_survive_model_dump(self):
        """Bug #4 workaround: fields should be in model_dump excluding metadata."""
        obs = AWMObservation(
            done=False,
            reward=None,
            reward_type="tool_call_ok",
            tool_name="search",
            tool_result="found items",
        )
        dumped = obs.model_dump(exclude={"reward", "done", "metadata"})
        assert dumped["reward_type"] == "tool_call_ok"
        assert dumped["tool_name"] == "search"
        assert dumped["tool_result"] == "found items"

    def test_observation_with_verify_result(self):
        obs = AWMObservation(
            done=True,
            reward=None,
            reward_type="complete",
            verify_result={"product_count": 3},
            scenario="test_shop",
            task="Buy something",
            steps_taken=5,
        )
        dumped = obs.model_dump(exclude={"reward", "done", "metadata"})
        assert dumped["reward_type"] == "complete"
        assert dumped["verify_result"]["product_count"] == 3
        assert dumped["scenario"] == "test_shop"
        assert dumped["steps_taken"] == 5

    def test_none_fields_excluded_by_default(self):
        """Bug #23: None-valued fields should not appear in model_dump output."""
        obs = AWMObservation(done=False, reward=None, reward_type="reset_ok")
        dumped = obs.model_dump(exclude={"reward", "done", "metadata"})
        assert dumped["reward_type"] == "reset_ok"
        assert "tool_name" not in dumped
        assert "error" not in dumped

    def test_none_fields_included_when_requested(self):
        """Caller can explicitly override exclude_none=False."""
        obs = AWMObservation(done=False, reward=None, reward_type="reset_ok")
        dumped = obs.model_dump(
            exclude={"reward", "done", "metadata"}, exclude_none=False
        )
        assert dumped["tool_name"] is None
        assert dumped["error"] is None

    def test_tool_name_present_when_set(self):
        """Bug #23: tool_name should appear when it has a value."""
        obs = AWMObservation(
            done=False, reward=None, reward_type="tool_call_ok", tool_name="search"
        )
        dumped = obs.model_dump(exclude={"reward", "done", "metadata"})
        assert "tool_name" in dumped
        assert dumped["tool_name"] == "search"

    def test_has_verifier_bool_converted_to_dict(self):
        """Legacy bool has_verifier should be converted to dict format."""
        obs = AWMObservation(
            done=False, reward=None, reward_type="reset_ok", has_verifier=True
        )
        assert obs.has_verifier == {"sql": True, "code": True}

        obs2 = AWMObservation(
            done=False, reward=None, reward_type="reset_ok", has_verifier=False
        )
        assert obs2.has_verifier is None

    def test_has_verifier_dict_preserved(self):
        """Dict has_verifier should be preserved as-is."""
        obs = AWMObservation(
            done=False,
            reward=None,
            reward_type="reset_ok",
            has_verifier={"sql": True, "code": False},
        )
        assert obs.has_verifier == {"sql": True, "code": False}


# ─── AWMAction union ────────────────────────────────────────────────


class TestAWMAction:
    def test_deserialize_call_tool(self):
        """Bug #24: AWMAction.model_validate should return CallToolAction."""
        action = AWMAction.model_validate(
            {"type": "call_tool", "tool_name": "search", "arguments": {"q": "test"}}
        )
        from openenv.core.env_server.mcp_types import CallToolAction

        assert isinstance(action, CallToolAction)
        assert action.tool_name == "search"

    def test_deserialize_list_tools(self):
        """Bug #24: AWMAction.model_validate should return ListToolsAction."""
        action = AWMAction.model_validate({"type": "list_tools"})
        from openenv.core.env_server.mcp_types import ListToolsAction

        assert isinstance(action, ListToolsAction)

    def test_model_json_schema(self):
        """Bug #24: AWMAction should expose a valid JSON schema."""
        schema = AWMAction.model_json_schema()
        assert isinstance(schema, dict)

    def test_invalid_type_rejected(self):
        """Bug #24: unknown action type should raise ValidationError."""
        import pydantic

        with pytest.raises(pydantic.ValidationError):
            AWMAction.model_validate({"type": "unknown_action"})


# ─── AWMEnvironment (no subprocess needed) ───────────────────────────


class TestAWMEnvironmentUnit:
    def test_reset_without_scenario(self, mock_data_dir):
        from envs.agent_world_model_env.server.awm_environment import AWMEnvironment

        loader = AWMDataLoader(cache_dir=mock_data_dir)
        env = AWMEnvironment(data_loader=loader)
        obs = env.reset()
        assert isinstance(obs, AWMObservation)
        assert obs.reward_type == "reset_error"
        assert "required" in obs.error

    def test_reset_nonexistent_scenario(self, mock_data_dir):
        from envs.agent_world_model_env.server.awm_environment import AWMEnvironment

        loader = AWMDataLoader(cache_dir=mock_data_dir)
        env = AWMEnvironment(data_loader=loader)
        obs = env.reset(scenario="nonexistent")
        assert obs.reward_type == "reset_error"
        assert "not found" in obs.error

    def test_reset_invalid_task_idx(self, mock_data_dir):
        from envs.agent_world_model_env.server.awm_environment import AWMEnvironment

        loader = AWMDataLoader(cache_dir=mock_data_dir)
        env = AWMEnvironment(data_loader=loader)
        obs = env.reset(scenario="test_shop", task_idx=99)
        assert obs.reward_type == "reset_error"
        assert "out of range" in obs.error

    def test_list_scenarios_tool(self, mock_data_dir):
        from envs.agent_world_model_env.server.awm_environment import AWMEnvironment
        from openenv.core.env_server.mcp_types import CallToolAction

        loader = AWMDataLoader(cache_dir=mock_data_dir)
        env = AWMEnvironment(data_loader=loader)
        obs = env.step(CallToolAction(tool_name="__list_scenarios__"))
        assert isinstance(obs, AWMObservation)
        assert obs.reward_type == "tool_call_ok"
        assert obs.total == 2
        names = {s["name"] for s in obs.scenarios}
        assert "test_shop" in names

    def test_reset_custom_reward_config(self, mock_data_dir):
        """Custom reward_config should be used when provided."""
        from envs.agent_world_model_env.server.awm_environment import AWMEnvironment

        loader = AWMDataLoader(cache_dir=mock_data_dir)
        env = AWMEnvironment(data_loader=loader)
        custom_config = {"complete": 2.0, "incomplete": 0.5}
        env.reset(scenario="test_shop", reward_config=custom_config)
        assert env._reward_config == custom_config

    def test_step_without_reset(self, mock_data_dir):
        from envs.agent_world_model_env.server.awm_environment import AWMEnvironment
        from openenv.core.env_server.mcp_types import CallToolAction

        loader = AWMDataLoader(cache_dir=mock_data_dir)
        env = AWMEnvironment(data_loader=loader)
        obs = env.step(CallToolAction(tool_name="some_tool", arguments={}))
        assert obs.reward_type == "server_error"

    def test_done_without_reset(self, mock_data_dir):
        """Bug #11: done should fail gracefully if reset was never called."""
        from envs.agent_world_model_env.server.awm_environment import AWMEnvironment
        from openenv.core.env_server.mcp_types import CallToolAction

        loader = AWMDataLoader(cache_dir=mock_data_dir)
        env = AWMEnvironment(data_loader=loader)
        obs = env.step(CallToolAction(tool_name="done"))
        assert obs.done is True
        assert obs.reward_type == "server_error"
        assert "not initialized" in obs.error

    def test_done_without_task(self, mock_data_dir):
        from envs.agent_world_model_env.server.awm_environment import AWMEnvironment
        from openenv.core.env_server.mcp_types import CallToolAction

        loader = AWMDataLoader(cache_dir=mock_data_dir)
        env = AWMEnvironment(data_loader=loader)
        env._scenario = "test_shop"
        env._task = None
        env._task_idx = None
        env._reset_ok = True
        obs = env.step(CallToolAction(tool_name="done"))
        assert obs.done is True
        assert obs.reward_type == "episode_done"
        assert obs.reward == 0.0

    def test_close_no_error(self, mock_data_dir):
        from envs.agent_world_model_env.server.awm_environment import AWMEnvironment

        loader = AWMDataLoader(cache_dir=mock_data_dir)
        env = AWMEnvironment(data_loader=loader)
        env.close()

    def test_verify_invalid_verifier_mode(self, mock_data_dir):
        """Invalid verifier_mode in verify tool should be rejected."""
        from envs.agent_world_model_env.server.awm_environment import AWMEnvironment
        from openenv.core.env_server.mcp_types import CallToolAction

        loader = AWMDataLoader(cache_dir=mock_data_dir)
        env = AWMEnvironment(data_loader=loader)
        env._scenario = "test_shop"
        env._task = "Buy a widget"
        env._task_idx = 0
        env._reset_ok = True
        env._has_verifier = {"sql": True, "code": True}
        obs = env.step(
            CallToolAction(
                tool_name="verify", arguments={"verifier_mode": "not_a_mode"}
            )
        )
        assert obs.reward_type == "invalid_args"
        assert "Invalid verifier_mode" in obs.error

    def test_valid_verifier_modes(self, mock_data_dir):
        """'sql' and 'code' should be accepted verifier modes."""
        from envs.agent_world_model_env.server.awm_environment import (
            VALID_VERIFIER_MODES,
        )

        assert VALID_VERIFIER_MODES == {"sql", "code"}

    def test_step_blocked_after_done(self, mock_data_dir):
        """Bug #21: step() should be blocked after done returns done=True."""
        from envs.agent_world_model_env.server.awm_environment import AWMEnvironment
        from openenv.core.env_server.mcp_types import CallToolAction

        loader = AWMDataLoader(cache_dir=mock_data_dir)
        env = AWMEnvironment(data_loader=loader)
        env._scenario = "test_shop"
        env._task = None
        env._task_idx = None
        env._reset_ok = True
        obs = env.step(CallToolAction(tool_name="done"))
        assert obs.done is True
        assert obs.reward_type == "episode_done"

        obs2 = env.step(CallToolAction(tool_name="some_tool", arguments={}))
        assert obs2.done is True
        assert obs2.reward_type == "episode_already_done"
        assert "reset()" in obs2.error

    def test_step_blocked_after_done_list_tools(self, mock_data_dir):
        """Bug #21: even list_tools should be blocked after done."""
        from envs.agent_world_model_env.server.awm_environment import AWMEnvironment
        from openenv.core.env_server.mcp_types import CallToolAction, ListToolsAction

        loader = AWMDataLoader(cache_dir=mock_data_dir)
        env = AWMEnvironment(data_loader=loader)
        env._scenario = "test_shop"
        env._task = None
        env._task_idx = None
        env._reset_ok = True
        env.step(CallToolAction(tool_name="done"))

        obs = env.step(ListToolsAction())
        assert obs.done is True
        assert obs.reward_type == "episode_already_done"

    def test_done_double_call_blocked(self, mock_data_dir):
        """Bug #21: calling done twice should return episode_already_done."""
        from envs.agent_world_model_env.server.awm_environment import AWMEnvironment
        from openenv.core.env_server.mcp_types import CallToolAction

        loader = AWMDataLoader(cache_dir=mock_data_dir)
        env = AWMEnvironment(data_loader=loader)
        env._scenario = "test_shop"
        env._task = None
        env._task_idx = None
        env._reset_ok = True
        obs = env.step(CallToolAction(tool_name="done"))
        assert obs.reward_type == "episode_done"

        obs2 = env.step(CallToolAction(tool_name="done"))
        assert obs2.done is True
        assert obs2.reward_type == "episode_already_done"

    def test_episode_done_resets_on_new_reset(self, mock_data_dir):
        """After done, a new reset() should unblock step()."""
        from envs.agent_world_model_env.server.awm_environment import AWMEnvironment
        from openenv.core.env_server.mcp_types import CallToolAction

        loader = AWMDataLoader(cache_dir=mock_data_dir)
        env = AWMEnvironment(data_loader=loader)
        env._scenario = "test_shop"
        env._task = None
        env._task_idx = None
        env._reset_ok = True
        env.step(CallToolAction(tool_name="done"))
        assert env._episode_done is True

        env.reset(scenario="test_shop")
        assert env._episode_done is False

    def test_subprocess_stopped_after_done(self, mock_data_dir):
        """Bug #22: subprocess should be stopped after done."""
        from unittest.mock import MagicMock

        from envs.agent_world_model_env.server.awm_environment import AWMEnvironment
        from openenv.core.env_server.mcp_types import CallToolAction

        loader = AWMDataLoader(cache_dir=mock_data_dir)
        env = AWMEnvironment(data_loader=loader)
        env._scenario = "test_shop"
        env._task = None
        env._task_idx = None
        env._reset_ok = True
        env._process = MagicMock()
        env._process.is_running = True

        env.step(CallToolAction(tool_name="done"))
        env._process.stop.assert_called_once()

    def test_has_verifier_none_for_empty_code(self, mock_data_dir):
        """Verifier with empty/trivial code should set has_verifier=None."""
        from envs.agent_world_model_env.server.awm_environment import AWMEnvironment

        loader = AWMDataLoader(cache_dir=mock_data_dir)

        # Override get_verifier to return empty code for both modes
        def mock_get_verifier(s, i, m):
            return {"verification": {"code": ""}}

        loader.get_verifier = mock_get_verifier
        env = AWMEnvironment(data_loader=loader)
        env.reset(scenario="test_shop", task_idx=0)
        # Both sql and code have empty verifiers, so has_verifier should be None
        assert env._has_verifier is None

    def test_has_verifier_dict_with_available_modes(self, mock_data_dir):
        """has_verifier should be a dict showing which modes are available."""
        from envs.agent_world_model_env.server.awm_environment import AWMEnvironment

        loader = AWMDataLoader(cache_dir=mock_data_dir)
        env = AWMEnvironment(data_loader=loader)
        # Using mock_data_dir which has both sql and code verifiers for test_shop task 0
        env.reset(scenario="test_shop", task_idx=0)
        assert env._has_verifier is not None
        assert isinstance(env._has_verifier, dict)
        assert "sql" in env._has_verifier
        assert "code" in env._has_verifier

    def test_verify_without_task(self, mock_data_dir):
        """verify should fail if no task was specified at reset."""
        from envs.agent_world_model_env.server.awm_environment import AWMEnvironment
        from openenv.core.env_server.mcp_types import CallToolAction

        loader = AWMDataLoader(cache_dir=mock_data_dir)
        env = AWMEnvironment(data_loader=loader)
        env._scenario = "test_shop"
        env._task = None
        env._task_idx = None
        env._reset_ok = True
        obs = env.step(
            CallToolAction(tool_name="verify", arguments={"verifier_mode": "code"})
        )
        assert obs.reward_type == "no_verifier"
        assert "no task" in obs.error.lower()

    def test_verify_without_reset(self, mock_data_dir):
        """verify should fail if reset was not called."""
        from envs.agent_world_model_env.server.awm_environment import AWMEnvironment
        from openenv.core.env_server.mcp_types import CallToolAction

        loader = AWMDataLoader(cache_dir=mock_data_dir)
        env = AWMEnvironment(data_loader=loader)
        obs = env.step(
            CallToolAction(tool_name="verify", arguments={"verifier_mode": "code"})
        )
        assert obs.reward_type == "server_error"
        assert "not initialized" in obs.error

    def test_error_classification(self):
        from envs.agent_world_model_env.server.awm_environment import (
            _classify_tool_error,
        )

        assert _classify_tool_error("Tool 'xyz' not found") == "tool_not_found"
        assert _classify_tool_error("Unknown tool: abc") == "tool_not_found"
        assert (
            _classify_tool_error(
                "Input validation error: 'country' is a required property"
            )
            == "invalid_args"
        )
        assert _classify_tool_error("Invalid argument: x") == "invalid_args"
        assert _classify_tool_error("missing required field") == "invalid_args"
        assert _classify_tool_error("Operation timed out") == "timeout"
        assert _classify_tool_error("Connection refused") == "server_error"

    def test_get_reward_default_config(self, mock_data_dir):
        """Test _get_reward with default reward config."""
        from envs.agent_world_model_env.server.awm_environment import AWMEnvironment

        loader = AWMDataLoader(cache_dir=mock_data_dir)
        env = AWMEnvironment(data_loader=loader)
        env.reset(scenario="test_shop")

        # Default config: complete=1.0, incomplete=0.1, format_error=-1.0
        assert env._get_reward("complete") == 1.0
        assert env._get_reward("incomplete") == 0.1
        assert env._get_reward("format_error") == -1.0
        # Format error types map to format_error
        assert env._get_reward("tool_not_found") == -1.0
        assert env._get_reward("invalid_args") == -1.0
        # Unknown types default to 0.0
        assert env._get_reward("server_error") == 0.0
        assert env._get_reward("unknown_type") == 0.0

    def test_get_reward_custom_config(self, mock_data_dir):
        """Test _get_reward with custom reward config."""
        from envs.agent_world_model_env.server.awm_environment import AWMEnvironment

        loader = AWMDataLoader(cache_dir=mock_data_dir)
        env = AWMEnvironment(data_loader=loader)
        custom_config = {"complete": 5.0, "incomplete": 0.5, "server_error": -0.5}
        env.reset(scenario="test_shop", reward_config=custom_config)

        assert env._get_reward("complete") == 5.0
        assert env._get_reward("incomplete") == 0.5
        assert env._get_reward("server_error") == -0.5
        # format_error not in custom config, defaults to 0.0
        assert env._get_reward("format_error") == 0.0

    def test_reset_with_llm_credentials(self, mock_data_dir):
        """LLM credentials passed at reset should be stored for sql verifier."""
        from envs.agent_world_model_env.server.awm_environment import AWMEnvironment

        loader = AWMDataLoader(cache_dir=mock_data_dir)
        env = AWMEnvironment(data_loader=loader)
        env.reset(
            scenario="test_shop",
            task_idx=0,
            llm_base_url="https://test.openai.azure.com/",
            llm_api_key="test-api-key-12345",
            llm_model="gpt-4o",
        )
        assert env._llm_base_url == "https://test.openai.azure.com/"
        assert env._llm_api_key == "test-api-key-12345"
        assert env._llm_model == "gpt-4o"

    def test_reset_llm_credentials_from_env(self, mock_data_dir, monkeypatch):
        """LLM credentials should fall back to environment variables."""
        from envs.agent_world_model_env.server.awm_environment import AWMEnvironment

        monkeypatch.setenv("OPENENV_AWM_LLM_BASE_URL", "https://env.openai.azure.com/")
        monkeypatch.setenv("OPENENV_AWM_LLM_API_KEY", "env-api-key")
        monkeypatch.setenv("OPENENV_AWM_LLM_MODEL", "gpt-env")

        loader = AWMDataLoader(cache_dir=mock_data_dir)
        env = AWMEnvironment(data_loader=loader)
        env.reset(scenario="test_shop", task_idx=0)

        assert env._llm_base_url == "https://env.openai.azure.com/"
        assert env._llm_api_key == "env-api-key"
        assert env._llm_model == "gpt-env"

    def test_verify_accepts_final_answer(self, mock_data_dir):
        """verify tool should accept final_answer parameter for code mode."""
        from envs.agent_world_model_env.server.awm_environment import AWMEnvironment
        from openenv.core.env_server.mcp_types import CallToolAction

        loader = AWMDataLoader(cache_dir=mock_data_dir)
        env = AWMEnvironment(data_loader=loader)
        env._scenario = "test_shop"
        env._task = "Buy a widget"
        env._task_idx = 0
        env._reset_ok = True
        env._has_verifier = {"sql": True, "code": True}
        env._db_path = "/tmp/test.db"
        env._initial_db_path = "/tmp/test_initial.db"

        # Mock the data loader to return a verifier that uses final_answer
        def mock_get_verifier(s, i, m):
            if m == "code":
                return {
                    "verification": {
                        "code": (
                            "def verify_task_completion(initial_db_path, final_db_path, final_answer):\n"
                            "    if final_answer == 'test_answer':\n"
                            "        return {'result': 'complete', 'answer_received': final_answer}\n"
                            "    return {'result': 'others'}\n"
                        )
                    }
                }
            return None

        loader.get_verifier = mock_get_verifier

        obs = env.step(
            CallToolAction(
                tool_name="verify",
                arguments={"verifier_mode": "code", "final_answer": "test_answer"},
            )
        )
        # Should not error - the final_answer should be passed to verifier
        assert obs.reward_type in ("complete", "others", "judge_error")

    def test_verify_returns_reward_value(self, mock_data_dir):
        """verify tool should return a numeric reward value, not None."""
        from envs.agent_world_model_env.server.awm_environment import AWMEnvironment
        from openenv.core.env_server.mcp_types import CallToolAction

        loader = AWMDataLoader(cache_dir=mock_data_dir)
        env = AWMEnvironment(data_loader=loader)
        env._scenario = "test_shop"
        env._task = "Buy a widget"
        env._task_idx = 0
        env._reset_ok = True
        env._has_verifier = {"sql": True, "code": True}
        env._db_path = "/tmp/test.db"
        env._initial_db_path = "/tmp/test_initial.db"

        def mock_get_verifier(s, i, m):
            return {
                "verification": {
                    "code": (
                        "def verify_task_completion(initial_db_path, final_db_path, final_answer):\n"
                        "    return {'result': 'complete'}\n"
                    )
                }
            }

        loader.get_verifier = mock_get_verifier

        obs = env.step(
            CallToolAction(tool_name="verify", arguments={"verifier_mode": "code"})
        )
        # Reward should be a float, not None
        assert obs.reward is not None
        assert isinstance(obs.reward, (int, float))
        # complete should give 1.0 reward
        if obs.reward_type == "complete":
            assert obs.reward == 1.0

    def test_done_returns_reward_zero(self, mock_data_dir):
        """done tool should return reward=0.0, not None."""
        from envs.agent_world_model_env.server.awm_environment import AWMEnvironment
        from openenv.core.env_server.mcp_types import CallToolAction

        loader = AWMDataLoader(cache_dir=mock_data_dir)
        env = AWMEnvironment(data_loader=loader)
        env._scenario = "test_shop"
        env._task = "Buy a widget"
        env._task_idx = 0
        env._reset_ok = True

        obs = env.step(CallToolAction(tool_name="done", arguments={}))
        assert obs.done is True
        assert obs.reward == 0.0
        assert obs.reward_type == "episode_done"

    def test_verify_multiple_times(self, mock_data_dir):
        """verify can be called multiple times with different modes."""
        from envs.agent_world_model_env.server.awm_environment import AWMEnvironment
        from openenv.core.env_server.mcp_types import CallToolAction

        loader = AWMDataLoader(cache_dir=mock_data_dir)
        env = AWMEnvironment(data_loader=loader)
        env._scenario = "test_shop"
        env._task = "Buy a widget"
        env._task_idx = 0
        env._reset_ok = True
        env._has_verifier = {"sql": True, "code": True}
        env._db_path = "/tmp/test.db"
        env._initial_db_path = "/tmp/test_initial.db"

        def mock_get_verifier(s, i, m):
            return {
                "verification": {
                    "code": (
                        "def verify_task_completion(initial_db_path, final_db_path, final_answer):\n"
                        "    return {'result': 'complete'}\n"
                    )
                }
            }

        loader.get_verifier = mock_get_verifier

        # First verify with code mode
        obs1 = env.step(
            CallToolAction(tool_name="verify", arguments={"verifier_mode": "code"})
        )
        assert obs1.done is False  # verify doesn't end episode

        # Second verify (can do sql mode too, but here we just test code again)
        obs2 = env.step(
            CallToolAction(tool_name="verify", arguments={"verifier_mode": "code"})
        )
        assert obs2.done is False  # still not done

        # Episode should still be active until done is called
        assert env._episode_done is False
