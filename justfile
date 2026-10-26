# Common development commands for OpenEnv. Run `just` to list recipes.

export PYTHONPATH := "src:envs"

# List available recipes
default:
    @just --list

# Install dependencies
install:
    uv sync --all-extras

# Run the test suite (excludes envs that need special setup, like CI)
test:
    uv run pytest tests/ -v --tb=short \
        --ignore=tests/envs/test_browsergym_environment.py \
        --ignore=tests/envs/test_dipg_environment.py \
        --ignore=tests/envs/test_websearch_environment.py \
        -m "not integration and not network and not docker"

# Run a single test file or pattern, e.g. `just test-one tests/core/test_state_schema_subclass.py`
test-one target:
    uv run pytest {{target}} -v --tb=short
