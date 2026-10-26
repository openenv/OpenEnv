# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the FinRL Environment.

This module creates an HTTP server that exposes the FinRLEnvironment
over HTTP and WebSocket endpoints, compatible with EnvClient.

The server expects environment configuration to be provided either:
1. Through environment variables (FINRL_CONFIG_PATH)
2. Through a mounted configuration file
3. Through default sample configuration

Usage:
    # With configuration file:
    export FINRL_CONFIG_PATH=/path/to/config.json
    uvicorn envs.finrl_env.server.app:app --host 0.0.0.0 --port 8000

    # Development (with auto-reload):
    uvicorn envs.finrl_env.server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn envs.finrl_env.server.app:app --host 0.0.0.0 --port 8000 --workers 4
"""

import json
import os
from pathlib import Path

import pandas as pd

from openenv.core.env_server import create_app

# Support both in-repo and standalone imports
try:
    # In-repo imports (when running from OpenEnv repository)
    from ..models import FinRLAction, FinRLObservation
    from .finrl_environment import FinRLEnvironment
except ImportError as e:
    if "relative import" not in str(e) and "no known parent package" not in str(e):
        raise
    # Standalone imports (when running via uvicorn server.app:app)
    from models import FinRLAction, FinRLObservation
    from server.finrl_environment import FinRLEnvironment


def load_finrl_config():
    """
    Load FinRL environment configuration.

    Configuration can be provided through:
    1. FINRL_CONFIG_PATH environment variable pointing to a JSON file
    2. Default sample configuration for testing

    Returns:
        tuple: (finrl_env_class, config_dict)
    """
    config_path = os.environ.get("FINRL_CONFIG_PATH")

    if config_path and Path(config_path).exists():
        print(f"Loading FinRL config from: {config_path}")
        with open(config_path) as f:
            config = json.load(f)

        # Load data file if specified
        if "data_path" in config:
            data_path = config["data_path"]
            print(f"Loading stock data from: {data_path}")
            df = pd.read_csv(data_path)
            config["df"] = df
            del config["data_path"]  # Remove path from config

        # Import FinRL environment class
        from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv

        return StockTradingEnv, config

    else:
        # Create a minimal default configuration for testing
        print("No config file found. Using default sample configuration.")
        print("Set FINRL_CONFIG_PATH environment variable to use custom config.")

        # Create sample data for testing (sine wave as "stock price")
        import numpy as np

        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        sample_df = pd.DataFrame(
            {
                "date": dates,
                "tic": "SAMPLE",
                "close": 100 + 10 * np.sin(np.linspace(0, 4 * np.pi, 100)),
                "high": 100 + 10 * np.sin(np.linspace(0, 4 * np.pi, 100)) + 2,
                "low": 100 + 10 * np.sin(np.linspace(0, 4 * np.pi, 100)) - 2,
                "open": 100 + 10 * np.sin(np.linspace(0, 4 * np.pi, 100)),
                "volume": 1000000,
                "macd": np.random.randn(100),
                "rsi_30": 50 + 20 * np.random.randn(100),
                "cci_30": np.random.randn(100) * 50,
                "dx_30": np.random.randn(100) * 20,
            }
        )

        config = {
            "df": sample_df,
            "stock_dim": 1,
            "hmax": 100,
            "initial_amount": 100000,
            "num_stock_shares": [0],
            "buy_cost_pct": [0.001],
            "sell_cost_pct": [0.001],
            "reward_scaling": 1e-4,
            "state_space": 1 + 1 + 1 + 4,  # balance + price + holding + 4 indicators
            "action_space": 1,
            "tech_indicator_list": ["macd", "rsi_30", "cci_30", "dx_30"],
        }

        from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv

        return StockTradingEnv, config


# Load configuration
finrl_env_class, finrl_config = load_finrl_config()


# Factory function to create FinRLEnvironment instances
def create_finrl_environment():
    """Factory function that creates FinRLEnvironment with config."""
    return FinRLEnvironment(
        finrl_env_class=finrl_env_class, finrl_env_config=finrl_config
    )


# Create the FastAPI app with routes
# Pass the factory function instead of an instance for WebSocket session support
app = create_app(
    create_finrl_environment, FinRLAction, FinRLObservation, env_name="finrl_env"
)


@app.get("/config")
def get_config():
    """
    Get the current environment configuration (excluding DataFrame).

    Returns:
        dict: Environment configuration
    """
    config_copy = finrl_config.copy()
    # Remove DataFrame from response (too large)
    config_copy.pop("df", None)
    return {
        "stock_dim": config_copy.get("stock_dim"),
        "initial_amount": config_copy.get("initial_amount"),
        "action_space": config_copy.get("action_space"),
        "state_space": config_copy.get("state_space"),
        "tech_indicators": config_copy.get("tech_indicator_list"),
    }


def main():
    import uvicorn

    print("=" * 60)
    print("FinRL Environment Server")
    print("=" * 60)
    print(f"Stock dimension: {finrl_config.get('stock_dim')}")
    print(f"Initial amount: ${finrl_config.get('initial_amount'):,.0f}")
    print(f"Action space: {finrl_config.get('action_space')}")
    print(f"State space: {finrl_config.get('state_space')}")
    print("=" * 60)
    print("Server starting on http://0.0.0.0:8000")
    print("=" * 60)

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
