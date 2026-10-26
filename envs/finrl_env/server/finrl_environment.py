# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FinRL Environment Implementation.

Wraps FinRL's StockTradingEnv to conform to the OpenEnv interface.
"""

from typing import Any

from uuid import uuid4

import numpy as np

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

# Support both in-repo and standalone imports
try:
    # In-repo imports (when running from OpenEnv repository)
    from ..models import FinRLAction, FinRLObservation
except ImportError as e:
    if "relative import" not in str(e) and "no known parent package" not in str(e):
        raise
    # Standalone imports (when running via uvicorn server.app:app)
    from models import FinRLAction, FinRLObservation


def _json_safe(value: Any) -> Any:
    """Convert library-specific values into JSON-serializable Python types."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()

    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            return tolist()
        except Exception:
            pass

    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except Exception:
            pass

    return str(value)


class FinRLEnvironment(Environment):
    """
    A FinRL stock trading environment wrapper for OpenEnv.

    This environment wraps FinRL's StockTradingEnv and provides the standard
    OpenEnv interface (reset, step, state). It enables RL training on financial
    trading tasks using the OpenEnv framework.

    Example:
        >>> import pandas as pd
        >>> from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
        >>>
        >>> # Load your stock data
        >>> df = pd.read_csv('stock_data.csv')
        >>>
        >>> # Configure FinRL environment parameters
        >>> config = {
        >>>     'df': df,
        >>>     'stock_dim': 1,
        >>>     'hmax': 100,
        >>>     'initial_amount': 100000,
        >>>     'num_stock_shares': [0],
        >>>     'buy_cost_pct': [0.001],
        >>>     'sell_cost_pct': [0.001],
        >>>     'reward_scaling': 1e-4,
        >>>     'state_space': 50,
        >>>     'action_space': 1,
        >>>     'tech_indicator_list': ['macd', 'rsi_30', 'cci_30', 'dx_30']
        >>> }
        >>>
        >>> # Create environment
        >>> env = FinRLEnvironment(finrl_env_class=StockTradingEnv, finrl_env_config=config)
        >>> obs = env.reset()
        >>> print(obs.state)  # Current state vector
        >>> print(obs.portfolio_value)  # Total portfolio value
    """

    def __init__(self, finrl_env_class, finrl_env_config: dict):
        """
        Initialize the FinRL environment wrapper.

        Args:
            finrl_env_class: The FinRL environment class (e.g., StockTradingEnv)
            finrl_env_config: Configuration dictionary for FinRL environment.
                             Should contain all required parameters like df, stock_dim, etc.
        """
        super().__init__()
        self.finrl_env_class = finrl_env_class
        self.finrl_env_config = finrl_env_config
        self.finrl_env = None
        self._state = State(episode_id=str(uuid4()), step_count=0)

    def reset(self) -> FinRLObservation:
        """
        Reset the environment to start a new episode.

        Returns:
            FinRLObservation with initial state and portfolio value
        """
        # Create a fresh FinRL environment instance
        self.finrl_env = self.finrl_env_class(**self.finrl_env_config)

        # Reset the FinRL environment
        state, _ = self.finrl_env.reset()

        # Update our state tracking
        self._state = State(episode_id=str(uuid4()), step_count=0)

        # Calculate initial portfolio value
        portfolio_value = self._calculate_portfolio_value(state)

        # Get date if available
        date = self._get_current_date()

        return FinRLObservation(
            state=state.tolist() if isinstance(state, np.ndarray) else list(state),
            portfolio_value=portfolio_value,
            date=date,
            done=False,
            reward=0.0,
        )

    def step(self, action: FinRLAction) -> FinRLObservation:  # type: ignore[override]
        """
        Execute a trading action in the environment.

        Args:
            action: FinRLAction containing the trading actions for each stock

        Returns:
            FinRLObservation with new state, reward, and done flag

        Raises:
            RuntimeError: If environment not initialized
            ValueError: If action dimensions don't match stock_dim
        """
        if self.finrl_env is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        # Validate action dimensions
        expected_dim = self.finrl_env_config.get("action_space", 1)
        if len(action.actions) != expected_dim:
            raise ValueError(
                f"Action dimension mismatch: expected {expected_dim}, "
                f"got {len(action.actions)}. "
                f"Actions should match config['action_space'] (= stock_dim)."
            )

        # Convert action list to numpy array
        action_array = np.array(action.actions)

        # Execute step in FinRL environment
        state, reward, terminal, truncated, info = self.finrl_env.step(action_array)

        # Update step count
        self._state.step_count += 1

        # Calculate portfolio value
        portfolio_value = self._calculate_portfolio_value(state)

        # Get date if available
        date = self._get_current_date()

        # Combine terminal and truncated into done
        done = terminal or truncated

        return FinRLObservation(
            state=state.tolist() if isinstance(state, np.ndarray) else list(state),
            portfolio_value=portfolio_value,
            date=date,
            done=done,
            reward=float(reward),
            metadata=_json_safe(info),
        )

    @property
    def state(self) -> State:
        """
        Get the current environment state metadata.

        Returns:
            Current State with episode_id and step_count
        """
        return self._state

    def _calculate_portfolio_value(self, state) -> float:
        """
        Calculate total portfolio value from state.

        The state structure in FinRL is typically:
        [balance, prices..., holdings..., indicators...]

        Args:
            state: The environment state

        Returns:
            Total portfolio value (cash + stock holdings value)
        """
        if self.finrl_env is None:
            return 0.0

        # First element is usually cash balance
        state_array = state if isinstance(state, np.ndarray) else np.array(state)

        # Get stock dimension
        stock_dim = self.finrl_env_config.get("stock_dim", 1)

        # State structure: [balance, prices..., holdings..., indicators...]
        balance = state_array[0]
        prices = state_array[1 : 1 + stock_dim]
        holdings = state_array[1 + stock_dim : 1 + 2 * stock_dim]

        # Calculate total value
        portfolio_value = balance + np.sum(prices * holdings)

        return float(portfolio_value)

    def _get_current_date(self) -> str:
        """
        Get the current trading date from FinRL environment.

        Returns:
            Current date as string, or empty string if not available
        """
        if self.finrl_env is None:
            return ""

        try:
            return str(self.finrl_env._get_date())
        except (AttributeError, Exception):
            # If date is not available, return empty string
            return ""
