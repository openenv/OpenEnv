# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the FinRL Environment.

The FinRL environment wraps FinRL's StockTradingEnv for reinforcement learning
based stock trading.
"""

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class FinRLAction(Action):
    """
    Action for the FinRL environment.

    Represents trading actions for multiple stocks. Each value in the actions
    array represents the number of shares to buy (positive) or sell (negative)
    for each stock.

    Attributes:
        actions: Array of action values, one per stock. Values are normalized
                 between -1 and 1, where:
                 - Positive values indicate buying
                 - Negative values indicate selling
                 - Magnitude indicates relative size of trade
    """

    actions: list[float]


class FinRLObservation(Observation):
    """
    Observation from the FinRL environment.

    Represents the current state of the trading environment including:
    - Account balance
    - Stock prices
    - Stock holdings
    - Technical indicators (MACD, RSI, etc.)

    Attributes:
        state: Flattened state vector containing all environment information.
               Structure: [balance, prices..., holdings..., indicators...]
        terminal: Whether the episode has ended
        portfolio_value: Total value of portfolio (cash + holdings)
        date: Current trading date
        metadata: Additional information about the state
    """

    state: list[float]
    portfolio_value: float = 0.0
    date: str = ""
