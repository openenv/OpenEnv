---
title: FinRL Environment Server
emoji: 📈
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# FinRL Environment

A wrapper around [FinRL](https://github.com/AI4Finance-Foundation/FinRL) stock trading environments that conforms to the OpenEnv specification.

## Overview

This environment enables reinforcement learning for stock trading tasks using FinRL's powerful StockTradingEnv, exposed through OpenEnv's simple HTTP API. It supports:

- **Stock Trading**: Buy/sell actions across multiple stocks
- **Portfolio Management**: Track balance, holdings, and portfolio value
- **Technical Indicators**: MACD, RSI, CCI, DX, and more
- **Flexible Configuration**: Custom data sources and trading parameters

## Quick Start

### 1. Build the Docker Image

First, build the base image (from OpenEnv root):

```bash
cd OpenEnv
docker build -t envtorch-base:latest -f src/openenv/core/containers/images/Dockerfile .
```

Then build the FinRL environment image:

```bash
docker build -t finrl-env:latest -f envs/finrl_env/server/Dockerfile .
```

### 2. Run the Server

#### Option A: With Default Sample Data

```bash
docker run -p 8000:8000 finrl-env:latest
```

This starts the server with synthetic sample data for testing.

#### Option B: With Custom Configuration

Create a configuration file `config.json`:

```json
{
  "data_path": "/data/stock_data.csv",
  "stock_dim": 3,
  "hmax": 100,
  "initial_amount": 100000,
  "num_stock_shares": [0, 0, 0],
  "buy_cost_pct": [0.001, 0.001, 0.001],
  "sell_cost_pct": [0.001, 0.001, 0.001],
  "reward_scaling": 0.0001,
  "state_space": 25,
  "action_space": 3,
  "tech_indicator_list": ["macd", "rsi_30", "cci_30", "dx_30"]
}
```

Run with configuration:

```bash
docker run -p 8000:8000 \
  -v $(pwd)/config.json:/config/config.json \
  -v $(pwd)/data:/data \
  -e FINRL_CONFIG_PATH=/config/config.json \
  finrl-env:latest
```

### 3. Use the Client

```python
from envs.finrl_env import FinRLEnv, FinRLAction
import numpy as np

# Connect to server
client = FinRLEnv(base_url="http://localhost:8000")

# Get configuration
config = client.get_config()
print(f"Trading {config['stock_dim']} stocks")
print(f"Initial capital: ${config['initial_amount']:,.0f}")

# Reset environment
result = client.reset()
print(f"Initial portfolio value: ${result.observation.portfolio_value:,.2f}")

# Trading loop
for step in range(100):
    # Get current state
    state = result.observation.state

    # Your RL policy here (example: random actions)
    num_stocks = config['stock_dim']
    actions = np.random.uniform(-1, 1, size=num_stocks).tolist()

    # Execute action
    result = client.step(FinRLAction(actions=actions))

    print(f"Step {step}: Portfolio=${result.observation.portfolio_value:,.2f}, "
          f"Reward={result.reward:.2f}")

    if result.done:
        print("Episode finished!")
        break

client.close()
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    RL Training Framework                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Policy Net   │  │ Value Net    │  │ Replay       │      │
│  │ (PyTorch)    │  │ (PyTorch)    │  │ Buffer       │      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
│         └──────────────────┴──────────────────┘              │
│                            │                                 │
│                   ┌────────▼────────┐                        │
│                   │ FinRLEnv        │ ← HTTP Client          │
│                   │ (EnvClient)     │                        │
│                   └────────┬────────┘                        │
└────────────────────────────┼─────────────────────────────────┘
                             │ HTTP (JSON)
                    ┌────────▼────────┐
                    │ Docker Container│
                    │  Port: 8000     │
                    │                 │
                    │ ┌─────────────┐ │
                    │ │FastAPI      │ │
                    │ │Server       │ │
                    │ └──────┬──────┘ │
                    │        │        │
                    │ ┌──────▼──────┐ │
                    │ │ FinRL       │ │
                    │ │ Environment │ │
                    │ └──────┬──────┘ │
                    │        │        │
                    │ ┌──────▼──────┐ │
                    │ │ FinRL       │ │
                    │ │ StockTrading│ │
                    │ │ Env         │ │
                    │ └─────────────┘ │
                    └─────────────────┘
```

## API Reference

### FinRLAction

Trading action for the environment.

**Attributes:**
- `actions: list[float]` - Array of normalized action values (-1 to 1) for each stock
  - Positive values: Buy
  - Negative values: Sell
  - Magnitude: Relative trade size

**Example:**
```python
# Buy stock 0, sell stock 1, hold stock 2
action = FinRLAction(actions=[0.5, -0.3, 0.0])
```

### FinRLObservation

Observation returned by the environment.

**Attributes:**
- `state: list[float]` - Flattened state vector
  - Structure: `[balance, prices..., holdings..., indicators...]`
- `portfolio_value: float` - Total portfolio value (cash + holdings)
- `date: str` - Current trading date
- `done: bool` - Whether episode has ended
- `reward: float` - Reward for the last action
- `metadata: dict` - Additional information

**Example:**
```python
obs = result.observation
print(f"Portfolio: ${obs.portfolio_value:,.2f}")
print(f"Date: {obs.date}")
print(f"State dimension: {len(obs.state)}")
```

### Client Methods

#### `reset() -> StepResult[FinRLObservation]`

Reset the environment to start a new episode.

```python
result = client.reset()
```

#### `step(action: FinRLAction) -> StepResult[FinRLObservation]`

Execute a trading action.

```python
action = FinRLAction(actions=[0.5, -0.3])
result = client.step(action)
```

#### `state() -> State`

Get episode metadata (episode_id, step_count).

```python
state = client.state()
print(f"Episode: {state.episode_id}, Step: {state.step_count}")
```

#### `get_config() -> dict`

Get environment configuration.

```python
config = client.get_config()
print(config['stock_dim'])
print(config['initial_amount'])
```

## Data Format

The environment expects stock data in the following CSV format:

| date       | tic    | close  | high   | low    | open   | volume  | macd  | rsi_30 | cci_30 | dx_30 |
|------------|--------|--------|--------|--------|--------|---------|-------|--------|--------|-------|
| 2020-01-01 | AAPL   | 100.0  | 102.0  | 98.0   | 99.0   | 1000000 | 0.5   | 55.0   | 10.0   | 15.0  |
| 2020-01-01 | GOOGL  | 1500.0 | 1520.0 | 1480.0 | 1490.0 | 500000  | -0.3  | 48.0   | -5.0   | 20.0  |

**Required columns:**
- `date`: Trading date
- `tic`: Stock ticker symbol
- `close`, `high`, `low`, `open`: Price data
- `volume`: Trading volume
- Technical indicators (as specified in `tech_indicator_list`)

## Configuration Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `data_path` | str | Path to CSV file with stock data |
| `stock_dim` | int | Number of stocks to trade |
| `hmax` | int | Maximum shares per trade |
| `initial_amount` | int | Starting cash balance |
| `num_stock_shares` | list[int] | Initial holdings for each stock |
| `buy_cost_pct` | list[float] | Transaction cost for buying (per stock) |
| `sell_cost_pct` | list[float] | Transaction cost for selling (per stock) |
| `reward_scaling` | float | Scaling factor for rewards |
| `state_space` | int | Dimension of state vector |
| `action_space` | int | Dimension of action space |
| `tech_indicator_list` | list[str] | Technical indicators to include |

## Integration with RL Frameworks

### Stable Baselines 3

```python
from stable_baselines3 import PPO
from envs.finrl_env import FinRLEnv, FinRLAction
import numpy as np

# Create custom wrapper for SB3
class SB3FinRLWrapper:
    def __init__(self, base_url):
        self.env = FinRLEnv(base_url=base_url)
        config = self.env.get_config()
        self.action_space = spaces.Box(
            low=-1, high=1,
            shape=(config['action_space'],),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(config['state_space'],),
            dtype=np.float32
        )

    def reset(self):
        result = self.env.reset()
        return np.array(result.observation.state, dtype=np.float32)

    def step(self, action):
        result = self.env.step(FinRLAction(actions=action.tolist()))
        return (
            np.array(result.observation.state, dtype=np.float32),
            result.reward or 0.0,
            result.done,
            result.observation.metadata
        )

# Train
env = SB3FinRLWrapper("http://localhost:8000")
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)
```

## Troubleshooting

### Server won't start

1. Check if base image exists:
   ```bash
   docker images | grep envtorch-base
   ```

2. Build base image if missing:
   ```bash
   docker build -t envtorch-base:latest -f src/openenv/core/containers/images/Dockerfile .
   ```

### Import errors

Make sure you're in the `src` directory:
```bash
cd OpenEnv/src
python -c "from envs.finrl_env import FinRLEnv"
```

### Configuration errors

Verify your data file has all required columns:
```python
import pandas as pd
df = pd.read_csv('your_data.csv')
print(df.columns.tolist())
```

## Examples

See the `examples/` directory for complete examples:
- `examples/finrl_simple.py` - Basic usage
- `examples/finrl_training.py` - Full training loop with PPO
- `examples/finrl_backtesting.py` - Backtesting a trained agent

## License

BSD 3-Clause License (see LICENSE file in repository root)

## References

- [FinRL Paper](https://arxiv.org/abs/2011.09607)
- [FinRL GitHub](https://github.com/AI4Finance-Foundation/FinRL)
- [OpenEnv Documentation](README.md)
