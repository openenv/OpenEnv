"""Training orchestrator demo; an agent should only receive the MCP tools."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from openenv.core.mcp_client import MCPToolClient

from examples.policies import reservations


def main(url):
    with MCPToolClient(base_url=url).sync() as env:
        env.reset(seed=17)  # Orchestration only.
        view = env.call_tool("observe_demand_history")
        while not view["done"]:
            view = env.call_tool("reserve_capacity", reservations=reservations(view))
            print(f"day={view['episode_day']} cost={view['previous']['modeled_cost']:.6f}")
        print(f"total_modeled_cost={view['cumulative_modeled_cost']:.6f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000")
