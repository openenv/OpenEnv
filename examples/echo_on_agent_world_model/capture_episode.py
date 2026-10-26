"""Capture a *real* AWM episode and write it to ``fixtures/`` for the offline demo.

This is how ``fixtures/awm_ecommerce_episode.json`` was produced: it runs a short,
correct solution to the real ``e_commerce_33`` task against a live
``agent_world_model_env`` server and records every ``(action, observation)`` with
the **actual** tool names, arguments, and tool results — so the offline demo and
``run_demo.py --live`` are faithful to the real environment, not hand-authored.

The scripted tool calls stand in for what a policy would choose (no trained policy
needed). Run a server first (see README), then::

    PYTHONPATH=../../src:../../envs python capture_episode.py \
        --base-url http://localhost:8899 --out fixtures/awm_ecommerce_episode.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import pathlib

HERE = pathlib.Path(__file__).parent


def _as_obj(tool_result):
    """AWM tool_result is often a JSON string; parse it when possible."""
    if isinstance(tool_result, str):
        try:
            return json.loads(tool_result)
        except json.JSONDecodeError:
            return tool_result
    return tool_result


async def capture(
    base_url: str, scenario: str, task_idx: int, max_price: float
) -> dict:
    from agent_world_model_env import AWMEnv  # lazy
    from openenv.core.env_server.mcp_types import CallToolAction  # lazy

    steps: list[dict] = []

    async def call(env, tool_name: str, **arguments):
        result = await env.step(
            CallToolAction(tool_name=tool_name, arguments=arguments)
        )
        obs = result.observation
        obs_dict = obs.model_dump() if hasattr(obs, "model_dump") else dict(obs)
        steps.append(
            {
                "action": {"tool_name": tool_name, "arguments": arguments},
                "observation": obs_dict,
            }
        )
        return obs_dict, getattr(result, "reward", None)

    async with AWMEnv(base_url=base_url) as env:
        reset_res = await env.reset(scenario=scenario, task_idx=task_idx)
        task = getattr(reset_res.observation, "task", "") or ""
        tools = [t.name for t in await env.list_tools()]

        # 1) search, sorted by rating (the task asks for the top-rated item under budget)
        s_obs, _ = await call(
            env,
            "search_products",
            query="wireless noise cancelling headphones",
            sort_by="average_rating",
            limit=5,
        )
        search = _as_obj(s_obs.get("tool_result")) or {}
        products = search.get("products", []) if isinstance(search, dict) else []

        # 2) pick the highest-rated product whose lowest active offer is under budget
        chosen_pid = None
        for p in products:
            offer = p.get("lowest_active_offer") or {}
            price = offer.get("price")
            if price is not None and price < max_price:
                chosen_pid = p["product"]["id"]
                break
        if chosen_pid is None and products:
            chosen_pid = products[0]["product"]["id"]

        # 3) get its offers and take the cheapest active one's id
        o_obs, _ = await call(env, "list_product_offers", product_id=chosen_pid)
        offers = (_as_obj(o_obs.get("tool_result")) or {}).get("offers", [])
        offers = [o for o in offers if o.get("is_active", True)]
        offers.sort(key=lambda o: o.get("price", 1e9))
        offer_id = offers[0]["id"] if offers else None

        # 4) cart: create, add, confirm
        await call(env, "get_or_create_active_cart")
        await call(env, "add_item_to_cart", product_offer_id=offer_id, quantity=1)
        await call(env, "list_cart_items")

        # 5) verify (real grader) and release the session
        _, reward = await call(env, "verify", verifier_mode="code")
        await env.step(
            CallToolAction(tool_name="done", arguments={"keep_session": False})
        )

    return {
        "scenario": scenario,
        "task": task,
        "task_idx": task_idx,
        "tools": tools,
        "steps": steps,
        "reward": float(reward) if reward is not None else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default="http://localhost:8899")
    ap.add_argument("--scenario", default="e_commerce_33")
    ap.add_argument("--task-idx", type=int, default=0)
    ap.add_argument("--max-price", type=float, default=200.0)
    ap.add_argument(
        "--out", default=str(HERE / "fixtures" / "awm_ecommerce_episode.json")
    )
    args = ap.parse_args()

    episode = asyncio.run(
        capture(args.base_url, args.scenario, args.task_idx, args.max_price)
    )
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(episode, indent=2) + "\n")
    print(f"captured {len(episode['steps'])} steps -> {out}")
    print(f"task   : {episode['task']}")
    print(f"reward : {episode['reward']}")


if __name__ == "__main__":
    main()
