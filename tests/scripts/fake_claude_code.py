# SPDX-License-Identifier: BSD-3-Clause

"""A stand-in for `claude -p --input-format/--output-format stream-json`.

Spawned by `test_claude_code_harness_eval_example.py` in place of Claude Code,
so the adapter is tested offline. It prints the same JSON-line events Claude
Code does, and calls the environment's tools for real over the MCP server
passed in `--mcp-config`:

- every message makes it look the customer up with `get_user_details`;
- a message containing "crash" makes the process exit mid-turn;
- a message containing "stall" makes it go quiet without exiting;
- a message containing "api error" makes it end the turn with an error result.

`$FAKE_CLAUDE_ARGV` names a file where the command line is written, so the
test can check the flags the adapter passed.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys

from fastmcp import Client


def emit(payload: dict) -> None:
    print(json.dumps(payload), flush=True)


async def run_turn(message: str, url: str) -> None:
    emit({"type": "system", "subtype": "init", "tools": []})
    if "crash" in message:
        sys.exit(3)
    if "stall" in message:
        await asyncio.sleep(60)
    if "api error" in message:
        error = "API Error: Connection dropped (ECONNRESET)"
        emit(
            {"type": "result", "subtype": "success", "is_error": True, "result": error}
        )
        return
    name, arguments = "get_user_details", {"user_id": "noah_muller_9847"}
    emit(
        {
            "type": "assistant",
            "message": {
                "content": [
                    {"type": "thinking", "thinking": "..."},
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": f"mcp__env__{name}",
                        "input": arguments,
                    },
                ]
            },
        }
    )
    async with Client(url) as client:
        result = await client.call_tool(name, arguments)
    content = result.content[0].text
    emit(
        {
            "type": "user",
            "message": {
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_1",
                        "content": content,
                    }
                ]
            },
        }
    )
    reply = f"{name} said {content}"
    emit(
        {"type": "assistant", "message": {"content": [{"type": "text", "text": reply}]}}
    )
    emit({"type": "result", "subtype": "success", "is_error": False, "result": reply})


def main() -> None:
    argv = sys.argv[1:]
    with open(os.environ["FAKE_CLAUDE_ARGV"], "w") as f:
        json.dump(argv, f)
    config = json.loads(argv[argv.index("--mcp-config") + 1])
    url = config["mcpServers"]["env"]["url"]
    for line in sys.stdin:
        request = json.loads(line)
        asyncio.run(run_turn(request["message"]["content"], url))


if __name__ == "__main__":
    main()
