# SPDX-License-Identifier: BSD-3-Clause

"""Minimal OpenEnv-protocol server used by the opt-in ACA integration test.

This is intentionally self-contained and depends only on ``fastapi`` +
``uvicorn`` (no ``openenv`` install), so the ACA integration test can run it on
a stock public ``python-3.11`` disk image without building a custom disk image
first. It implements exactly what ``GenericEnvClient`` (simulation mode) needs:

  * ``GET /health`` -> ``200`` (what ``wait_for_ready`` polls)
  * ``WS /ws``      -> JSON ``reset`` / ``step`` / ``state`` / ``close`` messages

Its purpose is to prove that an ACA anonymous-port ingress proxies a WebSocket
*upgrade* so ``EnvClient`` can complete a real ``reset()``/``step()``/``state()``
round-trip — something a ``200`` on ``/health`` alone does not prove.
"""

import json

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

app = FastAPI()
_episode = {"steps": 0, "open": False}


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.websocket("/ws")
async def ws(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        while True:
            msg = json.loads(await websocket.receive_text())
            mtype = msg.get("type")
            data = msg.get("data", {}) or {}

            if mtype == "reset":
                _episode["steps"] = 0
                _episode["open"] = True
                await websocket.send_text(
                    json.dumps(
                        {
                            "type": "observation",
                            "data": {
                                "observation": {"message": "reset", "echo": data},
                                "reward": 0.0,
                                "done": False,
                            },
                        }
                    )
                )
            elif mtype == "step":
                _episode["steps"] += 1
                await websocket.send_text(
                    json.dumps(
                        {
                            "type": "observation",
                            "data": {
                                "observation": {
                                    "message": "step",
                                    "echo": data,
                                    "steps": _episode["steps"],
                                },
                                "reward": 1.0,
                                "done": False,
                            },
                        }
                    )
                )
            elif mtype == "state":
                await websocket.send_text(
                    json.dumps(
                        {
                            "type": "state",
                            "data": {
                                "episode_steps": _episode["steps"],
                                "episode_open": _episode["open"],
                                "ready": True,
                            },
                        }
                    )
                )
            elif mtype == "close":
                _episode["open"] = False
                await websocket.send_text(json.dumps({"type": "ack", "data": {}}))
                await websocket.close()
                return
            else:
                await websocket.send_text(
                    json.dumps(
                        {
                            "type": "error",
                            "data": {
                                "message": f"unknown message type: {mtype}",
                                "code": "UNKNOWN_TYPE",
                            },
                        }
                    )
                )
    except WebSocketDisconnect:
        _episode["open"] = False


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
