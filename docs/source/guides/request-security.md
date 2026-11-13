# Request Security

An environment server is a plain HTTP and WebSocket service. On a public Space, or in a container behind public ingress, anyone who finds the URL can call `reset`, `step` and the MCP tools. OpenEnv can put [fastapi-guard](https://github.com/Guard-Core/fastapi-guard) in front of the server, adding per-IP rate limits, IP and user-agent blocking, and temporary bans for clients that probe for vulnerabilities. It is off by default: with `OPENENV_GUARD_ENABLED` unset, the server runs exactly as before.

## What It Covers

The guard is for an environment server on a public URL that unauthenticated clients can reach. It handles:

- **Floods from one IP.** Each client IP gets `OPENENV_GUARD_RATE_LIMIT` requests per window. Over the limit, HTTP requests get 429 until the window passes. WebSocket handshakes are counted in a separate bucket.
- **Scanners.** Penetration detection checks the path, query string and headers of each request against known attack patterns and refuses matches with 400. An IP that trips it `OPENENV_GUARD_AUTO_BAN_THRESHOLD` times is banned for `OPENENV_GUARD_AUTO_BAN_DURATION` seconds and gets 403 on every path.
- **Sources you want out.** IPs and CIDR ranges on a block or allow list, and user agents that match a pattern.

WebSocket connections (`/ws`, `/mcp`, `/harness`, `/ws/ui`) get the same checks on the handshake. A refused handshake is closed before it is accepted, so the client gets HTTP 403 on the upgrade and no socket. The web UI at `/web` is covered too.

`/health`, `/schema`, `/metadata`, `/docs`, `/redoc` and `/openapi.json` skip penetration detection and user-agent blocking, so orchestrators can keep probing them. IP lists, bans and rate limits still apply there.

It does not cover:

- **Authentication.** Anyone who gets past the guard can still reset and step. Use a private Space or a token to control who can connect.
- **WebSocket messages.** Only the handshake is checked. There is no per-message rate limit (the concurrent session limit still applies), and a ban issued during a session takes effect at the next handshake.
- **Request bodies.** They carry actions, which are model output such as text, code or tool arguments. Attack patterns would flag ordinary content, so bodies are not scanned.
- **Floods spread across many IPs**, beyond what per-IP limits catch.

## Enabling the Guard

1. Add the `security` extra to the environment's OpenEnv dependency in `pyproject.toml` (`"openenv[security]"` instead of `"openenv"`) and regenerate `uv.lock` with `uv lock`. An image that installs with pip needs `pip install "openenv[security]"`.
2. Set `OPENENV_GUARD_ENABLED=1` where the server runs. For a Space, declare it in `openenv.yaml` so `openenv push` sets it as a Space variable, or pass `-e OPENENV_GUARD_ENABLED=1` to `openenv push`:

   ```yaml
   variables:
     OPENENV_GUARD_ENABLED: "1"
   ```

   Elsewhere, set it in the container's environment, for example with `docker run -e OPENENV_GUARD_ENABLED=1`.

Servers created with `create_app` or `create_fastapi_app` attach the guard when the app is created. If it is enabled and fastapi-guard is not installed, the server fails at startup with an `ImportError` that names the install command. A value other than `1`, `true`, `yes`, `on`, `0`, `false`, `no` or `off` fails with a `ValueError`.

## Settings

All settings are environment variables, read when the app is created.

| Variable | Default | Effect |
|---|---|---|
| `OPENENV_GUARD_ENABLED` | off | `1`, `true`, `yes` or `on` turns the guard on |
| `OPENENV_GUARD_RATE_LIMIT` | `100` | Requests per client IP per window. HTTP requests and WebSocket handshakes are counted separately |
| `OPENENV_GUARD_RATE_LIMIT_WINDOW` | `60` | Window length, in seconds |
| `OPENENV_GUARD_AUTO_BAN_THRESHOLD` | `10` | Requests refused by penetration detection before the IP is banned. Rate-limited requests don't count |
| `OPENENV_GUARD_AUTO_BAN_DURATION` | `300` | Ban length, in seconds |
| `OPENENV_GUARD_BLOCKED_IPS` | none | Comma-separated IPs and CIDR ranges to refuse |
| `OPENENV_GUARD_ALLOWED_IPS` | none | Comma-separated IPs and CIDR ranges. When set, only these get in and `OPENENV_GUARD_BLOCKED_IPS` is ignored |
| `OPENENV_GUARD_BLOCKED_USER_AGENTS` | none | Comma-separated regular expressions, matched case-insensitively anywhere in the `User-Agent` header |
| `OPENENV_GUARD_TRUSTED_PROXIES` | none | Comma-separated IPs and CIDR ranges of the reverse proxies in front of the server, see [Trusted Proxies](#trusted-proxies) |
| `OPENENV_GUARD_TRUSTED_PROXY_DEPTH` | `1` | Number of proxies that append to `X-Forwarded-For` |
| `OPENENV_GUARD_REDIS_URL` | none | Redis URL for sharing rate limits and bans across workers and replicas. Without it, each process keeps its own state in memory |

With `OPENENV_GUARD_REDIS_URL` set, the guard connects to Redis at startup. While Redis is unreachable, HTTP requests are refused and WebSocket handshakes are closed with code 1013 (try again later). A Redis URL that contains a password belongs in a Space secret (`openenv push --secret`), not in `openenv.yaml`.

## Trusted Proxies

By default no proxy is trusted: the guard ignores `X-Forwarded-For` and treats the TCP peer as the client, so a client cannot choose its own address to get past a ban, a block list or a rate limit.

Behind a reverse proxy, every request comes from a proxy's address. A Space runs behind several proxies: the TCP peer (`request.client.host`) is one of a few private `10.x` addresses, and `X-Forwarded-For` holds the client's address. With no trusted proxies listed, IP lists, bans and rate limits apply to the proxy addresses. Blocking a client's IP has no effect, and all clients share one rate-limit bucket per proxy address: once those buckets fill, every client gets 429 until the window passes, `/health` included. A single client that trips penetration detection enough times gets the proxies banned, and with them everyone.

List the proxies in `OPENENV_GUARD_TRUSTED_PROXIES` to give each client its own bucket. The guard then takes the client address from the `X-Forwarded-For` entry the proxy appended. With more than one proxy in the chain, set `OPENENV_GUARD_TRUSTED_PROXY_DEPTH` to the number of proxies that append to the header. A Space appends one entry, so keep the default of `1`.

Find the proxy addresses instead of guessing a range:

1. Send a few requests through the public URL and read the server log (on a Space, `hf spaces logs <user>/<space>`). uvicorn's access log shows the address each request came from, as in `INFO: 10.16.0.5:43218 - "GET /health HTTP/1.1" 200 OK`. Requests through the proxy show the proxy's address.
2. Keep watching for a while, since a platform can run several proxy instances, and list every address you see. When this was checked, a Space used four, all seen within the first ten requests.
3. Don't list a whole private range such as `10.0.0.0/8` to cover addresses you haven't seen. Any peer in a trusted range can put any address in `X-Forwarded-For`, and so pick its own bucket or get past a block list.

A request through a proxy you haven't listed is treated as coming from that proxy, so check the log again if a block stops working or clients start sharing a limit.

Run uvicorn without `--forwarded-allow-ips '*'` when the guard is on. With that flag, uvicorn replaces the peer address with the left-most `X-Forwarded-For` entry before the guard sees the request, and the client chooses that entry: a Space appends to the header the client sends instead of replacing it.

> [!NOTE]
> Checked on a Hugging Face Space on 2026-11-12.

## Related Reading

- [Deploying an Environment](../getting_started/environment-builder), for Space variables and secrets
- [Simulation vs Production](simulation-vs-production), for keeping agents away from `/ws`
- [fastapi-guard documentation](https://github.com/Guard-Core/fastapi-guard)
