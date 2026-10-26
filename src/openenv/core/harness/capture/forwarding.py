"""Publish the intercept server at a URL the sandbox can reach.

The agent runs inside a sandbox on the public internet; the intercept runs next to the engine on a
cluster node with no inbound connectivity. Exactly one hop must be forwarded, and it is this one.
The engine itself never needs exposing: it stays on localhost behind the intercept.

`PortForwarder` is that hop, as a swappable strategy. Three implementations, chosen by what the
sandbox actually is rather than by preference:

    DirectExposure    the sandbox can already route to us (local docker, same VPC). No third party,
                      no expiry, no throughput ceiling. Prefer this whenever it is true.
    GradioForwarder      frpc via gradio.networking.setup_tunnel.
    CloudflareForwarder  cloudflared quick forwards, or a named forwards in production.

MEASURED, not assumed. Over a full day of harness bring-up on one intercept:

    gradio / frpc     521 POSTs, ZERO forwarding errors in the server log, ~370ms health round trip,
                      still up after 24h.
    cloudflared       10765 log lines on the sibling experiment, with repeated
                      `failed to accept QUIC stream: timeout`, `datagram manager encountered a
                      failure while serving`, and `lookup region1.v2.argotunnel.com: i/o timeout`.
                      It always reconnected, so this is churn rather than outage, but it is churn.

So gradio is the better default at eval scale. Cloudflare earns its place elsewhere: quick forwards
expire in a way named forwards do not, and a named forwards gives a stable hostname, real access
policies, and no shared relay. gradio.live URLs expire at 72h and are a single frpc hop, which is
fine for a sweep and a bottleneck at GRPO group width.

SHARE TOKENS ARE NOT AUTH. `share_token` identifies the forward to the share server; the resulting
URL is public either way. What protects the GPU behind it is the intercept's own key check, which is
why `SessionRegistry.require_registered` defaults to True.
"""

from __future__ import annotations

import logging
import re
import secrets
import select
import shutil
import subprocess
import threading
import time
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class ForwardingError(RuntimeError):
    """Raised when a forwarder cannot be established. Never returns a half-open forward.

    Failing here is strictly better than returning a URL that does not resolve: a stale URL that
    still looks valid produces a rollout that silently captures nothing, which is the exact class of
    failure this whole layer exists to make impossible.
    """


class PortForwarder(ABC):
    """Publish `local_host:local_port` and hand back a URL reachable from the sandbox."""

    def __init__(self) -> None:
        self._url: str | None = None
        self._local_port: int | None = None

    @classmethod
    def preflight(cls) -> None:
        """Raise ForwardingError if this strategy cannot possibly work here.

        Called BEFORE a run starts, so a missing binary or an uninstalled dependency is a startup
        error rather than a failure discovered after the first sandbox has been billed.
        """

    @abstractmethod
    def start(self, local_port: int, *, local_host: str = "127.0.0.1") -> str:
        """Begin forwarding and return the public URL."""

    @abstractmethod
    def stop(self) -> None:
        """Tear down. Must be idempotent: teardown runs on both success and failure paths."""

    @property
    def url(self) -> str | None:
        return self._url

    @property
    def name(self) -> str:
        return type(self).__name__

    def __enter__(self) -> "PortForwarder":
        return self

    def __exit__(self, *exc) -> None:
        self.stop()


class DirectExposure(PortForwarder):
    """No forward: hand back the address as-is.

    For local docker sandboxes, or any deployment where the sandbox can already route to the host.
    This is the production answer whenever it is available, and it is worth checking before reaching
    for a forward: a forward exists because E2B is off-cluster, not because the design needs one.
    """

    def __init__(self, advertise_host: str = "127.0.0.1", scheme: str = "http") -> None:
        super().__init__()
        self._host = advertise_host
        self._scheme = scheme

    def start(self, local_port: int, *, local_host: str = "127.0.0.1") -> str:
        self._local_port = local_port
        self._url = f"{self._scheme}://{self._host}:{local_port}"
        return self._url

    def stop(self) -> None:
        self._url = None


class GradioForwarder(PortForwarder):
    """frpc, via `gradio.networking.setup_tunnel`.

    Preferred over shelling out to a binary for a reason that matters operationally: it RETURNS the
    URL, rather than leaving us to grep a subprocess log for it and hope it appeared. It is also
    outbound-only, so it needs no inbound firewall rule, and gradio is already an OpenEnv dependency.
    Verifiers reached the same conclusion independently and forwards via frpc too.

    Pass `share_server_address` to point at your own frps: stable URLs, no 72h expiry, no third
    party, no shared throughput ceiling.
    """

    def __init__(
        self,
        share_server_address: str | None = None,
        share_server_tls_certificate: str | None = None,
    ) -> None:
        super().__init__()
        self._share_server = share_server_address
        self._tls_cert = share_server_tls_certificate
        self._tunnel = None
        self._readers: list[threading.Thread] = []

    @classmethod
    def preflight(cls) -> None:
        try:
            from gradio.networking import setup_tunnel  # noqa: F401
        except Exception as exc:  # noqa: BLE001
            raise ForwardingError(
                f"gradio is required for GradioForwarder: {exc}"
            ) from exc

    def start(self, local_port: int, *, local_host: str = "127.0.0.1") -> str:
        from gradio.networking import setup_tunnel
        from gradio.tunneling import CURRENT_TUNNELS

        self.stop()
        share_token = secrets.token_hex(16)
        try:
            url = setup_tunnel(
                local_host=local_host,
                local_port=local_port,
                share_token=share_token,
                share_server_address=self._share_server,
                share_server_tls_certificate=self._tls_cert,
            )
        except Exception as exc:  # noqa: BLE001
            raise ForwardingError(f"gradio forward failed to open: {exc}") from exc
        # Gradio reads stdout only until the URL appears, and never reads stderr.
        # Drain both pipes for the lifetime of this child: a full pipe can block
        # frpc's logging thread and prevent it from maintaining the relay.
        self._tunnel = next(
            (tunnel for tunnel in CURRENT_TUNNELS if tunnel.share_token == share_token),
            None,
        )
        if self._tunnel is None or self._tunnel.proc is None:
            raise ForwardingError(
                "gradio returned a URL without a managed tunnel process"
            )
        for pipe in (self._tunnel.proc.stdout, self._tunnel.proc.stderr):
            if pipe is not None:
                reader = threading.Thread(target=self._drain, args=(pipe,), daemon=True)
                reader.start()
                self._readers.append(reader)
        self._local_port, self._url = local_port, url
        return url

    @staticmethod
    def _drain(pipe) -> None:
        try:
            for line in iter(pipe.readline, b""):
                text = line.decode(errors="replace").rstrip()
                if any(
                    marker in text.lower() for marker in ("error", "failed", "warn")
                ):
                    logger.warning("gradio tunnel: %s", text[:1000])
                else:
                    logger.debug("gradio tunnel: %s", text[:1000])
        except (OSError, ValueError):
            pass  # The owner may close the pipe during shutdown.

    def stop(self) -> None:
        if self._tunnel is not None:
            process = self._tunnel.proc
            self._tunnel.kill()
            if process is not None:
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
                for reader in self._readers:
                    reader.join(timeout=1)
                for pipe in (process.stdout, process.stderr):
                    if pipe is not None:
                        pipe.close()
            self._tunnel = None
        self._readers.clear()
        self._url = None


class CloudflareForwarder(PortForwarder):
    """`cloudflared`, either a quick forwards or a named one.

    Quick tunnels (no `tunnel_name`) need no account and print a `*.trycloudflare.com` URL on
    stderr, which we parse. Named forwards need `cloudflared login` beforehand but give a stable
    hostname that survives restarts, which is what you want once this is not a sweep any more.

    The URL arrives asynchronously on stderr, so `start` blocks until it appears or gives up. That
    wait is the entire reason this class is more code than GradioForwarder.
    """

    _URL_RE = re.compile(r"https://[-a-z0-9]+\.trycloudflare\.com")

    def __init__(
        self,
        tunnel_name: str | None = None,
        hostname: str | None = None,
        binary: str = "cloudflared",
        startup_timeout_s: float = 60.0,
    ) -> None:
        super().__init__()
        self._tunnel_name = tunnel_name
        self._hostname = hostname
        self._binary = binary
        self._startup_timeout_s = startup_timeout_s
        self._proc: subprocess.Popen | None = None

    @classmethod
    def preflight(cls, binary: str = "cloudflared") -> None:
        if shutil.which(binary) is None:
            raise ForwardingError(
                f"`{binary}` not found on PATH. Install it, or use GradioForwarder, which needs no "
                "binary because frpc ships with gradio."
            )

    def start(self, local_port: int, *, local_host: str = "127.0.0.1") -> str:
        self.preflight(self._binary)
        target = f"http://{local_host}:{local_port}"

        if self._tunnel_name:
            cmd = [self._binary, "tunnel", "run", "--url", target, self._tunnel_name]
        else:
            # `cloudflared tunnel --url`, not `cloudflared forward`. `forward` is an alias for
            # `cloudflared access`, a completely different feature: it exits without ever printing a
            # *.trycloudflare.com URL, so this path failed at startup every time it was selected.
            cmd = [self._binary, "tunnel", "--no-autoupdate", "--url", target]

        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )
        self._local_port = local_port

        # A named forwards serves a hostname we already know, so there is nothing to parse.
        if self._tunnel_name and self._hostname:
            self._url = f"https://{self._hostname}"
            return self._url

        url = self._await_url()
        if url is None:
            self.stop()
            raise ForwardingError(
                f"cloudflared printed no forward URL within {self._startup_timeout_s:.0f}s. "
                "Check that the binary can reach Cloudflare, or use GradioForwarder."
            )
        self._url = url
        return url

    def _await_url(self) -> str | None:
        """Read stderr until the URL appears, the process dies, or we run out of patience.

        `select` before `readline`, because `readline` BLOCKS until a newline arrives. A cloudflared
        that starts and then goes quiet — no URL, no crash — held the loop inside that call forever,
        so `startup_timeout_s` was advisory and `start()` could hang indefinitely instead of failing
        cleanly. Waiting on readability first means the deadline is honoured whatever the child does.
        """
        assert self._proc is not None and self._proc.stdout is not None
        stream = self._proc.stdout
        deadline = time.monotonic() + self._startup_timeout_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            if self._proc.poll() is not None:
                return None  # died during startup
            # Capped so process death is noticed promptly even while the pipe stays silent.
            ready, _, _ = select.select([stream], [], [], min(remaining, 0.2))
            if not ready:
                continue
            line = stream.readline()
            if not line:
                # EOF: the pipe closed, so nothing further will arrive on it.
                return None
            match = self._URL_RE.search(line)
            if match:
                return match.group(0)

    def stop(self) -> None:
        proc, self._proc, self._url = self._proc, None, None
        if proc is None or proc.poll() is not None:
            return
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


_FORWARDERS: dict[str, type[PortForwarder]] = {
    "direct": DirectExposure,
    "gradio": GradioForwarder,
    "cloudflare": CloudflareForwarder,
}


def make_forwarder(kind: str = "gradio", **kwargs) -> PortForwarder:
    """Build a forwarder by name, for CLI wiring (`--expose gradio|cloudflare|direct`)."""
    try:
        cls = _FORWARDERS[kind]
    except KeyError:
        raise ForwardingError(
            f"unknown port forwarder {kind!r}; choose one of {sorted(_FORWARDERS)}"
        ) from None
    return cls(**kwargs)
