"""Resolve private server runtime settings for ThinkingBox episodes."""

import os
from dataclasses import dataclass
from typing import Any

from thinkingbox.common.config_types import SessionProxyConfig
from thinkingbox.common.credential_factory import create_credential
from thinkingbox.common.mcp_proxy_client import MCPProxyClient

from ..runtime import load_thinkingbox_config_with_sha256


SESSION_PROXY_URL = os.environ.get("OPENENV_TB_PROXY_URL", "http://127.0.0.1:7111")
AGENT = os.environ.get("OPENENV_TB_AGENT", "think")
THINKINGBOX_CONFIG = os.environ.get("OPENENV_TB_CONFIG", "")
PROXY_TIMEOUT = float(os.environ.get("OPENENV_TB_PROXY_TIMEOUT", "120"))
MAX_CONCURRENT_ENVS = int(os.environ.get("OPENENV_TB_MAX_CONCURRENT_ENVS", "8"))


@dataclass(frozen=True)
class RuntimeSettings:
    """Hold private native runtime settings for one episode."""

    proxy: SessionProxyConfig
    judge_model: Any | None = None
    judge_type: str = "legacy"
    user_model: Any | None = None
    user_can_end_conversation: bool = False
    config_sha256: str | None = None


def load_runtime_settings(
    config_path: str | None,
    default_proxy: SessionProxyConfig,
) -> RuntimeSettings:
    """Load native ThinkingBox configuration without exposing it on the wire."""
    if not config_path:
        return RuntimeSettings(proxy=default_proxy)
    parsed, config_sha256 = load_thinkingbox_config_with_sha256(config_path)
    return RuntimeSettings(
        proxy=parsed.mcp_proxy,
        config_sha256=config_sha256,
        judge_model=parsed.judge_model,
        judge_type=parsed.judge_type,
        user_model=parsed.user_model,
        user_can_end_conversation=parsed.user_can_end_conversation,
    )


def make_proxy_client(settings: RuntimeSettings) -> MCPProxyClient:
    """Construct the native Session Proxy client for private server use."""
    proxy = settings.proxy
    credential = (
        create_credential(proxy.credential) if proxy.credential is not None else None
    )
    return MCPProxyClient(
        endpoint=proxy.endpoint_url,
        timeout=proxy.timeout,
        use_dns_cache=proxy.use_dns_cache,
        credential=credential,
        client_certificate=proxy.client_certificate,
        trust_ca_path=proxy.trust_ca_path,
        headers=proxy.headers,
        max_retries_server_error=proxy.max_retries_server_error,
        retryable_server_errors=proxy.retryable_server_errors,
        always_json_output=proxy.always_json_output,
        geteffects_proxy_info=proxy.geteffects_proxy_info,
    )
