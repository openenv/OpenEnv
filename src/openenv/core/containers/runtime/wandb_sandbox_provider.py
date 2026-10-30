"""W&B Serverless Sandboxes provider for running OpenEnv environments."""

from __future__ import annotations

from typing import Any

from .cwsandbox_provider import CWSandboxProvider


class WandbSandboxProvider(CWSandboxProvider):
    """Container provider that runs OpenEnv servers in W&B Sandboxes.

    `wandb.sandbox` exposes the same cwsandbox-shaped SDK surface as the
    `cwsandbox` package, but uses W&B auth (`wandb login` / `WANDB_API_KEY`).
    Runtime lifecycle behavior is inherited from [`~openenv.core.containers.runtime.cwsandbox_provider.CWSandboxProvider`].
    """

    def __init__(self, *args: Any, sdk: Any | None = None, **kwargs: Any) -> None:
        super().__init__(*args, sdk=_resolve_wandb_sandbox_sdk(sdk), **kwargs)

    @classmethod
    def preflight(cls, *, sdk: Any | None = None) -> None:
        """Validate that W&B sandbox auth works before launching an environment."""
        resolved_sdk = _resolve_wandb_sandbox_sdk(sdk)
        try:
            resolved_sdk.Sandbox.list().result()
        except Exception as exc:
            auth_error = getattr(resolved_sdk, "CWSandboxAuthenticationError", None)
            if isinstance(auth_error, type) and isinstance(exc, auth_error):
                raise SystemExit(
                    f"W&B Sandboxes auth check failed: {exc}. "
                    "Run `wandb login` or set WANDB_API_KEY and try again."
                ) from exc
            raise


def _resolve_wandb_sandbox_sdk(sdk: Any | None = None) -> Any:
    """Return the W&B sandbox SDK module, setting OpenEnv metadata if supported."""
    resolved_sdk = sdk if sdk is not None else _import_wandb_sandbox()
    set_integration_metadata = getattr(resolved_sdk, "set_integration_metadata", None)
    if callable(set_integration_metadata):
        set_integration_metadata("openenv")
    return resolved_sdk


def _import_wandb_sandbox() -> Any:
    try:
        return __import__("wandb.sandbox", fromlist=["sandbox"])
    except ImportError as exc:
        raise RuntimeError(
            "WandbSandboxProvider requires wandb.sandbox. Install W&B with "
            "sandbox support, then run `wandb login` or set WANDB_API_KEY."
        ) from exc
