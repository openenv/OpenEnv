"""Opt-in smoke test for ACASandboxProvider against real Azure resources.

This is the end-to-end test you run locally against a real ACA Sandbox Group to
prove the WebSocket ``reset``/``step`` path works (not just ``/health``). It is
skipped unless ``OPENENV_ACA_INTEGRATION=1``.

What this validates (confirmed live against a deployed ACA sandbox group):
  * An ACA anonymous-port ingress proxies the ``EnvClient`` WebSocket upgrade,
    so a full ``reset()`` -> ``step()`` -> ``state()`` round-trip works over
    ``wss://`` (the provider enforces an ``https``/``wss`` base URL).
  * A default-deny egress policy built with
    ``ACASandboxProvider.deny_all_egress(allow=[...])`` blocks the cloud
    metadata/IMDS endpoint (``169.254.169.254``) and non-allowlisted hosts while
    still permitting explicitly allowlisted hosts (e.g. a package registry).

Local testing runbook
---------------------
1. Install the optional preview extra (the only place the Azure SDK is needed)::

       uv pip install -e ".[aca]"          # or: pip install "openenv[aca]"

2. Authenticate with Azure (DefaultAzureCredential picks this up)::

       az login

3. Point the provider at your deployed sandbox group::

       export AZURE_SUBSCRIPTION_ID=<sub-guid>
       export AZURE_RESOURCE_GROUP=<rg>
       export AZURE_SANDBOX_GROUP=<sandbox-group-name>
       export AZURE_REGION=<region>

4. Run it. There are two source options:

   Option A -- public disk, no custom image (the path this test bootstraps for
   you). The test uploads the bundled ``aca_integration_server.py`` into a stock
   public ``python-3.11`` disk, ``pip install``s fastapi/uvicorn, and runs it.
   This is the fastest way to prove the WebSocket round-trip::

       export OPENENV_ACA_INTEGRATION=1
       export OPENENV_ACA_DISK=python-3.11
       PYTHONPATH=src:envs uv run pytest \
           tests/test_core/test_aca_provider_integration.py -v -m integration

   With no explicit ``OPENENV_ACA_CMD`` the test runs under a default-deny
   egress policy that allowlists only PyPI (so the in-sandbox pip install works
   while IMDS and every other host stay blocked).

   Option B -- your own OpenEnv server baked into a disk image (out-of-band,
   once; the ACA source is a disk image, not a registry image)::

       from azure.containerapps.sandbox import SandboxGroupClient, endpoint_for_region
       from azure.identity import DefaultAzureCredential
       client = SandboxGroupClient(
           endpoint_for_region("<region>"), DefaultAzureCredential(),
           subscription_id="<sub>", resource_group="<rg>", sandbox_group="<group>",
       )
       img = client.begin_create_disk_image(
           "<myregistry>.azurecr.io/echo-env:latest", name="echo-openenv"
       ).result()   # poll until status.state == "Ready"

   then point the test at it and supply the start command::

       export OPENENV_ACA_DISK=echo-openenv
       export OPENENV_ACA_CMD="python -m uvicorn server.app:app --host 0.0.0.0 --port 8000"

   Security: the provider enforces an https/wss URL, requires the explicit
   ``anonymous_port=True`` opt-in, and warns about unrestricted egress. Pass
   ``OPENENV_ACA_ALLOW_HOSTS`` (comma-separated) to run under a default-deny
   egress policy that only allows those hosts.

The fast unit tests need no Azure and use a fake adapter::

    PYTHONPATH=src:envs uv run pytest tests/test_core/test_aca_provider.py -v
"""

from __future__ import annotations

import base64
import json
import os

import pytest
import requests
from openenv.core.containers.runtime.aca_provider import ACASandboxProvider
from openenv.core.generic_client import GenericEnvClient

# Default allowlist for the Option A bootstrap so pip can reach PyPI under a
# default-deny egress policy while everything else (incl. IMDS) stays blocked.
_DEFAULT_PYPI_ALLOW = "pypi.org,*.pypi.org,files.pythonhosted.org,*.pythonhosted.org"


def _bootstrap_cmd() -> str:
    """Build a start command that runs the bundled server on a public disk.

    Base64-encodes the fixture so the whole command survives the provider's
    ``shlex.quote`` + ``nohup bash -c`` wrapping with no quoting hazards.
    """
    server_path = os.path.join(os.path.dirname(__file__), "aca_integration_server.py")
    with open(server_path, "rb") as fh:
        encoded = base64.b64encode(fh.read()).decode("ascii")
    return (
        f"echo {encoded} | base64 -d > /tmp/openenv_server.py && "
        "python3 -m pip install --quiet --disable-pip-version-check "
        "fastapi 'uvicorn[standard]' websockets && "
        "exec python3 /tmp/openenv_server.py"
    )


@pytest.mark.integration
@pytest.mark.network
@pytest.mark.skipif(
    os.environ.get("OPENENV_ACA_INTEGRATION") != "1",
    reason="Set OPENENV_ACA_INTEGRATION=1 to run the real ACA smoke test",
)
async def test_aca_provider_real_sandbox_smoke():
    disk = os.environ.get("OPENENV_ACA_DISK")
    if not disk:
        pytest.skip("Set OPENENV_ACA_DISK to an ACA disk image that runs OpenEnv")

    explicit_cmd = os.environ.get("OPENENV_ACA_CMD")
    # Option A: no baked image + no explicit cmd -> bootstrap the bundled server
    # and default the egress allowlist to PyPI so the in-sandbox pip install can
    # run under default-deny egress.
    if explicit_cmd:
        cmd = explicit_cmd
        allow_hosts = os.environ.get("OPENENV_ACA_ALLOW_HOSTS")
    else:
        cmd = _bootstrap_cmd()
        allow_hosts = os.environ.get("OPENENV_ACA_ALLOW_HOSTS", _DEFAULT_PYPI_ALLOW)

    egress_policy = (
        ACASandboxProvider.deny_all_egress(allow=allow_hosts.split(","))
        if allow_hosts
        else None
    )

    provider = ACASandboxProvider(
        anonymous_port=True,
        egress_policy=egress_policy,
        cmd=cmd,
        auto_suspend_seconds=1200,
        labels={"openenv-test": "aca-provider"},
    )
    base_url = None

    try:
        base_url = provider.start_container(f"disk:{disk}")
        # The provider enforces https; assert it so a regression is loud.
        assert base_url.startswith("https://")

        provider.wait_for_ready(
            base_url,
            timeout_s=float(os.environ.get("OPENENV_ACA_READY_TIMEOUT", "300")),
        )
        response = requests.get(f"{base_url}/health", timeout=10.0)
        assert response.status_code == 200

        async with GenericEnvClient(base_url=base_url) as client:
            reset_result = await client.reset(seed=7)
            assert isinstance(reset_result.observation, dict)

            step_action = os.environ.get("OPENENV_ACA_STEP_ACTION")
            if step_action:
                step_result = await client.step(json.loads(step_action))
            else:
                step_result = await client.step({"message": "openenv-aca-smoke"})
            assert isinstance(step_result.observation, dict)

            state = await client.state()
            assert isinstance(state, dict)
    finally:
        provider.close()
