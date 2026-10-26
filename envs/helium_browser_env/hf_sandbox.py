"""Run the packaged Helium browser environment with HFSandboxProvider."""

import os

from openenv.core.containers.runtime.hf_sandbox_provider import HFSandboxProvider

from .client import BrowserClient
from .models import BrowserAction


def main():
    image = os.environ.get("HELIUM_BROWSER_IMAGE")
    if not image:
        raise RuntimeError("set HELIUM_BROWSER_IMAGE to the packaged environment image")

    env_vars = {
        "BROWSER_SANDBOX": "hf",
        "BROWSER_MAX_STEPS": os.getenv("BROWSER_MAX_STEPS", "20"),
        "BROWSER_ACTION_WAIT_SECONDS": os.getenv(
            "BROWSER_ACTION_WAIT_SECONDS", "1.5"
        ),
        "BROWSER_EXPLICIT_WAIT_SECONDS": os.getenv(
            "BROWSER_EXPLICIT_WAIT_SECONDS", "3"
        ),
    }

    with HFSandboxProvider(
        image=image,
        flavor="cpu-basic",
        env_vars=env_vars,
    ) as provider:
        base_url = provider.start_container()
        provider.wait_for_ready(base_url, timeout_s=300.0)

        with BrowserClient(base_url=base_url).sync() as browser:
            result = browser.reset(
                start_url=os.getenv("BROWSER_START_URL", "https://example.com")
            )
            result = browser.step(BrowserAction(op="scroll", dy=500))
            print(result.observation.url)


if __name__ == "__main__":
    main()
