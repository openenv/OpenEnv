"""Register a sandbox shell through NeMo's public function API."""

from nat.plugin_api import Builder, FunctionBaseConfig, FunctionInfo, register_function
from .shell import execute


class ShellConfig(FunctionBaseConfig, name="openenv_sandbox_shell"):
    timeout: float = 60.0


@register_function(config_type=ShellConfig)
async def sandbox_shell(config: ShellConfig, builder: Builder):
    async def shell(command: str) -> str:
        """Run a bash command in the task sandbox; returns exit code, stdout and stderr.

        Use this to inspect task files, run Python analysis and write the answer file.
        """
        return await execute(command, timeout=config.timeout)

    yield FunctionInfo.from_fn(shell, description=shell.__doc__)
