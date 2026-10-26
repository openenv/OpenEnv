"""Opt-in NeMo ReAct configuration for qualification on file-based tasks."""

import yaml
from harbor.agents.installed.nemo_agent import NemoAgent


class NemoShellProfile(NemoAgent):
    """Reuse Harbor's installation and endpoint configuration with a native NeMo agent."""

    def _generate_config_yaml(self, model_name: str, api_key: str) -> str:
        config = yaml.safe_load(super()._generate_config_yaml(model_name, api_key))
        llm_name = next(iter(config["llms"]))
        config["llms"][llm_name]["temperature"] = 0.8
        config["functions"] = {
            "shell": {"_type": "openenv_sandbox_shell", "timeout": 60}
        }
        config["workflow"] = {
            "_type": "react_agent",
            "llm_name": llm_name,
            "tool_names": ["shell"],
            "use_native_tool_calling": True,
            "max_tool_calls": 17,
            "max_history": 1000,
            "additional_instructions": "Use the shell tool to inspect task files and complete the requested work. Follow the task's submission protocol.",
        }
        return yaml.safe_dump(config, sort_keys=False)
