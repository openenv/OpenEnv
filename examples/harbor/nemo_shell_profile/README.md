# NeMo shell qualification profile

This opt-in profile runs NVIDIA NeMo Agent Toolkit 1.9.0's native ReAct agent
with a shell tool inside the Harbor task sandbox. It qualifies that concrete
workflow; Harbor's default single-call chat workflow remains unchanged.

Harbor installs this directory through `workflow_package`. Select
`openenv.harbor.nemo_profile:NemoShellProfile`, `llm_type=openai`, and
`version=1.9.0`, with `OPENAI_BASE_URL` and `OPENAI_API_KEY` pointing to the
OpenEnv capture session. The profile reuses Harbor's provider YAML generation.

The shell uses the sandbox filesystem and runs bash with a 60-second timeout.
Do not run this workflow on a host containing unrelated workloads. Tool failures
retain their exit code; a timeout terminates the command's process group.

The live qualification driver selects this explicitly with
`--nemo-profile shell-1.9.0` and records that selection with the evidence.
