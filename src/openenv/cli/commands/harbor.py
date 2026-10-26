"""`openenv harbor` — run Harbor tasks with token-level capture.

Four commands, in the order you would use them:

    openenv harbor info                     what can this machine run right now?
    openenv harbor rollout --task-index 0   one rollout, end to end, no server
    openenv harbor serve                    the env server, for trainers and clients
    openenv harbor push                     the same server, deployed to a Space

`info` and `rollout` exist so the whole path can be exercised without standing up a server, which
makes the failure surface much smaller when something is wrong: if `rollout` works and `serve` does
not, the problem is the serving layer, not Harbor, the sandbox, the agent or capture.

Examples:

```bash
# what is usable, given the credentials on this machine
openenv harbor info --llm-url $LLM --dataset AdithyaSK/data_agent_rl_environment_eval

# one rollout on E2B with opencode
openenv harbor rollout \\
    --llm-url $LLM \\
    --dataset AdithyaSK/data_agent_rl_environment_eval \\
    --task-index 0 --harness opencode --sandbox e2b

# the same task on Modal with codex — harness and sandbox are per-rollout
openenv harbor rollout --llm-url $LLM --dataset $DS \\
    --task-index 0 --harness codex --sandbox modal
```
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Annotated, Any, Optional

import typer

app = typer.Typer(
    name="harbor",
    help="Run Harbor tasks with token-level capture",
    no_args_is_help=True,
)

_DATASET_HELP = (
    "Dataset spec: HF repo id, local dir, or Harbor `name@version`. Repeatable."
)
_LLM_HELP = "OpenAI-spec inference endpoint (vLLM). Required: there is no default, because a\nwrong or stale endpoint produces rollouts that look fine and carry no token ids."
_LLM_HELP_OPTIONAL = (
    "OpenAI-spec inference endpoint. Optional here: without it, `info` "
    "still reports sandboxes, datasets and harnesses."
)
_LLM_HELP_PUSH = (
    "The Space's own inference endpoint. Optional: without one, visitors connect their own model "
    "in the UI (see --visitor-endpoints)."
)
_KEY_HELP = (
    "Credential for the inference endpoint, for a hosted provider (OpenAI, Anthropic, HF "
    "Inference Providers). Defaults to $OPENENV_LLM_API_KEY. This is NOT the key the agent "
    "gets: that one is a capture session id, minted per rollout, and this value never leaves "
    "the server process."
)
_AUTH_HEADER_HELP = (
    "Header to send --api-key under. `Authorization` gets a `Bearer ` prefix; anything else "
    "(e.g. x-api-key) gets the raw key."
)


_SHARE_HELP = (
    "Let visitors of the UI run on this server's endpoint and key. --no-share-endpoint makes each "
    "visitor connect their own model. Default: shared."
)
_SHARE_HELP_PUSH = (
    "Let visitors of the Space's UI run on its endpoint and key, at your cost. Default: not shared, "
    "so each visitor connects their own model (signing in with Hugging Face, a token, or an endpoint)."
)
_VISITOR_HELP = (
    "Let visitors connect their own model in the UI: a Hugging Face token and model, or any "
    "OpenAI-compatible URL such as vLLM. Default: allowed."
)
_VISIBILITY_HELP = (
    "Who sees runs in the UI: `all` (every visitor sees every run) or `own` (each browser sees the "
    "runs it started). Default: all locally, own on a Space."
)
_HISTORY_HELP = "Keep finished UI runs on disk across restarts. Default: on locally, off on a Space."
_ROLLOUTS_HELP = (
    "Let visitors start rollouts from the UI. --no-rollouts makes it a read-only task browser. "
    "Default: on."
)
_PRIVATE_URLS_HELP = (
    "Let visitors connect an endpoint on a private or local address, such as http://localhost:8000/v1. "
    "Default: allowed only when the server listens on 127.0.0.1, since --host 0.0.0.0 makes this "
    "server call those addresses for anyone who can reach it."
)
_ADD_HELP = (
    "Let visitors add and remove Hub datasets from the UI. On a Space with a bucket they are copied "
    "into it; otherwise downloaded. Default: on only for a server on 127.0.0.1."
)
_UI_VARIABLES = (
    "OPENENV_HARBOR_UI_SERVER_ENDPOINT",
    "OPENENV_HARBOR_UI_VISITOR_ENDPOINTS",
    "OPENENV_HARBOR_RUN_HISTORY",
    "OPENENV_HARBOR_UI_ADD_DATASETS",
    "OPENENV_HARBOR_UI_ROLLOUTS",
    "OPENENV_HARBOR_RUN_VISIBILITY",
    "OPENENV_HARBOR_REWARD_KEY",
)
_REWARD_KEY_HELP = (
    "Which reward UI runs report for tasks with several and none named `reward`, or a "
    "comma-separated preference order. Unset, such a run shows each of them instead."
)


def _ui_env(
    share_endpoint: Optional[bool],
    visitor_endpoints: Optional[bool],
    run_visibility: str,
    run_history: Optional[bool],
    add_datasets: Optional[bool] = None,
    rollouts: Optional[bool] = None,
    private_urls: Optional[bool] = None,
    reward_key: str = "",
) -> dict[str, str]:
    """The UI's deployment settings as the variables `openenv.harbor.ui_settings` reads.

    Only flags that were given become variables, so an unset one keeps the default for where the
    server runs (a laptop or a Space).
    """
    if run_visibility and run_visibility not in ("all", "own"):
        raise typer.BadParameter("--run-visibility is `all` or `own`")
    out = {}
    for name, value in (
        ("OPENENV_HARBOR_UI_SERVER_ENDPOINT", share_endpoint),
        ("OPENENV_HARBOR_UI_VISITOR_ENDPOINTS", visitor_endpoints),
        ("OPENENV_HARBOR_RUN_HISTORY", run_history),
        ("OPENENV_HARBOR_UI_ADD_DATASETS", add_datasets),
        ("OPENENV_HARBOR_UI_ROLLOUTS", rollouts),
        # `serve` only, so not one of `_UI_VARIABLES`: on a Space these addresses are its own network
        ("OPENENV_HARBOR_UI_PRIVATE_URLS", private_urls),
    ):
        if value is not None:
            out[name] = "1" if value else "0"
    if run_visibility:
        out["OPENENV_HARBOR_RUN_VISIBILITY"] = run_visibility
    if reward_key.strip():
        out["OPENENV_HARBOR_REWARD_KEY"] = reward_key.strip()
    return out


def _remove_omitted_ui_variables(repo_id: str, supplied: dict[str, str]) -> None:
    """Reset omitted UI flags to deployment defaults on an incremental Space push.

    Space variables survive a push. Without this reconciliation, one old `--share-endpoint` remains
    enabled forever even when later pushes omit it and the CLI says the Space default is not shared.
    Only variables owned by this command are removed; unrelated operator configuration and every
    secret are left alone.
    """
    from huggingface_hub import HfApi

    api = HfApi()
    existing = api.get_space_variables(repo_id)
    for key in _UI_VARIABLES:
        if key in existing and key not in supplied:
            api.delete_space_variable(repo_id=repo_id, key=key)
            print(f"variable  {key} reset to its Space default")


def _split(values: Optional[list[str]]) -> list[str]:
    """Accept both `--dataset a --dataset b` and `--dataset a,b`."""
    out: list[str] = []
    for value in values or []:
        out.extend(v.strip() for v in value.split(",") if v.strip())
    return out


def _hub_not_found(exc: Exception) -> bool:
    """Whether a Hub operation failed specifically because its resource does not exist."""
    if isinstance(exc, FileNotFoundError):
        return True
    response = getattr(exc, "response", None)
    return getattr(response, "status_code", None) == 404


@app.command("info")
def info(
    llm_url: Annotated[str, typer.Option("--llm-url", help=_LLM_HELP_OPTIONAL)] = "",
    model: Annotated[
        str,
        typer.Option(
            "--model",
            help="Served model id. Auto-detected if the engine serves exactly one.",
        ),
    ] = "",
    dataset: Annotated[
        Optional[list[str]], typer.Option("--dataset", help=_DATASET_HELP)
    ] = None,
    api_key: Annotated[str, typer.Option("--api-key", help=_KEY_HELP)] = "",
    auth_header: Annotated[
        str, typer.Option("--auth-header", help=_AUTH_HEADER_HELP)
    ] = "Authorization",
    env_file: Annotated[
        str, typer.Option("--env-file", help="dotenv file with provider credentials.")
    ] = "",
    verbose: Annotated[
        bool,
        typer.Option("--verbose", help="List every harness, not only validated ones."),
    ] = False,
    json_output: Annotated[
        bool, typer.Option("--json", help="Emit machine-readable JSON.")
    ] = False,
) -> None:
    """Report engine, sandboxes, datasets and harnesses available here."""
    from openenv.harbor.startup import prepare

    caps = prepare(
        llm_url=llm_url or None,
        model=model or None,
        datasets=_split(dataset) or None,
        env_file=env_file or None,
        require_llm=False,
        quiet=True,
        api_key=api_key or None,
        auth_header=auth_header,
    )
    print(
        json.dumps(caps.to_dict(), indent=2)
        if json_output
        else caps.render(verbose=verbose)
    )


@app.command("rollout")
def rollout(
    llm_url: Annotated[str, typer.Option("--llm-url", help=_LLM_HELP)],
    model: Annotated[str, typer.Option("--model", help="Served model id.")] = "",
    dataset: Annotated[
        Optional[list[str]], typer.Option("--dataset", help=_DATASET_HELP)
    ] = None,
    task_index: Annotated[
        int, typer.Option("--task-index", help="Index into the split.")
    ] = 0,
    harness: Annotated[
        str, typer.Option("--harness", help="Seam name, or `module:Class`.")
    ] = "opencode",
    sandbox: Annotated[
        str,
        typer.Option(
            "--sandbox", help="Harbor environment type, e.g. e2b | modal | docker."
        ),
    ] = "e2b",
    n: Annotated[
        int, typer.Option("-n", "--n-tasks", help="Run this many consecutive tasks.")
    ] = 1,
    port: Annotated[
        int, typer.Option("--port", help="Local port for the capture proxy.")
    ] = 8100,
    expose: Annotated[
        str,
        typer.Option(
            "--expose",
            help="How the sandbox reaches the capture proxy: gradio | cloudflare | direct.",
        ),
    ] = "gradio",
    trials_dir: Annotated[
        str, typer.Option("--trials-dir", help="Where Harbor writes trial artifacts.")
    ] = "",
    reward_key: Annotated[
        str,
        typer.Option("--reward-key", help="Which reward key is the training signal."),
    ] = "",
    keep_sandbox: Annotated[
        bool,
        typer.Option("--keep-sandbox", help="Leave sandboxes alive for debugging."),
    ] = False,
    force_build: Annotated[
        bool,
        typer.Option(
            "--force-build",
            help="Rebuild the sandbox image, bypassing the content-hash cache. Needed when a task pins deps loosely and its cached image has drifted.",
        ),
    ] = False,
    api_key: Annotated[str, typer.Option("--api-key", help=_KEY_HELP)] = "",
    auth_header: Annotated[
        str, typer.Option("--auth-header", help=_AUTH_HEADER_HELP)
    ] = "Authorization",
    env_file: Annotated[
        str, typer.Option("--env-file", help="dotenv file with provider credentials.")
    ] = "",
    out: Annotated[str, typer.Option("--out", help="Write the result JSON here.")] = "",
) -> None:
    """Run one or more rollouts without starting a server."""
    from openenv.harbor.runner import run_batch

    datasets = _split(dataset)
    if not datasets:
        raise typer.BadParameter("--dataset is required")

    results = asyncio.run(
        run_batch(
            llm_url=llm_url,
            model=model or None,
            dataset=datasets[0],
            task_indices=list(range(task_index, task_index + max(1, n))),
            harness=harness,
            sandbox=sandbox,
            port=port,
            expose=expose,
            trials_dir=Path(trials_dir) if trials_dir else None,
            reward_key=reward_key,
            keep_sandbox=keep_sandbox,
            force_build=force_build,
            env_file=env_file or None,
            api_key=api_key or None,
            auth_header=auth_header,
        )
    )

    if out:
        Path(out).write_text(json.dumps([r.model_dump() for r in results], indent=2))
        print(f"\nwrote {out}")
    raise typer.Exit(0 if all(r.ok for r in results) else 1)


@app.command("serve")
def serve(
    llm_url: Annotated[str, typer.Option("--llm-url", help=_LLM_HELP)] = "",
    model: Annotated[str, typer.Option("--model", help="Served model id.")] = "",
    dataset: Annotated[
        Optional[list[str]], typer.Option("--dataset", help=_DATASET_HELP)
    ] = None,
    max_output_tokens: Annotated[
        int,
        typer.Option(
            "--max-output-tokens",
            help=(
                "Cap what an AGENT may request per turn. The default of 8192 is exactly what some "
                "harnesses ask for (opencode), so the clamp does nothing for them and their first "
                "call can exceed a small context window. A real agent turn is short; 4096 is ample."
            ),
        ),
    ] = 8192,
    host: Annotated[str, typer.Option("--host")] = "0.0.0.0",
    port: Annotated[
        int, typer.Option("--port", help="Env server port (faces the trainer).")
    ] = 8000,
    capture_port: Annotated[
        int,
        typer.Option("--capture-port", help="Capture proxy port (faces the sandbox)."),
    ] = 8100,
    expose: Annotated[
        str,
        typer.Option(
            "--expose",
            help="How the sandbox reaches the capture proxy: gradio | cloudflare | direct.",
        ),
    ] = "gradio",
    api_key: Annotated[str, typer.Option("--api-key", help=_KEY_HELP)] = "",
    auth_header: Annotated[
        str, typer.Option("--auth-header", help=_AUTH_HEADER_HELP)
    ] = "Authorization",
    env_file: Annotated[str, typer.Option("--env-file")] = "",
    share_endpoint: Annotated[
        Optional[bool],
        typer.Option("--share-endpoint/--no-share-endpoint", help=_SHARE_HELP),
    ] = None,
    visitor_endpoints: Annotated[
        Optional[bool],
        typer.Option("--visitor-endpoints/--no-visitor-endpoints", help=_VISITOR_HELP),
    ] = None,
    run_visibility: Annotated[
        str, typer.Option("--run-visibility", help=_VISIBILITY_HELP)
    ] = "",
    run_history: Annotated[
        Optional[bool],
        typer.Option("--run-history/--no-run-history", help=_HISTORY_HELP),
    ] = None,
    add_datasets: Annotated[
        Optional[bool],
        typer.Option("--add-datasets/--no-add-datasets", help=_ADD_HELP),
    ] = None,
    rollouts: Annotated[
        Optional[bool],
        typer.Option("--rollouts/--no-rollouts", help=_ROLLOUTS_HELP),
    ] = None,
    private_urls: Annotated[
        Optional[bool],
        typer.Option("--private-urls/--no-private-urls", help=_PRIVATE_URLS_HELP),
    ] = None,
    reward_key: Annotated[
        str, typer.Option("--reward-key", help=_REWARD_KEY_HELP)
    ] = "",
) -> None:
    """Serve Harbor tasks over the OpenEnv Task API and MCP.

    Two ports on purpose. The env server faces the trainer on an internal network; the capture proxy
    faces the sandbox and is the only thing published. Sharing one port would expose the env
    server as soon as the capture proxy became reachable.
    """
    from openenv.harbor.serving import serve_harbor

    os.environ.update(
        _ui_env(
            share_endpoint,
            visitor_endpoints,
            run_visibility,
            run_history,
            add_datasets,
            rollouts,
            private_urls,
            reward_key,
        )
    )
    # The UI's laptop defaults (this machine's token, private URLs) are for a loopback-only server.
    os.environ["OPENENV_HARBOR_UI_HOST"] = host
    serve_harbor(
        llm_url=llm_url,
        model=model or None,
        max_output_tokens=max_output_tokens or None,
        datasets=_split(dataset),
        host=host,
        port=port,
        capture_port=capture_port,
        expose=expose,
        env_file=env_file or None,
        api_key=api_key or None,
        auth_header=auth_header,
    )


@app.command("push")
def push(
    llm_url: Annotated[str, typer.Option("--llm-url", help=_LLM_HELP_PUSH)] = "",
    repo_id: Annotated[
        str, typer.Option("--repo-id", help="Target, e.g. your-org/harbor-env.")
    ] = "",
    model: Annotated[str, typer.Option("--model", help="Served model id.")] = "",
    dataset: Annotated[
        Optional[list[str]], typer.Option("--dataset", help=_DATASET_HELP)
    ] = None,
    private: Annotated[
        bool,
        typer.Option(
            "--private",
            help="Create the Space private. The sandbox then cannot reach the capture proxy, so rollouts are not possible; use it only to park a deployment.",
        ),
    ] = False,
    hardware: Annotated[
        str, typer.Option("--hardware", help="Space hardware, e.g. cpu-basic.")
    ] = "",
    api_key: Annotated[str, typer.Option("--api-key", help=_KEY_HELP)] = "",
    auth_header: Annotated[
        str, typer.Option("--auth-header", help=_AUTH_HEADER_HELP)
    ] = "Authorization",
    env_file: Annotated[
        str,
        typer.Option(
            "--env-file", help="dotenv whose provider keys become Space SECRETS."
        ),
    ] = "",
    bucket: Annotated[
        str,
        typer.Option(
            "--bucket",
            help="Storage bucket holding the task suites. Defaults to a bucket named after the Space. Pass `none` to skip the bucket and let the Space download datasets instead.",
        ),
    ] = "",
    public_bucket: Annotated[
        Optional[bool],
        typer.Option(
            "--public-bucket/--private-bucket",
            help="Visibility of the Space's bucket. Private unless --public-bucket: it holds the task "
            "suites and, with run history on, every visitor's runs. Given for an existing bucket, it "
            "changes that bucket's visibility; not given, an existing bucket is left as it is.",
        ),
    ] = None,
    hf_login: Annotated[
        bool,
        typer.Option(
            "--hf-login/--no-hf-login",
            help="Let visitors sign in with Hugging Face to use their own account for Inference "
            "Providers (the `inference-api` scope), instead of pasting a token.",
        ),
    ] = True,
    recreate: Annotated[
        bool,
        typer.Option(
            "--recreate",
            help="Delete the Space first, then deploy fresh. A Space keeps variables, secrets, volumes and any file a previous push wrote, so an incremental deploy is not a clean test of what this bundle produces.",
        ),
    ] = False,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Show what would be pushed and stop.")
    ] = False,
    share_endpoint: Annotated[
        Optional[bool],
        typer.Option("--share-endpoint/--no-share-endpoint", help=_SHARE_HELP_PUSH),
    ] = None,
    visitor_endpoints: Annotated[
        Optional[bool],
        typer.Option("--visitor-endpoints/--no-visitor-endpoints", help=_VISITOR_HELP),
    ] = None,
    run_visibility: Annotated[
        str, typer.Option("--run-visibility", help=_VISIBILITY_HELP)
    ] = "",
    run_history: Annotated[
        Optional[bool],
        typer.Option("--run-history/--no-run-history", help=_HISTORY_HELP),
    ] = None,
    add_datasets: Annotated[
        Optional[bool],
        typer.Option("--add-datasets/--no-add-datasets", help=_ADD_HELP),
    ] = None,
    rollouts: Annotated[
        Optional[bool],
        typer.Option("--rollouts/--no-rollouts", help=_ROLLOUTS_HELP),
    ] = None,
    reward_key: Annotated[
        str, typer.Option("--reward-key", help=_REWARD_KEY_HELP)
    ] = "",
) -> None:
    """Deploy this environment to a Hugging Face Space.

    Takes the same arguments as `serve`, because a deployed Space needs exactly the same
    configuration — they are forwarded as Space variables, while provider credentials from
    `--env-file` are forwarded as Space *secrets* so they are not readable from the repo.
    """
    from pathlib import Path

    from openenv.cli.commands.push import push as _push
    from openenv.harbor.startup import load_env_file

    if not repo_id:
        raise typer.BadParameter("--repo-id is required, e.g. your-org/harbor-env")

    datasets = _split(dataset)
    ui_variables = _ui_env(
        share_endpoint,
        visitor_endpoints,
        run_visibility,
        run_history,
        add_datasets,
        rollouts,
        reward_key=reward_key,
    )
    if not share_endpoint and visitor_endpoints is False:
        # On a Space the endpoint is shared only when asked, so unset counts as not shared.
        raise typer.BadParameter(
            "--no-visitor-endpoints needs --share-endpoint: on a Space visitors use its endpoint "
            "only when it is shared, so they would have no model to run with."
        )
    if not llm_url and visitor_endpoints is False:
        raise typer.BadParameter(
            "--llm-url is required with --no-visitor-endpoints: a Space with no engine of its own "
            "that accepts none from visitors cannot run anything, and finding that out after "
            "deploying is worse than finding it out now."
        )
    if not llm_url:
        print(
            "NOTE: no --llm-url. Visitors run rollouts on a model they connect themselves (a "
            "Hugging Face token, or their own endpoint)."
        )
    elif share_endpoint is None:
        print(
            "NOTE: UI visitors connect their own model; the endpoint serves the Task API and MCP. "
            "Pass --share-endpoint to let them run on it too, at your cost."
        )

    if private:
        # A hosted deployment serves the capture proxy at <space-url>/capture. On a private Space
        # that URL demands an auth header the agent inside the sandbox does not send, so every model
        # call 401s and the rollout records nothing. Worth a warning rather than a hard failure:
        # parking a private deployment is legitimate, running rollouts against one is not.
        print(
            "WARNING: a private Space is not reachable from a sandbox. The capture proxy is "
            "served at <space-url>/capture, and a private Space requires an auth header the "
            "agent will not send, so rollouts will capture no model calls. Deploy public for "
            "rollouts. The proxy still refuses callers without a registered session id, which "
            "is what keeps a public deployment from being an open relay."
        )

    # HF datasets are attached as read-only volumes rather than downloaded. A Harbor suite is
    # thousands of small files (13k+ for a 2.2k-task dataset), so downloading on first request takes
    # minutes and burns the Space's ephemeral disk; a mount is instant and survives restarts. The
    # server needs no special case for it, because a mount path is just a local directory and
    # `resolve_task_dirs` already accepts one.
    # Every suite lives under one bucket, mounted at /data, named after the Space so the two stay
    # obviously paired. `--bucket none` opts out and falls back to downloading.
    hf_specs = [spec for spec in datasets if _looks_like_hf_repo(spec)]
    bucket = "" if bucket.lower() == "none" else (bucket or repo_id)
    mounts = {spec: f"{_MOUNT_ROOT}/{spec.replace('/', '__')}" for spec in hf_specs}

    # Non-secret configuration travels as plain Space variables.
    variables = {
        "OPENENV_LLM_URL": llm_url,
        "ENABLE_WEB_INTERFACE": "true",
        **ui_variables,
    }
    if bucket:
        # Where the UI puts datasets added from the page: a server-side copy into this bucket,
        # read back through the mount, as for the datasets pushed here.
        variables["OPENENV_HARBOR_BUCKET"] = bucket
        variables["OPENENV_HARBOR_BUCKET_MOUNT"] = _MOUNT_ROOT
    if datasets:
        variables["OPENENV_DATASETS"] = ",".join(datasets)
    if model:
        variables["OPENENV_MODEL"] = model
    # The header NAME is configuration, not a credential, so it belongs here. Its value never is.
    if auth_header and auth_header != "Authorization":
        variables["OPENENV_LLM_AUTH_HEADER"] = auth_header

    # Provider credentials travel as secrets. Only the keys the sandboxes need — never the whole
    # dotenv, which usually holds unrelated tokens.
    load_env_file(env_file or None)
    # Sandbox credentials, plus the keys a task's own verifier may need. A grader that cannot run
    # returns no reward at all, which is reported as `reward=None` rather than 0, so the rollout is
    # correctly not scored as a wrong answer, but it is also not usable for training. The DataAgent
    # grader reads OPENAI_API_KEY for its LLM-judge tier, and without it every semantically correct
    # answer that is not an exact string match goes ungraded.
    wanted = (
        "E2B_API_KEY",
        "MODAL_TOKEN_ID",
        "MODAL_TOKEN_SECRET",
        "DAYTONA_API_KEY",
        "HF_TOKEN",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GEMINI_API_KEY",
        # The upstream inference credential. A SECRET rather than a variable: Space variables are
        # readable from the repo page, and this one buys inference against a paid endpoint.
        "OPENENV_LLM_API_KEY",
    )
    secrets = {k: os.environ[k] for k in wanted if os.environ.get(k)}
    # An --api-key passed on the command line outranks the dotenv, matching every other command.
    if api_key:
        secrets["OPENENV_LLM_API_KEY"] = api_key

    # src/openenv/cli/commands/harbor.py -> repo root is parents[4].
    # An installed wheel has no sibling envs/ dir, so fall back to $OPENENV_HARBOR_ENV_DIR.
    env_dir = Path(
        os.environ.get("OPENENV_HARBOR_ENV_DIR")
        or Path(__file__).resolve().parents[4] / "envs" / "harbor_env"
    )
    if not (env_dir / "openenv.yaml").is_file():
        raise typer.BadParameter(
            f"no harbor_env package at {env_dir}. Set OPENENV_HARBOR_ENV_DIR to its location "
            "(an installed openenv wheel does not ship the envs/ directory)."
        )
    # `openenv.harbor` does not exist in any released wheel, so a Space that pip-installs `openenv`
    # imports the release and dies on `No module named 'openenv.harbor'`. When pushing from a source
    # checkout, bundle the working tree instead; the Dockerfile puts /app/env ahead of site-packages.
    source_pkg = Path(__file__).resolve().parents[2]  # .../src/openenv
    bundle = source_pkg if (source_pkg / "harbor").is_dir() else None

    print(f"env       {env_dir}")
    print(f"repo      {repo_id}{'  (private)' if private else ''}")
    print(f"llm       {llm_url}")
    print(
        f"datasets  {', '.join(datasets) or '(none, set OPENENV_DATASETS on the Space)'}"
    )
    print(f"variables {sorted(variables)}")
    print(f"secrets   {sorted(secrets)}   (values never printed)")
    print(f"openenv   {'bundled from ' + str(source_pkg) if bundle else 'from PyPI'}")
    if bucket:
        print(f"bucket    {bucket}  ->  {_MOUNT_ROOT}   ({len(hf_specs)} suite(s))")
    for spec, path in mounts.items():
        print(
            f"mount     {spec}  ->  {path}"
            + ("  (via bucket)" if bucket else "  (read-only, not downloaded)")
        )
    if dry_run:
        print("\ndry run: nothing pushed")
        return

    if recreate:
        _delete_space(repo_id)

    if bucket:
        # Created even with no datasets to copy: datasets added from the page go into it too.
        _fill_bucket(bucket, hf_specs, public=public_bucket)

    with tempfile.TemporaryDirectory(prefix="openenv-harbor-push-") as tmp:
        staged = Path(tmp) / "env"
        shutil.copytree(
            env_dir,
            staged,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".venv"),
        )
        if bundle is not None:
            # A hosted Space mounts the capture proxy on its own app and reaches it at the Space's
            # public URL, so the forwarding backends are dead code there. They are excluded rather
            # than merely unused: shipping code that shells out to `cloudflared` into a Space is
            # both pointless and the kind of thing platform abuse checks reject. `cli` goes for the
            # same reason, it is 36 files the server never imports.
            shutil.copytree(
                bundle,
                staged / "openenv",
                ignore=shutil.ignore_patterns(
                    "__pycache__", "*.pyc", "forwarding.py", "cli"
                ),
            )
        if hf_login:
            _enable_hf_login(staged / "README.md")
        _prune_removed_files(repo_id, staged)
        _push(
            directory=str(staged),
            repo_id=repo_id,
            private=private,
            hardware=hardware or None,
            env_vars=[f"{k}={v}" for k, v in variables.items()],
            secrets=[f"{k}={v}" for k, v in secrets.items()],
        )
        _remove_omitted_ui_variables(repo_id, ui_variables)

    # After the push, because volumes attach to a Space that already exists and `--recreate` has
    # just deleted it. Setting them triggers one more rebuild, which is why this is last.
    attached = (
        _attach_bucket(repo_id, bucket, mounts)
        if bucket
        else _mount_datasets(repo_id, mounts)
    )
    if attached:
        # Only now is it safe to point the server at mount paths. Until the mount is confirmed,
        # `OPENENV_DATASETS` holds repo ids, so an unattached volume degrades to downloading rather
        # than to a server pointed at directories that do not exist.
        from huggingface_hub import HfApi

        HfApi().add_space_variable(
            repo_id=repo_id,
            key="OPENENV_DATASETS",
            value=",".join(mounts.get(d, d) for d in datasets),
        )
        print("mount     OPENENV_DATASETS switched to mount paths")


# What visitors may sign in for: Inference Providers, billed to their own account. Nothing else.
_HF_LOGIN = (
    "hf_oauth: true",
    "hf_oauth_scopes:",
    "  - inference-api",
    "hf_oauth_expiration_minutes: 480",
)


def _enable_hf_login(readme: Path) -> None:
    """Turn on "Sign in with Hugging Face" for the Space, in its README front matter.

    The Hub then provides the OAuth app (`OAUTH_CLIENT_ID` and friends), and the UI offers sign-in
    as a way to use Inference Providers with the visitor's own account.
    """
    text = readme.read_text() if readme.is_file() else ""
    if text.startswith("---"):
        end = text.index("\n---", 3)
        # Only the front matter counts: the README's own text may well mention `hf_oauth:`.
        if any(line.startswith("hf_oauth:") for line in text[:end].splitlines()):
            return
        text = text[:end] + "\n" + "\n".join(_HF_LOGIN) + text[end:]
    else:
        text = "---\n" + "\n".join(_HF_LOGIN) + "\n---\n\n" + text
    readme.write_text(text)


def _prune_removed_files(repo_id: str, staged: Path) -> None:
    """Delete files on the Space that this push no longer produces.

    `push` uploads but never deletes, so a file dropped from the bundle keeps running in the
    deployment forever. That is not a tidiness point: the first version of this command shipped the
    port-forwarding backends, and removing them locally left the deployed Space still carrying code
    that shells out to `cloudflared`, which is exactly what a platform abuse check objects to. A
    deployment has to reflect the bundle, not the union of every bundle ever pushed.

    Only the bundled `openenv/` subtree is pruned. Everything else in the Space may legitimately have
    been added out of band (a README edit through the web UI, a `.gitattributes`), and deleting a
    file this command never wrote is not its business.
    """
    from huggingface_hub import CommitOperationDelete, HfApi

    api = HfApi()
    try:
        remote = api.list_repo_files(repo_id, repo_type="space")
    except Exception as exc:  # noqa: BLE001 - a new Space has nothing to prune
        print(f"prune     skipped ({type(exc).__name__}); the Space may not exist yet")
        return

    local = {str(p.relative_to(staged)) for p in staged.rglob("*") if p.is_file()}
    stale = sorted(f for f in remote if f.startswith("openenv/") and f not in local)
    if not stale:
        return

    print(f"prune     {len(stale)} file(s) no longer in the bundle, e.g. {stale[0]}")
    api.create_commit(
        repo_id=repo_id,
        repo_type="space",
        operations=[CommitOperationDelete(path_in_repo=f) for f in stale],
        commit_message="Remove files no longer part of the harbor_env bundle",
    )


# Where dataset volumes are attached inside the Space container.
_MOUNT_ROOT = "/data"

# Mirrors `openenv.harbor.tasks._DATASET_ROOT`; kept local so the CLI does not import the
# harbor extra just to compute a path.
_DATASET_CACHE = Path(
    os.environ.get("OPENENV_DATASET_CACHE")
    or (Path.home() / ".cache" / "openenv" / "harbor-datasets")
)


def _looks_like_hf_repo(spec: str) -> bool:
    """True for `owner/name`, false for a local path or a Harbor `name@version`."""
    return (
        spec.count("/") == 1
        and "@" not in spec
        and not spec.startswith((".", "/", "~"))
    )


def _mount_datasets(repo_id: str, mounts: dict[str, str]) -> bool:
    """Attach each dataset repo to the Space as a read-only volume.

    Downloading a Harbor suite inside a Space is the slow path twice over: thousands of small files
    fetched one round trip at a time, onto a disk that is wiped on restart, so the cost is paid again
    on every rebuild. A mounted repo is available as ordinary files immediately.

    Volumes are replaced wholesale by the API, so anything already attached is read first and kept.

    Returns:
        `bool`: Whether the volumes are confirmed attached. `False` means the caller must keep using
            repo ids and let the Space download, which is slower but works.
    """
    if not mounts:
        return False
    try:
        from huggingface_hub import HfApi, Volume
    except ImportError:
        print(
            "mount     skipped: this huggingface_hub has no Volume support; "
            "the Space will download datasets instead"
        )
        return False

    api = HfApi()
    existing: list[Any] = []
    with contextlib.suppress(Exception):
        existing = [
            v
            for v in _attached_volumes(api, repo_id)
            if getattr(v, "mount_path", None) not in set(mounts.values())
        ]

    volumes = existing + [
        Volume(type="dataset", source=spec, mount_path=path, read_only=True)
        for spec, path in sorted(mounts.items())
    ]
    try:
        api.set_space_volumes(repo_id=repo_id, volumes=volumes)
    except Exception as exc:  # noqa: BLE001 - a Space that cannot mount still works by downloading
        print(
            f"mount     failed ({type(exc).__name__}: {str(exc)[:160]}). The Space will download "
            "datasets instead, which is slow but functional."
        )
        return False

    # Accepting the call is not evidence that the volume exists. Read it back, because the failure
    # mode of trusting it is a server configured to read directories that were never mounted.
    attached = _attached_mount_paths(api, repo_id)
    if not set(mounts.values()) <= attached:
        print(
            "mount     not confirmed: the Space reports no attached volumes, so the datasets will "
            "be downloaded instead. Attach them from the Space settings if you want the mount."
        )
        return False
    print(f"mount     attached {len(mounts)} dataset volume(s)")
    return True


def _delete_space(repo_id: str) -> None:
    """Delete the Space so the next push is a clean deployment.

    A Space accumulates state a push does not own: variables and secrets set by earlier runs, mounted
    volumes, and every file any previous push wrote. That makes an incremental deploy a poor test,
    because it can succeed on leftovers the bundle no longer produces. Deleting first means what runs
    is exactly what this command uploaded.

    Deliberately destructive, so it only ever happens behind `--recreate`.
    """
    from huggingface_hub import HfApi

    api = HfApi()
    try:
        api.delete_repo(repo_id=repo_id, repo_type="space")
        print(f"recreate  deleted {repo_id}")
    except Exception as exc:  # noqa: BLE001 - nothing to delete is the expected first-run case
        print(f"recreate  nothing to delete ({type(exc).__name__})")


def _fill_bucket(bucket: str, specs: list[str], public: bool | None = None) -> None:
    """Create `bucket` if missing and copy each task suite into it, server side.

    A new bucket is private unless `public` is `True`. An existing one keeps its visibility unless
    `public` says otherwise: changing who can read a bucket is not something to do by default.

    `copy_files` copies by xet hash: the Hub moves the references, nothing is downloaded here and
    nothing is re-uploaded. That is the difference between seconds and the ~47k-file upload a local
    sync performs, and it is why the bucket is filled before the Space exists rather than after.

    Suites already present are skipped, so adding a dataset to a later `push` copies only the new
    one and leaves the rest untouched.
    """
    from huggingface_hub import HfApi

    api = HfApi()
    try:
        existing = api.bucket_info(bucket)
    except Exception as exc:  # noqa: BLE001 - Hub exception classes vary across client versions
        if not _hub_not_found(exc):
            raise
        existing = None
    if existing is None:
        api.create_bucket(bucket, private=not public, exist_ok=True)
        print(f"bucket    {bucket} created ({'public' if public else 'private'})")
    else:
        private = bool(existing.private)
        if public is not None and private == public:
            api.update_bucket_settings(bucket, private=not public)
            print(f"bucket    {bucket} is now {'public' if public else 'private'}")
        elif not private and public is None:
            print(
                f"bucket    {bucket} is PUBLIC. Anyone can read it, including run history if that is "
                "on. Pass --private-bucket to make it private."
            )

    try:
        present = {
            entry.path.split("/", 1)[0]
            for entry in api.list_bucket_tree(bucket)
            if getattr(entry, "path", "")
        }
    except Exception:  # noqa: BLE001 - a brand new bucket may not be listable yet
        present = set()

    for spec in specs:
        prefix = spec.replace("/", "__")
        if prefix in present:
            print(f"bucket    {spec} already present, skipped")
            continue
        print(
            f"copy      hf://datasets/{spec} -> hf://buckets/{bucket}/{prefix}  (server side)"
        )
        api.copy_files(f"hf://datasets/{spec}/", f"hf://buckets/{bucket}/{prefix}/")


def _attach_bucket(repo_id: str, bucket: str, mounts: dict[str, str]) -> bool:
    """Mount `bucket` on the Space and confirm it attached.

    Returns:
        `bool`: Whether the mount is confirmed. `False` leaves the caller on repo ids so the Space
            downloads rather than reading a mount that may not be there.
    """
    from huggingface_hub import HfApi, Volume

    api = HfApi()
    api.set_space_volumes(
        repo_id=repo_id,
        volumes=[Volume(type="bucket", source=bucket, mount_path=_MOUNT_ROOT)],
    )
    if _MOUNT_ROOT not in _attached_mount_paths(api, repo_id):
        print(
            f"mount     not confirmed: no volume at {_MOUNT_ROOT}. Datasets will be downloaded "
            "instead. Attach the bucket from the Space settings to use the mount."
        )
        return False
    print(f"mount     {bucket} attached at {_MOUNT_ROOT}")
    return bool(mounts)


def _attached_volumes(api: Any, repo_id: str) -> list[Any]:
    """Volumes currently mounted on `repo_id`.

    Read through `space_info().runtime`, not `get_space_runtime()`. The latter is served by an
    endpoint that does not carry a `volumes` key at all, so it always answers `None` and a check
    built on it reports every mount as missing. That false negative is worse than no check: it makes
    a working mount look broken and sends the caller down the slow path forever.
    """
    with contextlib.suppress(Exception):
        runtime = api.space_info(repo_id).runtime
        if runtime is not None:
            return list(runtime.volumes or [])
    return []


def _attached_mount_paths(api: Any, repo_id: str) -> set[str]:
    """Mount paths currently attached to `repo_id`."""
    return {getattr(v, "mount_path", None) for v in _attached_volumes(api, repo_id)}
