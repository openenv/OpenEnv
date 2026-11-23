---
title: Data Analysis Environment Server
emoji: 📊
colorFrom: indigo
colorTo: green
sdk: docker
pinned: false
app_port: 8000
base_path: /web
datasets:
  - FineEnvs/SmolDataEnvs
tags:
  - openenv
---

# Data Analysis Environment

Train and evaluate LLMs on **answering questions about real data files** from a live Python session, with a
verifiable reward.

Each episode is one task from [SmolDataEnvs](https://huggingface.co/datasets/FineEnvs/SmolDataEnvs): a question about
one or more CSV, TSV or JSON files taken from real data-science notebooks over Kaggle datasets. The agent explores the
files with pandas in a persistent, sandboxed CPython session and submits a short answer, which the dataset's own
deterministic grader scores (exact match, then numeric with tolerances, lists, and symbolic equivalence, no LLM judge).

The session runs real CPython, like a notebook kernel, so any valid pandas works (`df[['a', 'b']]`,
`s.str.split(',', expand=True)[0]`, named aggregations, merges). An AST interpreter rejects much of this, which turns
correct analysis into tool errors.

## Quick start

```bash
# From the repo root
docker build -t data-analysis-env:latest -f envs/data_analysis_env/server/Dockerfile envs/data_analysis_env
docker run --rm -p 8000:8000 data-analysis-env:latest
# Optional: --cap-add SYS_ADMIN adds a network namespace for model code (see "Sandbox")
```

```python
from openenv.core.env_server.mcp_types import CallToolAction

from data_analysis_env import DataAnalysisEnv

with DataAnalysisEnv(base_url="http://localhost:8000").sync() as env:
    result = env.reset(split="eval", seed=0)
    meta = result.observation.metadata
    print(meta["prompt"])  # the question, the file paths and the rules

    print(
        env.call_tool(
            "run_python",
            code=f"import pandas as pd\ndf = pd.read_csv({meta['files'][0]!r})\ndf.shape",
        )
    )

    result = env.step(
        CallToolAction(tool_name="submit_answer", arguments={"answer": "0.83"})
    )
    print(result.reward, result.done)  # 1.0 or 0.0, True
```

`call_tool` returns the tool's text only. The reward arrives on the observation of the step that calls
`submit_answer`, so call it through `step`.

[`examples/data_analysis_env_episode.py`](../../examples/data_analysis_env_episode.py) plays a full scripted episode
through the tools.

## Tools

| Tool | What it does |
|---|---|
| `run_python(code)` | Runs code in the episode's session and returns stdout, then stderr. Variables persist between calls, and the value of a last-line expression is printed, as in a notebook. Errors come back as `ExceptionType: message` after a traceback of the model's own lines only. Warnings are shown but are not errors. |
| `submit_answer(answer)` | Grades the answer and ends the episode. Only the first answer counts. |

## Reset arguments

| Argument | Default | Meaning |
|---|---|---|
| `split` | `DATA_ANALYSIS_SPLIT` or `eval` | `train` (5,000 tasks), `eval` (144) or `test` (250, held out and harder). `eval_verified` (61) and `test_verified` (85), the checked subsets (see [Data](#data)). |
| `task_id` | random | A specific task of the split. Random tasks are drawn among those whose files are all CSV, TSV or JSON and under the size cap. |
| `seed` | `None` | Reseeds the task choice: the same seed picks the same task, and the same sequence on the unseeded resets that follow. |
| `episode_id` | generated | Episode identifier, kept in the state. |
| `max_python_calls` | 30 | `run_python` calls allowed (at most 200). Past it, the tool asks for an answer. |

The reset observation's metadata holds `prompt`, `task_id`, `split`, `question`, `files` (absolute paths, as model code
sees them), `difficulty_tier` and `tools`. Harnesses can use the bundled `prompt` or build their own from `question` and
`files`.

## Reward

`submit_answer` scores 1.0 when the grader accepts the answer, with the row's own `reward_mode`, `atol` and `rtol`, and
0.0 otherwise (also for an empty answer or one over 1,000 characters). The state records `grade_method` (`exact`,
`numeric`, `numeric_scaled`, `list`, `math_verify` or `miss`). An episode that never submits has no reward.
The reward is on the observation of the submitting step only: later steps of the episode return `done=True` and no
reward, and `state.reward` keeps it.

For harnesses that end an episode on a reply without a tool call, `data_analysis_env.extract_answer(text)` reads the
answer from that reply (the last `\boxed{}`, else the last `Answer:` or `Final answer:` line, else the whole reply), so
it can be submitted and graded the same way.

The gold answer never leaves the server process: it is not in the prompt, the observations, the state or the logs, and
the split is kept in memory only, never on disk where model code could read it.

## Data

Nothing is bundled with the environment.

- **Tasks**: SmolDataEnvs (MIT license) at a pinned version (`DATASET_REVISION` in `server/tasks.py`), downloaded once
  per server into memory.
- **Verified splits**: [SmolDataEnvs](https://huggingface.co/datasets/FineEnvs/SmolDataEnvs) v1.1 adds `eval_verified`
  (61 tasks) and `test_verified` (85): golds checked with an independent solution, ambiguous or seed-dependent tasks
  left out, and no data file shared with `train`. It also fixes 5 golds in `eval`.
  `eval_verified` is the recommended split for evaluation.
- **Data files**: downloaded on first use of each task from the public Hugging Face bucket the dataset row points to,
  into `<DATA_ANALYSIS_TASKS_ROOT>/<split>/<task_id>/`, read-only for model code. Files over
  `DATA_ANALYSIS_MAX_FILE_MB` (10 MiB) are not served. They come from public Kaggle datasets, each under its own
  license.
- **Grader**: SmolDataEnvs' `grader.py`, vendored unmodified at the same commit in `server/grader.py` (a test checks its
  hash).

Every download is anonymous: requests carry no `Authorization` header, whatever is in the server's environment, so the
server never needs a token.

Some tasks fit a model, so scikit-learn is available in the session next to pandas and numpy.

### Licenses

- The task rows (questions, answers, grading settings) and `server/grader.py` come from SmolDataEnvs, MIT license as
  declared on its dataset card. The vendored grader keeps a provenance header with the MIT text.
- The data files belong to the Kaggle datasets they come from, each under its own license, and the bucket that serves
  them declares none. That is why they are not in this repository or in the image: the environment downloads them from
  the public bucket when a task is used, and anyone running it is responsible for using them within those licenses.

## Sandbox

Model code runs as real CPython, so it can do whatever its process can. Each session runs in its own worker process,
forked from a spawner that has pandas, numpy and scikit-learn loaded, with these layers, from the strongest:

1. **A user per session.** When the server is root on Linux (the Docker image runs it as root on purpose), each
   worker switches to its own unprivileged uid (from 61000, no groups, no capabilities) before any model code runs. It
   can't read the server's environment, memory or root-only files, another session's home, or signal another session's
   processes. When a session ends, every process left with its uid is killed, and its files in `/tmp`, `/var/tmp`,
   `/dev/shm`, `/run/lock` and `/dev/mqueue` and its SysV IPC objects (shared memory, message queues, semaphores) are
   removed, before the uid is used again.
2. **No network.** Two mechanisms, and which ones apply depends on the host. The server says so once, at startup, in
   the log line `data_analysis_env sandbox: workers run as ..., network: ...`:

   | Log says | What model code gets |
   |---|---|
   | `isolated (namespace and seccomp)` | Both mechanisms below. |
   | `blocked by seccomp (no namespace: <reason>)` | The seccomp filter only. This is Docker's default (no `CAP_SYS_ADMIN`), a non-root server, `DATA_ANALYSIS_ISOLATE_NETWORK=0`, or a runtime that refuses `unshare` even so (Hugging Face Jobs, for example). |
   | `isolated (namespace, <reason>)` | The namespace only. |
   | `NOT isolated (<reasons>)` | Neither. Only the audit hook (4) stops sockets, and it can be bypassed. This is macOS and Windows, Linux on a CPU other than x86_64 or aarch64, or a kernel that refuses the filter. |

   - **seccomp filter** (Linux on x86_64 or aarch64, no capability needed): each worker installs it before model code
     runs, and it can't be removed from inside the process. It refuses `socket(2)` for every address family and
     `io_uring_setup(2)`, so model code can't open a network connection or connect to a Unix socket. The spawner
     checks at startup that the kernel enforces it.
   - **network namespace** (Linux, server running as root, and `unshare(CLONE_NEWNET)` allowed, which takes
     `CAP_SYS_ADMIN`, e.g. `docker run --cap-add SYS_ADMIN`): the spawner moves into a new, empty namespace before it
     forks any worker, so workers have no interface, not even loopback.
3. **Limits.** A hard timeout per call (30 s, then the worker is killed and the session restarts, with a message
   saying so), 4 GiB of address space, at most 64 processes per session user, 256 open files, 64 MiB per written file,
   no core files. The session's home may hold 512 MiB and 100,000 files and directories: the server checks it every
   second while a call runs and after it, and past either cap the call is stopped and the session restarts with an
   empty home, with a message saying so. Files written outside the home, which only code past the audit hook can do,
   are not counted. The worker's environment is an allowlist (`PATH`, locale, `HOME`), never a copy of the server's, so
   tokens and keys don't reach it. The server and the spawner are non-dumpable, so a process of the same uid can't
   ptrace them or read their memory. The server reads the worker's replies as bounded JSON, never pickle.
4. **An audit hook** (PEP 578), armed before model code: reads only under the task's data, the session's home and the
   Python installation, writes (and opening a directory) only under the session's home (its working directory), and no
   sockets, subprocesses, signals, `ctypes`, `cffi` or `pyarrow` (its readers skip audit events), no `gc.get_referrers` and similar escapes.
   Audit hooks can be bypassed from Python, so this is defense in depth, not a boundary.

When the server is **not root** (local development, or platforms that run containers as a non-root user, such as
Hugging Face Docker Spaces), layer 1 is not available: workers run as the server's user and only layers 3 and 4
separate sessions. The non-dumpable server still keeps its memory and environment out of reach. Run the server as root
in its own container when model code is untrusted, which is the case in RL training.

Residual risks, by design:

- Code that gets past the audit hook can read anything world-readable on the host. Where the startup log says
  `network: NOT isolated`, it can also open connections: with the internet it could fetch the public dataset and look
  up the gold answer, which would make the reward meaningless. For training, run the server on Linux (x86_64 or
  aarch64) and check that the log says `blocked by seccomp` or `isolated`.
- Without a uid per session, code past the hook could interfere with other sessions on the same server.

`tests/containers/check_data_analysis_sandbox.py` checks these properties in the production image, as root with a fake
token in the server's environment, both with and without `CAP_SYS_ADMIN`.

## Deployment

The env is marked `SKIP_HF_DEPLOYMENT`: Hugging Face Spaces don't run the container as root, so sessions couldn't get
a uid each, and the server needs network access to fetch tasks. Run it locally, in your own container, or next to the
trainer.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `DATA_ANALYSIS_SPLIT` | `eval` | Default split for `reset` |
| `DATA_ANALYSIS_TASKS_ROOT` | `<tmp>/data_analysis_env_tasks` | Where task files are downloaded |
| `DATA_ANALYSIS_MAX_FILE_MB` | `10` | Largest data file a task may have |
| `DATA_ANALYSIS_EXEC_TIMEOUT_S` | `30` | Hard limit per `run_python` call |
| `DATA_ANALYSIS_MEM_LIMIT_MB` | `4096` | Address-space limit per session |
| `DATA_ANALYSIS_SANDBOX_UID_BASE`, `DATA_ANALYSIS_SANDBOX_UID_COUNT` | `61000`, `1024` | Uids for sessions when root. Uids that belong to an account are skipped. |
| `DATA_ANALYSIS_ISOLATE_NETWORK` | `1` | `0` skips the workers' network namespace |
| `MAX_CONCURRENT_ENVS` | `64` | Concurrent WebSocket sessions |

### Running the server inside a training job

The server can also run next to the trainer in the same machine or container (`uvicorn
data_analysis_env.server.app:app --host 127.0.0.1 --port 8000`). Start it as root, so sessions get their own users, and
keep the Python environment it runs from readable by other users: workers import modules from it after they drop
privileges. An interpreter installed under `/root` (mode 0700) is not, so install one elsewhere (for example with
`UV_PYTHON_INSTALL_DIR=/opt/python`). The trainer's own secrets stay out of reach as long as they live in its
environment or in files only root can read, never on a command line (`/proc/<pid>/cmdline` is world-readable) or in
world-readable files.

## Development and testing

```bash
# From the repo root. tests/data_analysis_runtime locks the env against the repo's core.
uv sync --project tests/data_analysis_runtime
PYTHONPATH=src:envs uv run --project tests/data_analysis_runtime pytest \
  tests/envs/test_data_analysis_env.py tests/envs/test_data_analysis_sandbox.py -m "not network"

# The OS boundary, in the production image
docker build -t data-analysis-env -f envs/data_analysis_env/server/Dockerfile envs/data_analysis_env
docker run --rm --network none --cap-add SYS_ADMIN --cap-add SYS_PTRACE \
  -e HF_TOKEN=hf_FAKEFAKEFAKEFAKEFAKEFAKEFAKEFAKE00 -e EXPECT_NETNS=1 \
  --mount type=bind,src="$PWD/tests/containers/check_data_analysis_sandbox.py",dst=/checks.py,readonly \
  --entrypoint /app/.venv/bin/python data-analysis-env /checks.py
```

## Project structure

```
data_analysis_env/
├── __init__.py            # DataAnalysisEnv, DataAnalysisState, extract_answer
├── answers.py             # extract_answer: the answer in a final text reply
├── client.py              # DataAnalysisEnv (MCPToolClient)
├── models.py              # DataAnalysisState, AVAILABLE_TOOLS
├── openenv.yaml
├── pyproject.toml
└── server/
    ├── app.py             # FastAPI app
    ├── data_analysis_environment.py  # MCP tools, reset, reward
    ├── tasks.py           # split loading, task files, prompt
    ├── grading.py         # runs the grader in a helper process
    ├── grader.py          # SmolDataEnvs' grader, vendored
    ├── sandbox.py         # spawner, workers, users, limits, audit hook
    ├── session.py         # the notebook-like Python session
    └── Dockerfile
```

## Citation

The tasks and the grader come from SmolDataEnvs:

```bibtex
@misc{fineenvs,
  author = {Kolavi, Adithya S},
  title  = {FineEnvs: Open Source RL Environments for LLM Agents},
  year   = {2026},
  url    = {https://github.com/adithya-s-k/FineEnvs}
}
```
