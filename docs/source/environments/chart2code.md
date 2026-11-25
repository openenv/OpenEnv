<!-- openenv-source: chart2code_env -->
# Chart2Code Environment

Image to code, with visual feedback. The agent sees a PNG of a chart and writes
matplotlib code that draws it again. It can render drafts to see what its code
draws, then submits once. The reward reads the submitted figure's artists, not
its pixels, and compares them with the target's: bar heights, line and scatter
points, wedge angles, text, colors and axes. It is deterministic and needs no
judge.

## Episode

1. `reset(seed=...)` returns the target chart (`target_png_base64`), its size in
   inches and its dpi, and the instructions (`prompt`).
2. `step(Chart2CodeAction(code=..., mode="render"))` runs the code and returns
   the PNG it draws (`rendered_png_base64`) or the traceback (`error`). No score
   comes back, so renders can't be used to climb the reward. There are
   `max_renders` of them (default 4).
3. `step(Chart2CodeAction(code=..., mode="submit"))` scores the code and ends the
   episode with a reward in [0, 1] and the per-component `scores`.

After the submission, further steps run nothing and return the final
observation until the next `reset()`.

```python
from envs.chart2code_env import Chart2CodeAction, Chart2CodeEnv

with Chart2CodeEnv(base_url="http://localhost:8000") as env:
    observation = env.reset(seed=3).observation
    code = my_model(observation.prompt, observation.target_png_base64)
    draft = env.step(Chart2CodeAction(code=code, mode="render")).observation
    result = env.step(Chart2CodeAction(code=code, mode="submit"))
    print(result.reward, result.observation.scores)
```

## Tasks

Every task is a chart spec sampled from a seed, and its target is drawn by the
spec's reference code with the same pinned matplotlib that reads submissions.
So the reference code always scores 1.0.

| Level | Charts |
|---|---|
| `easy` | One series as a bar, horizontal bar, line or pie chart, with its values drawn on the chart. |
| `medium` | Two or three series as grouped bars, lines or scatter points, with a legend and no value labels. |
| `hard` | Two charts side by side, of different types, one of them with a log y-axis. |

Splits: `train` takes any seed below 10^9. `test` holds 500 fixed tasks per
level (seeds from 10^9 up), which no train seed reaches. Pin a task with
`reset(task_index=..., split=..., level=...)`; a `seed` on the `test` split picks
task `seed % 500`.

## Reward

The submission first has to pass a gate, or the reward is 0:

- the code runs within `timeout_s` (default 30 s) and leaves a figure open;
- the figure has data in it;
- the figure holds no raster image (`imshow`, `figimage`, `OffsetImage`, ...),
  so pasting the target scores nothing.

Past the gate, four components are combined by weight:

| Component | Weight | What is compared |
|---|---|---|
| structure | 0.2 | Axes and the kind of each series (bar, barh, line, scatter, pie), and log scales. |
| data | 0.45 | Values of each matched series. Error is relative to the target's range (in log space on a log axis): full credit within 2%, none beyond 10%. |
| text | 0.2 | Title, axis labels, category tick labels, legend entries and pie labels, after lowercasing and collapsing spaces. Numbers, such as value labels and numeric ticks, are left out. |
| color | 0.15 | Distance between the colors of each matched series, or of each wedge. |

Matching is F1-style: missing series, points and words cost recall, and extra
ones cost precision, so spraying extra bars or labels lowers the score. The
figure is read through matplotlib's artists, so `ax.bar`, `plt.bar`, a pandas
`.plot.bar()` and seaborn draw the same chart to the same score.

## Running the code

Each render and submission runs in a new Python process, started from an
environment allowlist (no server secrets reach it), in an empty temporary
directory, with a timeout and a 2 GB memory cap on POSIX. The process only gets
the code; the spec and the target stay in the server. This is isolation from
the server's state, not a security sandbox: deploy the environment in its
container when the code comes from an untrusted model.

The code may import anything installed in the environment; matplotlib, numpy,
pandas and seaborn are.

## Configuration

| Argument | Default | Meaning |
|---|---|---|
| `level` | `"easy"` | Default level, overridable per `reset()`. |
| `split` | `"train"` | Default split, overridable per `reset()`. |
| `max_renders` | `4` | Drafts allowed before submitting. |
| `timeout_s` | `30.0` | Seconds the code may run per render or submission. |

## Running locally

```bash
PYTHONPATH=src:envs uv run --project envs/chart2code_env uvicorn chart2code_env.server.app:app --port 8000
PYTHONPATH=src:envs uv run --project envs/chart2code_env pytest tests/envs/test_chart2code_env.py
```

The `/web` playground shows the target next to the latest drawing, with the
scores below it.
