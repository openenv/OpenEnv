---
title: Pelican SVG Environment
emoji: 🚲
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
app_port: 8000
tags:
  - openenv
---

# Pelican SVG Environment

Blind vector drawing, scored. The model is asked for an SVG of an animal riding
a vehicle and never sees the result, so it has to hold the spatial arrangement
in its head and emit coordinates for it. This is the check Simon Willison
popularised as "generate an SVG of a pelican riding a bicycle", turned into
something you can run repeatedly, at scale, and train against.

![Pelican SVG web playground](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/openenv/pelican_svg.png)

*The `/web` playground after submitting an SVG: the rendered drawing and its reward breakdown.*

## What this is, and what it is not

**Not a model ranking.** Simon Willison, whose prompt this is, is explicit that
"the correlation between pelican performance and actual model quality has been
mostly severed now", and ends with "don't go using pelicans to compare models"
([Kimi K3, and what we can still learn from the pelican
benchmark](https://simonwillison.net/2026/Jul/16/kimi-k3/)). Our own numbers
agree with him: on the canonical task, across 139 scored samples from 7 frontier models,
136 scored a perfect 1.000 on the structural layer and 47 scored a perfect 1.000 overall.
It is saturated.

**Not a contamination study either.** [Dylan
Castillo](https://dylancastillo.co/posts/pelicanmaxxing.html) already ran that,
across an 8 by 6 grid and 1008 SVGs from 7 frontier models, and found little
evidence that labs optimise for the pelican. The extra animals and vehicles here
are ours and serve a smaller purpose: checking that the scorer measures drawing
ability rather than one memorised picture. Only `pelican_bicycle` is Simon's, and
it is the default.

**What it is** is a reproducible, executable target. The same code that scores a
frontier model over an API scores a 0.6B model mid-training, the reward is dense
enough to run GRPO against, and the whole thing deploys as a Space. That is the
part a prompt cannot do:

- [`examples/pelican_svg_eval.py`](https://github.com/huggingface/OpenEnv/blob/main/examples/pelican_svg_eval.py) scores any set of models through HF Inference Providers.
- [`examples/pelican_svg_grpo.py`](https://github.com/huggingface/OpenEnv/blob/main/examples/pelican_svg_grpo.py) trains against it with TRL, runnable on HF Jobs.

The interesting question stops being "which model draws the best pelican" and
becomes "can a small model be taught to, and does it learn to draw or to game
the scorer".

## The task catalogue

6 animals by 5 vehicles, so 30 tasks. All three
scoring layers adapt to whichever one is served, so nothing about the pipeline is special
to the pelican:

- **The judge's checklist is generated per task** from the features below. A pelican is
  asked about a throat pouch, a capybara about a blunt snout, an octopus about curling
  tentacles.
- **The geometry layer reads `wheels`** from the vehicle, and drops the wheel-pair checks
  entirely for a single-wheeled vehicle rather than failing them.
- **The anti-cheat terms are derived from the task**, so `axolotl_scooter` forbids
  `axolotl, salamander, amphibian, scooter, kick scooter` and not the pelican's words.

### Animals

| subject | features the judge asks about | counts as partial recognition |
| --- | --- | --- |
| `pelican` | a long beak, a throat pouch under the beak | bird, seabird, waterbird |
| `flamingo` | a long thin neck, long thin legs | bird, waterbird |
| `capybara` | a blunt rectangular snout, small rounded ears | rodent, animal |
| `axolotl` | feathery external gills on the head, a wide flat smiling mouth | salamander, amphibian, animal |
| `octopus` | a large bulbous head, multiple curling tentacles | cephalopod, animal |
| `hedgehog` | a coat of spines, a small pointed snout | animal |

### Vehicles

| vehicle | wheels in side view | features the judge asks about |
| --- | --- | --- |
| `bicycle` | 2 | two wheels of similar size, a frame joining the wheels, handlebars |
| `unicycle` | 1 | a single wheel, a seat post rising from the wheel |
| `tandem bicycle` | 2 | two wheels, an extended frame with two saddles |
| `scooter` | 2 | two small wheels, a deck with an upright steering column |
| `skateboard` | 2 | a flat deck, wheels under the deck |

### How much of the grid has actually been run

**All 30 combinations have been scored end to end by a live model. None crashed and none
was rejected by the gate.** The counts below are how much traffic each cell has actually
seen, which is a different question from whether it works:

| animal | `bicycle` | `unicycle` | `tandem bicycle` | `scooter` | `skateboard` |
| --- | --- | --- | --- | --- | --- |
| `pelican` | **224** | 1 | 1 | 1 | 1 |
| `flamingo` | 1 | 1 | 1 | **16** | 1 |
| `capybara` | 1 | **16** | 1 | 1 | 1 |
| `axolotl` | 1 | 1 | **16** | 1 | 1 |
| `octopus` | 1 | 1 | 1 | 1 | 1 |
| `hedgehog` | 1 | 1 | 1 | 1 | 1 |

Read a `1` as "the plumbing works", not as a result. It says the task builds, the prompt
renders, the gate admits the reply, the geometry resolves against that vehicle's wheel
count and the judge returns a checklist. It says nothing about how well the scoring
behaves on that combination, and it cannot: one sample has no variance to report.

Across the 30-task sweep, reward ranged 0.362 to 1.000 with a median of 0.838, structure
averaged 0.905 and the judged semantic score 0.785. The weakest cell was
`hedgehog_tandem-bicycle` at 0.362, where the judge would not accept the animal. Three of
the 30 were corrected upward by the `<use>` fix described below, `capybara_unicycle` most
of all, from 0.333 to 1.000 on structure.

`pelican_bicycle` is Simon Willison's original prompt and the default. It is also the only
task the GRPO example trains on: `examples/pelican_svg_grpo.py` pins it deliberately,
since that script is a small demonstration that the environment is trainable rather than
an attempt to teach a model the whole grid.

## Scoring

Three layers, ordered by cost. Each earns the right to run the next, so a run
against a model producing garbage stays cheap.

### 1. Gate, free, pass or fail

Rejects anything that is not an honest attempt to draw. A rejected submission
scores zero and never reaches the judge.

| Code | Meaning |
| --- | --- |
| `no_svg_in_response` / `truncated_svg` | No SVG, or one cut off mid-generation. Kept separate because "the model refused" and "the harness cut it off" are different facts. |
| `unparseable` / `render_failed` | Not well-formed, declares a DTD, or will not rasterise. |
| `embedded_raster` | An `<image>` element or a `data:image` URI. |
| `external_reference` | An `href` pointing off-document. |
| `forbidden_element` | `<script>`, `<foreignObject>` and friends. |
| `text_label` / `text_heavy` | Writing the answer instead of drawing it. |
| `blank_canvas` / `content_off_canvas` | Nothing visible, or geometry placed outside the viewBox. |
| `too_few_elements` / `too_many_elements` | Not a drawing, or a denial-of-service payload. |

The two cheats worth knowing about:

- **Embedded raster.** A base64 PNG inside an `<image>` renders as a completely
  convincing picture. A vision judge scores it highly. It is only visible in the
  source, which is why source inspection runs before anything looks at pixels.
- **Text label.** `<text>a pelican riding a bicycle</text>` also renders, and
  also fools a careless judge.

### 2. Structure, free, weight 0.35

Geometry over the parsed document with transforms applied and coordinates
normalised by the viewBox, so nothing depends on the units the model picked.
Seven checks: wheel count, wheels similar in size, level, sensibly far apart,
something spanning between them, a rider present, and the rider proportionate
to the wheelbase.

Wheels are found as a row of similar-sized round shapes at similar height, with
the lowest row preferred. Not by picking the biggest circle: see the correction
below for what that cost.

Deliberately shape-level. It cannot tell a pelican from a capybara and does not
try. What it can do is answer questions no judge should be trusted with because
the answer is arithmetic.

### 3. Vision judge, one call pair per sample, weight 0.65

- **A blind caption.** The judge is shown the picture and asked what it is, with
  no mention of the task. If it says "a pelican on a bicycle" unprompted, that
  is the strongest available evidence, and a leading question cannot inflate it.
  In testing this mattered: asked "is this a pelican riding a bicycle?" the
  judge said yes; asked to describe the same image cold, it said "a cartoon
  duck".
- **A binary feature checklist.** Individual yes-or-no questions rather than a
  0-10 rating, because individual features are reproducible and a global rating
  drifts.

Failing the riding posture costs 75% of the semantic score rather than all of
it. Posture is a spatial relation and the least reliable item on the checklist,
so hanging the whole component on that one boolean gives the judge's worst call
the loudest vote.


## What training against it revealed

Four GRPO runs through TRL on HF Jobs, 80 steps each. Three with the judge
disabled, so the reward is deterministic arithmetic, and one with `--judge-in-reward`. The
training reward climbed in every one of them, which turns out to say very little.

What says more is re-scoring each checkpoint on 24 fresh samples *with* the judge on, and
comparing against the untrained model. Every run measures that first, so there are five
independent measurements of the starting point:

| | untrained, 5 probes | 1.7B, judge off | 1.7B, judge on |
| --- | --- | --- | --- |
| structural score | 0.374 ± 0.064 | **0.619** and 0.643 | 0.321 |
| judged semantic score | 0.0049 ± 0.0042 | 0.0075 and 0.0139 | 0.0023 |
| submissions rejected by the gate | 3 to 13 of 24 | 1 and 2 of 24 | 11 of 24 |

The judge-off runs finish 3.8 standard deviations above the untrained structural mean. The
judge-on run finishes *inside* the untrained band, 0.8 below the mean, and its semantic
score went down. Turning the judge on drops the weight on structure from 1.0 to 0.35, so
it dilutes the only term that teaches anything and replaces it with one that teaches
nothing: across all 48 probe drawings, before and after, the judge never once answered yes
to the riding question.

It learned two real things: to emit well-formed SVG, and to build bicycle-like geometry,
both on samples it never trained on. It did not learn to draw a bird. **Twice,
independently**, which is why this is stated as a finding rather than an anecdote.

Looking at the probe drawings rather than the table sharpens it. Between before and after,
samples with no detectable structure at all fell from 9 of 24 to 2 of 24 and the median
structural score rose from 0.43 to 0.71, so something real was learned. But the single
best semantic score in the probe went *down*, 0.045 to 0.028, and the word "bird" appears
in one blind caption before training and none after. The output got more measurable and
slightly less bird-like. Weights and all 48 probe drawings are public for both the
judge-off run,
[`pelican-svg-grpo-Qwen3-1.7B`](https://huggingface.co/sergiopaniego/pelican-svg-grpo-Qwen3-1.7B),
and the judge-on run,
[`pelican-svg-grpo-Qwen3-1.7B-judged`](https://huggingface.co/sergiopaniego/pelican-svg-grpo-Qwen3-1.7B-judged),
with both training curves at
[`pelican-svg-grpo-curves`](https://huggingface.co/spaces/sergiopaniego/pelican-svg-grpo-curves).

Read that carefully, because the obvious label is the wrong one. **The judge was switched
off during training, so the semantic score was never part of the reward.** The policy
optimised exactly what it was paid for. This is not a model subverting its objective, it
is a demonstration that **the deterministic layer alone is an insufficient proxy for the
task**: two round shapes of equal size at equal height, a bar between them and any
blob above the axle line scores 1.000 on all seven checks. Four primitives and no
animal, and it is the exact figure the trained policy converged on.

Whether a model *would* game a reward the judge is part of was the obvious follow-up, and
the answer is that at this size the question does not arise: the judged term is almost
always zero, so there is nothing to game. In 24 fresh samples the untrained 1.7B scored
non-zero on the semantic component **twice**, and GRPO takes its gradient from the
variance inside a group of completions, so a term that is zero for every member of a group
contributes nothing however heavily it is weighted. The same judge scores non-zero on 138
of 139 frontier samples. Before spending GPU time on a judged reward, measure how often it
is non-zero on the base model you are about to train.

`examples/pelican_svg_grpo.py` ships the probe that produced all of this: it scores fresh
samples with the judge before and after, and flags a run where the structural gain
outpaces the judged gain. That ratio was 23 to 1 on the first 1.7B run and 45 to 1 on the
second.

## Usage

```python
from envs.pelican_svg_env import PelicanSvgAction, PelicanSvgEnv

with PelicanSvgEnv(base_url="http://localhost:8000") as env:
    observation = env.reset(task_id="capybara_unicycle").observation
    result = env.step(PelicanSvgAction(response=my_model(observation.prompt)))
    print(result.reward, result.observation.feedback)
```

Pin the task with `task_id` for any benchmark run. Without it every reset draws
a fresh task and two models are never asked the same question.

Run the server locally:

```bash
PYTHONPATH=src:envs uv run uvicorn pelican_svg_env.server.app:app --port 8000
```

### Configuration

| Argument | Default | Effect |
| --- | --- | --- |
| `subject`, `vehicle` | sampled | Pin the task for every episode. |
| `held_out_only` | `False` | Exclude the canonical pelican-and-bicycle pair. |
| `enable_judge` | `True` | `False` scores on the deterministic layers alone. Note that passing `judge=None` asks for the default judge; this is the flag that turns it off. |
| `return_image` | `False` | Include the rendered PNG in the observation. |

Environment variables: `HF_TOKEN` for the judge, `PELICAN_SVG_JUDGE_MODEL` to
change it from `Qwen/Qwen2.5-VL-72B-Instruct`, `PELICAN_SVG_DISABLE_JUDGE=1` to
force offline scoring, `MAX_CONCURRENT_ENVS` for the maximum number of
concurrent WebSocket sessions (default `8`).

With no judge configured, structure carries the full weight. A judge that is
configured but *fails* is treated differently: the semantic component stays at
zero and `judged` is set false on the observation, so a harness can drop or
retry the sample. Renormalising there would mean a submission scores higher
precisely when nobody could look at it, which makes breaking the judge a
winning strategy.

## Two failure modes worth knowing about

Both bugs the scorer has had took the same shape: **a correct drawing expressed in
a less common way was read as a wrong drawing.** Wheel detection anchored on the
largest round shape, so a rider drawn bigger than the wheels became the anchor
and the real wheels were discarded. And the shape walker descended into `<defs>`
while ignoring `<use>`, so a model that wrote its wheel once and placed it twice
had the template counted at the origin and the instances skipped.

Neither was visible in the aggregate numbers. Both were obvious the moment the
low-scoring drawings were rendered and looked at, which is what
`--save-svgs` on `examples/pelican_svg_eval.py` is for.
`fixtures/wheels_as_paths.svg` and `fixtures/wheels_via_use.svg` are the
regression cases.

## Known limitations

- **The judge model matters more than anything else.** On one identical
  drawing, `Qwen2.5-VL-72B` scored the semantic component 0.639 and
  `Qwen3-VL-30B` scored it 0.875. That 0.24 spread dwarfs every other source
  of noise, so a published number is meaningless without naming the judge.
  Within a fixed judge the scoring is stable: 8 repeat runs of the same
  drawing through the deployed Space returned sd 0.0000 and a
  character-identical caption. Across longer spans one checklist item has been
  seen to flip, worth about 0.04 of final reward.
- **Structure cannot identify a species.** By design, but it means a model that
  draws an excellent generic blob on an excellent bicycle collects the full
  structural weight.

## Rendering

`resvg` via `resvg-py`, pinned. It ships self-contained wheels for manylinux,
musllinux, macOS and Windows, so the container needs no system libraries. It is
deterministic across runs, and it refuses to resolve external entities or fetch
remote `href`s, which matters when the input is untrusted model output.

## Development

```bash
PYTHONPATH=src:envs uv run pytest tests/envs/test_pelican_svg_env.py -v
```

The tests need no network: the judge is exercised through a stub client, which
is also the only way to cover the failure paths that matter.

Test fixtures live in `fixtures/`, a graded corpus from a complete scene down to
the two cheats. Thresholds in the gate and the structural layer were calibrated
against it and then re-checked against real model output, which caught two
cases the hand-written corpus could not: an edge-density check that rejected
legitimate filled-shape drawings, and single-wheeled vehicles being scored
against bicycle geometry.
