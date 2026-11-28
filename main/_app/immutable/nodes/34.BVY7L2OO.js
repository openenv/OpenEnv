import"../chunks/DsnmJJEf.js";import{i as J,h as R,C as W,H as t,a,E as A,s as B}from"../chunks/Dz7d4_v9.js";import{p as C,o as L,s as e,f as z,a as S,b as P,c as V,n as F}from"../chunks/BSuxAqoA.js";const Q='{"title":"Pelican SVG Environment","local":"pelican-svg-environment","sections":[{"title":"What this is, and what it is not","local":"what-this-is-and-what-it-is-not","sections":[],"depth":2},{"title":"The task catalogue","local":"the-task-catalogue","sections":[{"title":"Animals","local":"animals","sections":[],"depth":3},{"title":"Vehicles","local":"vehicles","sections":[],"depth":3},{"title":"How much of the grid has actually been run","local":"how-much-of-the-grid-has-actually-been-run","sections":[],"depth":3}],"depth":2},{"title":"Scoring","local":"scoring","sections":[{"title":"1. Gate, free, pass or fail","local":"1-gate-free-pass-or-fail","sections":[],"depth":3},{"title":"2. Structure, free, weight 0.35","local":"2-structure-free-weight-035","sections":[],"depth":3},{"title":"3. Vision judge, one call pair per sample, weight 0.65","local":"3-vision-judge-one-call-pair-per-sample-weight-065","sections":[],"depth":3}],"depth":2},{"title":"What training against it revealed","local":"what-training-against-it-revealed","sections":[],"depth":2},{"title":"Usage","local":"usage","sections":[{"title":"Configuration","local":"configuration","sections":[],"depth":3},{"title":"Judging with a local model","local":"judging-with-a-local-model","sections":[],"depth":3}],"depth":2},{"title":"Two failure modes worth knowing about","local":"two-failure-modes-worth-knowing-about","sections":[],"depth":2},{"title":"Known limitations","local":"known-limitations","sections":[],"depth":2},{"title":"Rendering","local":"rendering","sections":[],"depth":2},{"title":"Development","local":"development","sections":[],"depth":2}],"depth":1}';var H=V('<meta name="hf:doc:metadata"/>'),O=V(`<p></p> <!> <!> <p>Blind vector drawing, scored. The model is asked for an SVG of an animal riding
a vehicle and never sees the result, so it has to hold the spatial arrangement
in its head and emit coordinates for it. This is the check Simon Willison
popularised as “generate an SVG of a pelican riding a bicycle”, turned into
something you can run repeatedly, at scale, and train against.</p> <p><img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/openenv/pelican_svg.png" alt="Pelican SVG web playground"/></p> <p><em>The <code>/web</code> playground after submitting an SVG: the rendered drawing and its reward breakdown.</em></p> <!> <p><strong>Not a model ranking.</strong> Simon Willison, whose prompt this is, is explicit that
“the correlation between pelican performance and actual model quality has been
mostly severed now”, and ends with “don’t go using pelicans to compare models”
(<a href="https://simonwillison.net/2026/Jul/16/kimi-k3/" rel="nofollow">Kimi K3, and what we can still learn from the pelican
benchmark</a>). Our own numbers
agree with him: on the canonical task, across 139 scored samples from 7 frontier models,
136 scored a perfect 1.000 on the structural layer and 47 scored a perfect 1.000 overall.
It is saturated.</p> <p><strong>Not a contamination study either.</strong> <a href="https://dylancastillo.co/posts/pelicanmaxxing.html" rel="nofollow">Dylan
Castillo</a> already ran that,
across an 8 by 6 grid and 1008 SVGs from 7 frontier models, and found little
evidence that labs optimise for the pelican. The extra animals and vehicles here
are ours and serve a smaller purpose: checking that the scorer measures drawing
ability rather than one memorised picture. Only <code>pelican_bicycle</code> is Simon’s, and
it is the default.</p> <p><strong>What it is</strong> is a reproducible, executable target. The same code that scores a
frontier model over an API scores a 0.6B model mid-training, the reward is dense
enough to run GRPO against, and the whole thing deploys as a Space. That is the
part a prompt cannot do:</p> <ul><li><a href="https://github.com/openenv/OpenEnv/blob/main/examples/pelican_svg_eval.py" rel="nofollow"><code>examples/pelican_svg_eval.py</code></a> scores any set of models through HF Inference Providers.</li> <li><a href="https://github.com/openenv/OpenEnv/blob/main/examples/pelican_svg_grpo.py" rel="nofollow"><code>examples/pelican_svg_grpo.py</code></a> trains against it with TRL, runnable on HF Jobs.</li></ul> <p>The interesting question stops being “which model draws the best pelican” and
becomes “can a small model be taught to, and does it learn to draw or to game
the scorer”.</p> <!> <p>6 animals by 5 vehicles, so 30 tasks. All three
scoring layers adapt to whichever one is served, so nothing about the pipeline is special
to the pelican:</p> <ul><li><strong>The judge’s checklist is generated per task</strong> from the features below. A pelican is
asked about a throat pouch, a capybara about a blunt snout, an octopus about curling
tentacles.</li> <li><strong>The geometry layer reads <code>wheels</code></strong> from the vehicle, and drops the wheel-pair checks
entirely for a single-wheeled vehicle rather than failing them.</li> <li><strong>The anti-cheat terms are derived from the task</strong>, so <code>axolotl_scooter</code> forbids <code>axolotl, salamander, amphibian, scooter, kick scooter</code> and not the pelican’s words.</li></ul> <!> <table><thead><tr><th>subject</th><th>features the judge asks about</th><th>counts as partial recognition</th></tr></thead><tbody><tr><td><code>pelican</code></td><td>a long beak, a throat pouch under the beak</td><td>bird, seabird, waterbird</td></tr><tr><td><code>flamingo</code></td><td>a long thin neck, long thin legs</td><td>bird, waterbird</td></tr><tr><td><code>capybara</code></td><td>a blunt rectangular snout, small rounded ears</td><td>rodent, animal</td></tr><tr><td><code>axolotl</code></td><td>feathery external gills on the head, a wide flat smiling mouth</td><td>salamander, amphibian, animal</td></tr><tr><td><code>octopus</code></td><td>a large bulbous head, multiple curling tentacles</td><td>cephalopod, animal</td></tr><tr><td><code>hedgehog</code></td><td>a coat of spines, a small pointed snout</td><td>animal</td></tr></tbody></table> <!> <table><thead><tr><th>vehicle</th><th>wheels in side view</th><th>features the judge asks about</th></tr></thead><tbody><tr><td><code>bicycle</code></td><td>2</td><td>two wheels of similar size, a frame joining the wheels, handlebars</td></tr><tr><td><code>unicycle</code></td><td>1</td><td>a single wheel, a seat post rising from the wheel</td></tr><tr><td><code>tandem bicycle</code></td><td>2</td><td>two wheels, an extended frame with two saddles</td></tr><tr><td><code>scooter</code></td><td>2</td><td>two small wheels, a deck with an upright steering column</td></tr><tr><td><code>skateboard</code></td><td>2</td><td>a flat deck, wheels under the deck</td></tr></tbody></table> <!> <p><strong>All 30 combinations have been scored end to end by a live model. None crashed and none
was rejected by the gate.</strong> The counts below are how much traffic each cell has actually
seen, which is a different question from whether it works:</p> <table><thead><tr><th>animal</th><th><code>bicycle</code></th><th><code>unicycle</code></th><th><code>tandem bicycle</code></th><th><code>scooter</code></th><th><code>skateboard</code></th></tr></thead><tbody><tr><td><code>pelican</code></td><td><strong>224</strong></td><td>1</td><td>1</td><td>1</td><td>1</td></tr><tr><td><code>flamingo</code></td><td>1</td><td>1</td><td>1</td><td><strong>16</strong></td><td>1</td></tr><tr><td><code>capybara</code></td><td>1</td><td><strong>16</strong></td><td>1</td><td>1</td><td>1</td></tr><tr><td><code>axolotl</code></td><td>1</td><td>1</td><td><strong>16</strong></td><td>1</td><td>1</td></tr><tr><td><code>octopus</code></td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td></tr><tr><td><code>hedgehog</code></td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td></tr></tbody></table> <p>Read a <code>1</code> as “the plumbing works”, not as a result. It says the task builds, the prompt
renders, the gate admits the reply, the geometry resolves against that vehicle’s wheel
count and the judge returns a checklist. It says nothing about how well the scoring
behaves on that combination, and it cannot: one sample has no variance to report.</p> <p>Across the 30-task sweep, reward ranged 0.362 to 1.000 with a median of 0.838, structure
averaged 0.905 and the judged semantic score 0.785. The weakest cell was <code>hedgehog_tandem-bicycle</code> at 0.362, where the judge would not accept the animal. Three of
the 30 were corrected upward by the <code>&lt;use&gt;</code> fix described below, <code>capybara_unicycle</code> most
of all, from 0.333 to 1.000 on structure.</p> <p><code>pelican_bicycle</code> is Simon Willison’s original prompt and the default. It is also the only
task the GRPO example trains on: <code>examples/pelican_svg_grpo.py</code> pins it deliberately,
since that script is a small demonstration that the environment is trainable rather than
an attempt to teach a model the whole grid.</p> <!> <p>Three layers, ordered by cost. Each earns the right to run the next, so a run
against a model producing garbage stays cheap.</p> <!> <p>Rejects anything that is not an honest attempt to draw. A rejected submission
scores zero and never reaches the judge.</p> <table><thead><tr><th>Code</th><th>Meaning</th></tr></thead><tbody><tr><td><code>no_svg_in_response</code> / <code>truncated_svg</code></td><td>No SVG, or one cut off mid-generation. Kept separate because “the model refused” and “the harness cut it off” are different facts.</td></tr><tr><td><code>unparseable</code> / <code>render_failed</code></td><td>Not well-formed, declares a DTD, or will not rasterise.</td></tr><tr><td><code>embedded_raster</code></td><td>An <code>&lt;image&gt;</code> element or any <code>data:</code> URI.</td></tr><tr><td><code>external_reference</code></td><td>An <code>href</code> pointing off-document.</td></tr><tr><td><code>forbidden_element</code></td><td><code>&lt;script&gt;</code>, <code>&lt;foreignObject&gt;</code> and friends.</td></tr><tr><td><code>text_label</code> / <code>text_heavy</code></td><td>Writing the answer instead of drawing it.</td></tr><tr><td><code>blank_canvas</code> / <code>content_off_canvas</code></td><td>Nothing visible, or geometry placed outside the viewBox.</td></tr><tr><td><code>too_few_elements</code> / <code>too_many_elements</code></td><td>Not a drawing, or a denial-of-service payload.</td></tr></tbody></table> <p>The two cheats worth knowing about:</p> <ul><li><strong>Embedded raster.</strong> A base64 PNG inside an <code>&lt;image&gt;</code> renders as a completely
convincing picture. A vision judge scores it highly. It is only visible in the
source, which is why source inspection runs before anything looks at pixels.</li> <li><strong>Text label.</strong> <code>&lt;text&gt;a pelican riding a bicycle&lt;/text&gt;</code> also renders, and
also fools a careless judge.</li></ul> <!> <p>Geometry over the parsed document with transforms applied and coordinates
normalised by the viewBox, so nothing depends on the units the model picked.
Seven checks: wheel count, wheels similar in size, level, sensibly far apart,
something spanning between them, a rider present, and the rider proportionate
to the wheelbase.</p> <p>Wheels are found as a row of similar-sized round shapes at similar height, with
the lowest row preferred. Not by picking the biggest circle: see the correction
below for what that cost.</p> <p>Deliberately shape-level. It cannot tell a pelican from a capybara and does not
try. What it can do is answer questions no judge should be trusted with because
the answer is arithmetic.</p> <!> <ul><li><strong>A blind caption.</strong> The judge is shown the picture and asked what it is, with
no mention of the task. If it says “a pelican on a bicycle” unprompted, that
is the strongest available evidence, and a leading question cannot inflate it.
In testing this mattered: asked “is this a pelican riding a bicycle?” the
judge said yes; asked to describe the same image cold, it said “a cartoon
duck”.</li> <li><strong>A binary feature checklist.</strong> Individual yes-or-no questions rather than a
0-10 rating, because individual features are reproducible and a global rating
drifts.</li></ul> <p>Failing the riding posture costs 75% of the semantic score rather than all of
it. Posture is a spatial relation and the least reliable item on the checklist,
so hanging the whole component on that one boolean gives the judge’s worst call
the loudest vote.</p> <!> <p>Four GRPO runs through TRL on HF Jobs, 80 steps each. Three with the judge
disabled, so the reward is deterministic arithmetic, and one with <code>--judge-in-reward</code>. The
training reward climbed in every one of them, which turns out to say very little.</p> <p>What says more is re-scoring each checkpoint on 24 fresh samples <em>with</em> the judge on, and
comparing against the untrained model. Every run measures that first, so there are five
independent measurements of the starting point:</p> <table><thead><tr><th></th><th>untrained, 5 probes</th><th>1.7B, judge off</th><th>1.7B, judge on</th></tr></thead><tbody><tr><td>structural score</td><td>0.374 ± 0.064</td><td><strong>0.619</strong> and 0.643</td><td>0.321</td></tr><tr><td>judged semantic score</td><td>0.0049 ± 0.0042</td><td>0.0075 and 0.0139</td><td>0.0023</td></tr><tr><td>submissions rejected by the gate</td><td>3 to 13 of 24</td><td>1 and 2 of 24</td><td>11 of 24</td></tr></tbody></table> <p>The judge-off runs finish 3.8 standard deviations above the untrained structural mean. The
judge-on run finishes <em>inside</em> the untrained band, 0.8 below the mean, and its semantic
score went down. Turning the judge on drops the weight on structure from 1.0 to 0.35, so
it dilutes the only term that teaches anything and replaces it with one that teaches
nothing: across all 48 probe drawings, before and after, the judge never once answered yes
to the riding question.</p> <p>It learned two real things: to emit well-formed SVG, and to build bicycle-like geometry,
both on samples it never trained on. It did not learn to draw a bird. <strong>Twice,
independently</strong>, which is why this is stated as a finding rather than an anecdote.</p> <p>Looking at the probe drawings rather than the table sharpens it. Between before and after,
samples with no detectable structure at all fell from 9 of 24 to 2 of 24 and the median
structural score rose from 0.43 to 0.71, so something real was learned. But the single
best semantic score in the probe went <em>down</em>, 0.045 to 0.028, and the word “bird” appears
in one blind caption before training and none after. The output got more measurable and
slightly less bird-like. Weights and all 48 probe drawings are public for both the
judge-off run, <a href="https://huggingface.co/sergiopaniego/pelican-svg-grpo-Qwen3-1.7B" rel="nofollow"><code>pelican-svg-grpo-Qwen3-1.7B</code></a>,
and the judge-on run, <a href="https://huggingface.co/sergiopaniego/pelican-svg-grpo-Qwen3-1.7B-judged" rel="nofollow"><code>pelican-svg-grpo-Qwen3-1.7B-judged</code></a>,
with both training curves at <a href="https://huggingface.co/spaces/sergiopaniego/pelican-svg-grpo-curves" rel="nofollow"><code>pelican-svg-grpo-curves</code></a>.</p> <p>Read that carefully, because the obvious label is the wrong one. <strong>The judge was switched
off during training, so the semantic score was never part of the reward.</strong> The policy
optimised exactly what it was paid for. This is not a model subverting its objective, it
is a demonstration that <strong>the deterministic layer alone is an insufficient proxy for the
task</strong>: two round shapes of equal size at equal height, a bar between them and any
blob above the axle line scores 1.000 on all seven checks. Four primitives and no
animal, and it is the exact figure the trained policy converged on.</p> <p>Whether a model <em>would</em> game a reward the judge is part of was the obvious follow-up, and
the answer is that at this size the question does not arise: the judged term is almost
always zero, so there is nothing to game. In 24 fresh samples the untrained 1.7B scored
non-zero on the semantic component <strong>twice</strong>, and GRPO takes its gradient from the
variance inside a group of completions, so a term that is zero for every member of a group
contributes nothing however heavily it is weighted. The same judge scores non-zero on 138
of 139 frontier samples. Before spending GPU time on a judged reward, measure how often it
is non-zero on the base model you are about to train.</p> <p><code>examples/pelican_svg_grpo.py</code> ships the probe that produced all of this: it scores fresh
samples with the judge before and after, and flags a run where the structural gain
outpaces the judged gain. That ratio was 23 to 1 on the first 1.7B run and 45 to 1 on the
second.</p> <!> <!> <p>Pin the task with <code>task_id</code> for any benchmark run. Without it every reset draws
a fresh task and two models are never asked the same question.</p> <p>Run the server locally:</p> <!> <!> <table><thead><tr><th>Argument</th><th>Default</th><th>Effect</th></tr></thead><tbody><tr><td><code>subject</code>, <code>vehicle</code></td><td>sampled</td><td>Pin the task for every episode.</td></tr><tr><td><code>held_out_only</code></td><td><code>False</code></td><td>Exclude the canonical pelican-and-bicycle pair.</td></tr><tr><td><code>enable_judge</code></td><td><code>True</code></td><td><code>False</code> scores on the deterministic layers alone. Note that passing <code>judge=None</code> asks for the default judge; this is the flag that turns it off.</td></tr><tr><td><code>return_image</code></td><td><code>False</code></td><td>Include the rendered PNG in the observation.</td></tr></tbody></table> <p>Environment variables: <code>HF_TOKEN</code> for the judge, <code>PELICAN_SVG_JUDGE_MODEL</code> to
change it from <code>Qwen/Qwen2.5-VL-72B-Instruct</code>, <code>PELICAN_SVG_DISABLE_JUDGE=1</code> to
force offline scoring, and <code>PELICAN_SVG_JUDGE_BASE_URL</code> (with <code>PELICAN_SVG_JUDGE_API_KEY</code> where the endpoint wants one) to judge with a
self-hosted model instead. <code>MAX_CONCURRENT_ENVS</code> sets the maximum number of
concurrent WebSocket sessions (default <code>8</code>).</p> <!> <p>The judge speaks the OpenAI-compatible chat API, so any server that does will
do. Serving a vision model on one GPU keeps the judged component in the reward
without paying per rollout:</p> <!> <p><code>PELICAN_SVG_JUDGE_MODEL</code> is required here, since the hosted default names a
72B repository the local server is unlikely to be serving, and a judge whose
every call fails is worse than no judge at all; the server refuses to start
without it. No Hugging Face token is needed, and an ambient one is never sent
to the endpoint.</p> <p>Turn thinking off if the model has it. The caption call is capped at 120
tokens and the checklist at 400, and a reasoning model spends that budget
thinking and returns an empty caption, which scores the semantic component
zero on every sample. With vLLM that is <code>--default-chat-template-kwargs '&#123;"enable_thinking": false&#125;'</code>. Instruct-only
vision models need nothing.</p> <p><code>huggingface_hub</code> sends the checklist schema to a custom endpoint in TGI’s
format, <code>&#123;"type": "json_object", "value": &lt;schema&gt;&#125;</code>. TGI enforces the schema;
vLLM reads it as plain JSON mode, so the reply is asked to be JSON but not held
to the keys. The prompt lists them and a missing key counts as <code>false</code>, so a
drifting reply scores lower rather than failing.</p> <p>With no judge configured, structure carries the full weight. A judge that is
configured but <em>fails</em> is treated differently: the semantic component stays at
zero and <code>judged</code> is set false on the observation, so a harness can drop or
retry the sample. Renormalising there would mean a submission scores higher
precisely when nobody could look at it, which makes breaking the judge a
winning strategy.</p> <!> <p>Both bugs the scorer has had took the same shape: <strong>a correct drawing expressed in
a less common way was read as a wrong drawing.</strong> Wheel detection anchored on the
largest round shape, so a rider drawn bigger than the wheels became the anchor
and the real wheels were discarded. And the shape walker descended into <code>&lt;defs&gt;</code> while ignoring <code>&lt;use&gt;</code>, so a model that wrote its wheel once and placed it twice
had the template counted at the origin and the instances skipped.</p> <p>Neither was visible in the aggregate numbers. Both were obvious the moment the
low-scoring drawings were rendered and looked at, which is what <code>--save-svgs</code> on <code>examples/pelican_svg_eval.py</code> is for. <code>fixtures/wheels_as_paths.svg</code> and <code>fixtures/wheels_via_use.svg</code> are the
regression cases.</p> <!> <ul><li><strong>The judge model matters more than anything else.</strong> On one identical
drawing, <code>Qwen2.5-VL-72B</code> scored the semantic component 0.639 and <code>Qwen3-VL-30B</code> scored it 0.875. That 0.24 spread dwarfs every other source
of noise, so a published number is meaningless without naming the judge.
Within a fixed judge the scoring is stable: 8 repeat runs of the same
drawing through the deployed Space returned sd 0.0000 and a
character-identical caption. Across longer spans one checklist item has been
seen to flip, worth about 0.04 of final reward.</li> <li><strong>Structure cannot identify a species.</strong> By design, but it means a model that
draws an excellent generic blob on an excellent bicycle collects the full
structural weight.</li></ul> <!> <p><code>resvg</code> via <code>resvg-py</code>, pinned. It ships self-contained wheels for manylinux,
musllinux, macOS and Windows, so the container needs no system libraries. It is
deterministic across runs, and it refuses to resolve external entities or fetch
remote <code>href</code>s, which matters when the input is untrusted model output.</p> <!> <!> <p>The tests need no network: the judge is exercised through a stub client, which
is also the only way to cover the failure paths that matter.</p> <p>Test fixtures live in <code>fixtures/</code>, a graded corpus from a complete scene down to
the two cheats. Thresholds in the gate and the structural layer were calibrated
against it and then re-checked against real model output, which caught two
cases the hand-written corpus could not: an edge-density check that rejected
legitimate filled-shape drawings, and single-wheeled vehicles being scored
against bicycle geometry.</p> <!> <p></p>`,1);function X(E,N){C(N,!1),L(()=>{new URLSearchParams(window.location.search).get("fw")}),J();var o=O();R("ttjqyb",x=>{var G=H();B(G,"content",Q),S(x,G)});var n=e(z(o),2);W(n,{containerStyle:"float: inline-end; margin-inline-start: 10px; display: inline-flex; position: relative; z-index: 10;"});var s=e(n,2);t(s,{title:"Pelican SVG Environment",local:"pelican-svg-environment",headingTag:"h1"});var d=e(s,8);t(d,{title:"What this is, and what it is not",local:"what-this-is-and-what-it-is-not",headingTag:"h2"});var i=e(d,12);t(i,{title:"The task catalogue",local:"the-task-catalogue",headingTag:"h2"});var r=e(i,6);t(r,{title:"Animals",local:"animals",headingTag:"h3"});var l=e(r,4);t(l,{title:"Vehicles",local:"vehicles",headingTag:"h3"});var c=e(l,4);t(c,{title:"How much of the grid has actually been run",local:"how-much-of-the-grid-has-actually-been-run",headingTag:"h3"});var h=e(c,12);t(h,{title:"Scoring",local:"scoring",headingTag:"h2"});var g=e(h,4);t(g,{title:"1. Gate, free, pass or fail",local:"1-gate-free-pass-or-fail",headingTag:"h3"});var p=e(g,10);t(p,{title:"2. Structure, free, weight 0.35",local:"2-structure-free-weight-035",headingTag:"h3"});var u=e(p,8);t(u,{title:"3. Vision judge, one call pair per sample, weight 0.65",local:"3-vision-judge-one-call-pair-per-sample-weight-065",headingTag:"h3"});var m=e(u,6);t(m,{title:"What training against it revealed",local:"what-training-against-it-revealed",headingTag:"h2"});var w=e(m,20);t(w,{title:"Usage",local:"usage",headingTag:"h2"});var b=e(w,2);a(b,{code:"ZnJvbSUyMGVudnMucGVsaWNhbl9zdmdfZW52JTIwaW1wb3J0JTIwUGVsaWNhblN2Z0FjdGlvbiUyQyUyMFBlbGljYW5TdmdFbnYlMEElMEF3aXRoJTIwUGVsaWNhblN2Z0VudihiYXNlX3VybCUzRCUyMmh0dHAlM0ElMkYlMkZsb2NhbGhvc3QlM0E4MDAwJTIyKSUyMGFzJTIwZW52JTNBJTBBJTIwJTIwJTIwJTIwb2JzZXJ2YXRpb24lMjAlM0QlMjBlbnYucmVzZXQodGFza19pZCUzRCUyMmNhcHliYXJhX3VuaWN5Y2xlJTIyKS5vYnNlcnZhdGlvbiUwQSUyMCUyMCUyMCUyMHJlc3VsdCUyMCUzRCUyMGVudi5zdGVwKFBlbGljYW5TdmdBY3Rpb24ocmVzcG9uc2UlM0RteV9tb2RlbChvYnNlcnZhdGlvbi5wcm9tcHQpKSklMEElMjAlMjAlMjAlMjBwcmludChyZXN1bHQucmV3YXJkJTJDJTIwcmVzdWx0Lm9ic2VydmF0aW9uLmZlZWRiYWNrKQ==",highlighted:`<span class="hljs-keyword">from</span> envs.pelican_svg_env <span class="hljs-keyword">import</span> PelicanSvgAction, PelicanSvgEnv

<span class="hljs-keyword">with</span> PelicanSvgEnv(base_url=<span class="hljs-string">&quot;http://localhost:8000&quot;</span>) <span class="hljs-keyword">as</span> env:
    observation = env.reset(task_id=<span class="hljs-string">&quot;capybara_unicycle&quot;</span>).observation
    result = env.step(PelicanSvgAction(response=my_model(observation.prompt)))
    <span class="hljs-built_in">print</span>(result.reward, result.observation.feedback)`,lang:"python",wrap:!1});var f=e(b,6);a(f,{code:"UFlUSE9OUEFUSCUzRHNyYyUzQWVudnMlMjB1diUyMHJ1biUyMHV2aWNvcm4lMjBwZWxpY2FuX3N2Z19lbnYuc2VydmVyLmFwcCUzQWFwcCUyMC0tcG9ydCUyMDgwMDA=",highlighted:"PYTHONPATH=src:envs uv run uvicorn pelican_svg_env.server.app:app --port 8000",lang:"bash",wrap:!1});var v=e(f,2);t(v,{title:"Configuration",local:"configuration",headingTag:"h3"});var y=e(v,6);t(y,{title:"Judging with a local model",local:"judging-with-a-local-model",headingTag:"h3"});var _=e(y,4);a(_,{code:"dmxsbSUyMHNlcnZlJTIwUXdlbiUyRlF3ZW4yLjUtVkwtN0ItSW5zdHJ1Y3QlMjAtLXBvcnQlMjA4MDAwJTIwLS1saW1pdC1tbS1wZXItcHJvbXB0JTIwJyU3QiUyMmltYWdlJTIyJTNBMSU3RCclMEElMEFleHBvcnQlMjBQRUxJQ0FOX1NWR19KVURHRV9CQVNFX1VSTCUzRGh0dHAlM0ElMkYlMkZsb2NhbGhvc3QlM0E4MDAwJTJGdjElMEFleHBvcnQlMjBQRUxJQ0FOX1NWR19KVURHRV9NT0RFTCUzRFF3ZW4lMkZRd2VuMi41LVZMLTdCLUluc3RydWN0",highlighted:`vllm serve Qwen/Qwen2.5-VL-7B-Instruct --port 8000 --limit-mm-per-prompt <span class="hljs-string">&#x27;{&quot;image&quot;:1}&#x27;</span>

<span class="hljs-built_in">export</span> PELICAN_SVG_JUDGE_BASE_URL=http://localhost:8000/v1
<span class="hljs-built_in">export</span> PELICAN_SVG_JUDGE_MODEL=Qwen/Qwen2.5-VL-7B-Instruct`,lang:"bash",wrap:!1});var k=e(_,10);t(k,{title:"Two failure modes worth knowing about",local:"two-failure-modes-worth-knowing-about",headingTag:"h2"});var T=e(k,6);t(T,{title:"Known limitations",local:"known-limitations",headingTag:"h2"});var j=e(T,4);t(j,{title:"Rendering",local:"rendering",headingTag:"h2"});var M=e(j,4);t(M,{title:"Development",local:"development",headingTag:"h2"});var U=e(M,2);a(U,{code:"UFlUSE9OUEFUSCUzRHNyYyUzQWVudnMlMjB1diUyMHJ1biUyMHB5dGVzdCUyMHRlc3RzJTJGZW52cyUyRnRlc3RfcGVsaWNhbl9zdmdfZW52LnB5JTIwLXY=",highlighted:"PYTHONPATH=src:envs uv run pytest tests/envs/test_pelican_svg_env.py -v",lang:"bash",wrap:!1});var I=e(U,6);A(I,{source:"https://github.com/openenv/OpenEnv/blob/main/docs/source/environments/pelican_svg.md"}),F(2),S(E,o),P()}export{X as component};
