import"../chunks/DsnmJJEf.js";import{i as k,h as E,C as R,H as t,a as s,E as B,s as q}from"../chunks/Dz7d4_v9.js";import{p as S,o as V,s as e,f as N,a as C,b as Q,c as v,n as F}from"../chunks/BSuxAqoA.js";const Z='{"title":"Post-training Environment (experimental)","local":"post-training-environment-experimental","sections":[{"title":"Tools","local":"tools","sections":[],"depth":2},{"title":"How a job runs","local":"how-a-job-runs","sections":[{"title":"No network in agent jobs","local":"no-network-in-agent-jobs","sections":[],"depth":3},{"title":"Evaluations","local":"evaluations","sections":[],"depth":3}],"depth":2},{"title":"Budget and spend","local":"budget-and-spend","sections":[],"depth":2},{"title":"Episode end","local":"episode-end","sections":[],"depth":2},{"title":"Using it with a trainer","local":"using-it-with-a-trainer","sections":[],"depth":2},{"title":"Tasks","local":"tasks","sections":[],"depth":2},{"title":"Running the server","local":"running-the-server","sections":[],"depth":2},{"title":"Client","local":"client","sections":[],"depth":2},{"title":"Known gaps","local":"known-gaps","sections":[],"depth":2},{"title":"Tests","local":"tests","sections":[],"depth":2}],"depth":1}';var z=v('<meta name="hf:doc:metadata"/>'),W=v(`<p></p> <!> <!> <p><code>posttrain_env</code> is an MCP environment where the agent post-trains a model on <a href="https://huggingface.co/docs/huggingface_hub/guides/jobs" rel="nofollow">HF Jobs</a>. It is the
experimental, eval-first prototype for <a href="https://github.com/openenv/OpenEnv/pull/1508" rel="nofollow">RFC 014: Environment Services
(PR #1508)</a>; discussion in <a href="https://github.com/openenv/OpenEnv/issues/1382" rel="nofollow">#1382</a>.</p> <p>The agent never holds a credential. It submits training scripts through MCP
tools, and the server’s broker starts them on HF Jobs with its own token. Each
job’s worst-case cost is reserved against a per-rollout budget before it starts
and settled at the real cost when it ends. <code>dev_evaluate</code> scores a checkpoint
on a public dev split. <code>submit</code> scores it on a hidden split in a grader-only
job, and that score is the reward.</p> <!> <table><thead><tr><th>Tool</th><th>What it does</th></tr></thead><tbody><tr><td><code>budget()</code></td><td>Remaining budget, allowed hardware with its price, jobs and dev evaluations left</td></tr><tr><td><code>jobs_submit(script, flavor, timeout_minutes, dependencies=None, datasets=None, idempotency_key=None)</code></td><td>Run a Python script as an HF Job, like <code>hf jobs uv run</code>, without network access. Reserves <code>price/min × timeout</code> first</td></tr><tr><td><code>jobs_status(job_id)</code></td><td>State, cost once settled, artifact files of a succeeded job, result of a finished evaluation</td></tr><tr><td><code>jobs_wait(job_id, timeout_s=60)</code></td><td>Wait up to 300 s for a job to finish</td></tr><tr><td><code>jobs_logs(job_id, tail=100)</code></td><td>Last lines of one of your own jobs’ logs</td></tr><tr><td><code>jobs_cancel(job_id)</code></td><td>Cancel a job; it settles for the minutes it ran</td></tr><tr><td><code>dev_evaluate(job_id)</code></td><td>Score a succeeded job’s artifact on the dev split, as a new job charged to your budget</td></tr><tr><td><code>submit(job_id)</code></td><td>Final answer: cancels your other jobs, scores the artifact on the hidden split, ends the episode</td></tr></tbody></table> <p>Errors come back as data, with a type the agent can act on: <code>budget_exhausted</code>, <code>flavor_not_allowed</code>, <code>timeout_not_allowed</code>, <code>invalid_script</code>, <code>invalid_dependency</code>, <code>dataset_not_allowed</code>, <code>limit_reached</code>, <code>not_ready</code>, <code>unknown_job</code>, <code>not_allowed</code>, <code>episode_over</code> or <code>hf_error</code>.</p> <!> <p>Agent scripts run with <code>HfApi.run_uv_job</code>, following its usual behaviour: the
script is uploaded to the namespace’s private <code>jobs-artifacts</code> bucket, and the
job’s folder there is mounted read-write at <code>/data</code>. Anything the job writes
under <code>/data/outputs/</code> is its artifact:</p> <!> <p>Jobs are started with an explicit flavor, namespace, timeout and labels, and
with no secrets. The broker never passes its own token into a job: a job holding
a “Start and manage Jobs” token could start further jobs that skip the budget.
It also never shells out to <code>hf jobs uv run</code>, whose script header can pull the
caller’s <code>HF_TOKEN</code> into the job.</p> <!> <p>A job that can reach the internet can download a model already trained for the
task, or the evaluation’s own test data. So agent jobs get their inputs as
read-only mounts, which the broker checks and the episode records, and run
without network access:</p> <ul><li>The task’s base model is mounted at <code>/base_model</code>, at a pinned revision.</li> <li>Each dataset in <code>datasets</code> is mounted at <code>/datasets/&lt;repo_id&gt;</code>, at its current
revision. It must be public and ungated, and not on the task’s denylist. The
broker looks it up without its token, so the token can’t open private repos.</li> <li>Packages from <code>dependencies</code> and the script’s PEP 723 header are installed
into a fresh venv before the script runs, from PyPI wheels only. Installing a
wheel runs none of its code, so no package code runs while the job still has
network access. URLs and git dependencies are refused. Other settings in the
script’s header are ignored.</li></ul> <p>Mounts and wheels narrow what a job can fetch, but a public dataset or a PyPI
wheel can still hold weights or test data. Tasks whose test data is public also
need a contamination check after the run.</p> <p>HF Jobs has no egress control, and its containers can’t create network
namespaces or firewall rules. So the broker sends <a href="https://github.com/openenv/OpenEnv/blob/main/envs/posttrain_env/server/job_launcher.py" rel="nofollow"><code>server/job_launcher.py</code></a> as the job’s script, with the
agent’s script and packages embedded. Once it has installed the packages, the
launcher installs a seccomp filter that refuses every socket except <code>AF_UNIX</code>,
and io_uring and cross-process memory access; systemd’s <code>RestrictAddressFamilies=</code> works the same way. It then closes inherited file
descriptors, switches to a uid no other process in the job uses, and runs the
agent’s script with <code>HF_HUB_OFFLINE=1</code>. The filter can’t be removed and covers every child
process. If any step fails, the script doesn’t run. Training on a single GPU,
DataLoader workers and writing <code>/data/outputs/</code> work under the filter; <code>torch.distributed</code> over localhost TCP does not.</p> <!> <p>Evaluations run the task’s evaluator script as their own job with that folder
mounted read-only at <code>/model</code>, and the task’s base model at <code>/base_model</code>. The
evaluator prints one <code>POSTTRAIN_SCORE &#123;...&#125;</code> line. A missing line, or a score
outside [0, 1], scores 0 as <code>invalid</code>. Evaluators take only weights from the
artifact, loaded as safetensors. The tokenizer, chat template, config and
generation settings come from the base model mount, because a chat template is
a program that sees the question and could write the answer into the prompt. Each evaluator pins its
dependencies in its PEP 723 header, with an <code>exclude-newer</code> date for the rest.</p> <!> <p>Money is tracked in micro-USD from the prices <code>list_jobs_hardware</code> reports.
A job settles at its price per minute times its running minutes, rounded up to
whole minutes. Grader evaluations are not charged to the agent.</p> <p>Every observation carries the episode’s spend, for accounting and for budgets,
concurrency and rollout allocation. It is not a sampling weight (RFC 013), and
an episode that ends unscored still reports it:</p> <!> <p><code>jobs</code> counts agent jobs and <code>calls</code> counts dev evaluations. <code>final</code> is false
while a job is still running or a cancellation was never confirmed.</p> <!> <p>Through <code>step(CallToolAction(...))</code>, an episode ends when the agent calls <code>submit</code>, after <code>max_steps</code> tool calls (default 100) or after the task’s time limit.
The step and time limits give reward 0.</p> <p>Direct <code>/mcp</code> calls, including <code>env.call_tool</code>, bypass the step and time limits
and do not return the final step observation with reward and <code>service_spend</code>.</p> <p>If the score job itself fails, times out or is cancelled, the episode ends
unscored: <code>reward</code> is <code>None</code> and <code>episode_end.reason</code> is <code>score_failed</code>, since
the fault is the grader’s. Exclude such a rollout from reward-based training
and reward estimates, rather than treating it as zero, while retaining its
episode and spend for accounting. Under the current RFC 013 (#1256) proposal,
pass its <code>None</code> reward and spend to the sampler’s <code>update</code> before excluding it
from reward-based training. RFC 013 remains a proposal until accepted; this
integration is subject to its acceptance and final unscored-rollout semantics.
Ending the episode cancels every live job. <code>reset</code> cancels the previous
episode’s jobs first and raises if HF Jobs does not confirm a cancellation,
since that job may still be running and billing.</p> <!> <p>This environment is eval-first today. Post-training episodes can take hours,
beyond the staleness window of asynchronous GRPO trainers. Use it for evaluation;
training integration needs a service that fits the trainer’s staleness window or
a stand-in for the slow service.</p> <p><code>submit</code> blocks for up to <code>score_timeout_s</code>, reported in reset metadata (2700 s
for <code>t0_arithmetic_format</code>). Size the client’s tool/message timeout above that
value, with room for cancellation and transport overhead; the examples use <code>message_timeout_s=3600</code>.</p> <p>Read <code>final.reward</code> and <code>final.observation.metadata["service_spend"]</code> from the
final <code>step</code> result. For RFC 013 integration, forward one settled amount per
rollout, for example <code>service_spend["settled"]["usd"]</code>, as <code>costs</code> in the trainer’s
chosen unit. Spend is accounting, never a sampling weight. <code>final: true</code> means
all jobs have settled, even mid-episode; it does not mean the episode is done.
Use the step result’s <code>done</code> for that. An orphan keeps spend <code>final: false</code> and
may still bill; retain that accounting even when reward is <code>None</code>.</p> <!> <table><thead><tr><th>Task</th><th>What the agent does</th><th>Hardware</th><th>Budget</th></tr></thead><tbody><tr><td><code>t0_artifact</code></td><td>Write the episode’s nonce to <code>/data/outputs/answer.txt</code>. Checks the plumbing</td><td><code>cpu-basic</code></td><td>$0.05</td></tr><tr><td><code>t0_arithmetic_format</code></td><td>Fine-tune <code>HuggingFaceTB/SmolLM2-135M-Instruct</code> to answer two-digit addition as <code>#### &#123;a+b&#125;</code>. Reward is exact-match accuracy on a hidden test split</td><td><code>cpu-upgrade</code>, <code>t4-small</code>, <code>a10g-small</code></td><td>$3.00</td></tr></tbody></table> <!> <!> <p>The local entry point binds <code>127.0.0.1:8000</code>. The Docker command explicitly
binds <code>0.0.0.0:8000</code> so a private host can reach the published port. This
experimental env is excluded from automatic Spaces releases via <code>SKIP_HF_DEPLOYMENT</code>; do not publish it to Spaces or add it to HF collections.</p> <p>Configuration, all read by the server only:</p> <table><thead><tr><th>Variable</th><th>Meaning</th></tr></thead><tbody><tr><td><code>HF_TOKEN</code></td><td>Broker token. A fine-grained token with “Start and manage Jobs” plus write access to <code>&lt;namespace&gt;/jobs-artifacts</code>. A Jobs-only token is refused when the script is uploaded</td></tr><tr><td><code>POSTTRAIN_NAMESPACE</code></td><td>Namespace jobs run and bill in. Defaults to the token’s owner</td></tr><tr><td><code>POSTTRAIN_RESOURCE_GROUP_ID</code></td><td>Resource Group to bill. Its Jobs spend limit caps total spend even if the ledger is wrong</td></tr><tr><td><code>POSTTRAIN_TEST_SEED</code></td><td>Seed of the hidden test split. Set it for any run whose scores you compare: if unset it is random per process (the server logs a warning), so scores are only comparable within one server process. Keep it secret from the agent</td></tr></tbody></table> <p>Run the server where its token is not shared with the agent, for example a
container you control. Do not deploy it as a public Space: a Space gives no
isolation guarantee for the credential it holds.</p> <!> <!> <p><a href="https://github.com/openenv/OpenEnv/blob/main/examples/posttrain_env_agent.py" rel="nofollow"><code>examples/posttrain_env_agent.py</code></a> plays a whole episode with an LLM through any OpenAI-compatible endpoint,
Hugging Face Inference Providers by default. The agent’s API key is its own;
it never needs the broker’s token.</p> <!> <ul><li>The server has no authentication of its own: anyone who can reach it can
start episodes and spend its budget. Keep it on localhost or a private network.</li> <li>Job folders stay in the <code>jobs-artifacts</code> bucket, and the size of what a job
writes there is not capped or charged to the budget.</li> <li><code>max_steps</code> and the time limit are checked in <code>step</code>; a client calling the
MCP endpoint directly bypasses them, though not the budget. Core enforcement
is tracked in <a href="https://github.com/openenv/OpenEnv/issues/1507" rel="nofollow">#1507</a>.</li> <li><code>torch.distributed</code> over localhost TCP is blocked in agent jobs, so tasks use
single-GPU flavors.</li></ul> <!> <!> <p>The tests replace HF Jobs with an in-memory fake, so they make no network calls
and spend nothing.</p> <!> <p></p>`,1);function H(f,A){S(A,!1),V(()=>{new URLSearchParams(window.location.search).get("fw")}),k();var o=W();E("14xc4n6",g=>{var I=z();q(I,"content",Z),C(g,I)});var a=e(N(o),2);R(a,{containerStyle:"float: inline-end; margin-inline-start: 10px; display: inline-flex; position: relative; z-index: 10;"});var n=e(a,2);t(n,{title:"Post-training Environment (experimental)",local:"post-training-environment-experimental",headingTag:"h1"});var l=e(n,6);t(l,{title:"Tools",local:"tools",headingTag:"h2"});var d=e(l,6);t(d,{title:"How a job runs",local:"how-a-job-runs",headingTag:"h2"});var i=e(d,4);s(i,{code:"b3BlbmVudi1hcnRpZmFjdCUzQSUyRiUyRiUzQ2VwaXNvZGUlM0UlMkYlM0Nqb2JfaWQlM0UlMkZvdXRwdXRz",highlighted:'openenv-artifact://<span class="hljs-tag">&lt;<span class="hljs-name">episode</span>&gt;</span>/<span class="hljs-tag">&lt;<span class="hljs-name">job_id</span>&gt;</span>/outputs',lang:"",wrap:!1});var r=e(i,4);t(r,{title:"No network in agent jobs",local:"no-network-in-agent-jobs",headingTag:"h3"});var c=e(r,10);t(c,{title:"Evaluations",local:"evaluations",headingTag:"h3"});var p=e(c,4);t(p,{title:"Budget and spend",local:"budget-and-spend",headingTag:"h2"});var h=e(p,6);s(h,{code:"b2JzLm1ldGFkYXRhJTVCJTIyc2VydmljZV9zcGVuZCUyMiU1RCUwQSUyMyUyMCU3QiUyMnZlcnNpb24lMjIlM0ElMjAxJTJDJTBBJTIzJTIwJTIwJTIyc2V0dGxlZCUyMiUzQSUyMCU3QiUyMnVzZCUyMiUzQSUyMDAuMDAwMzM0JTJDJTIwJTIyZ3B1X3MlMjIlM0ElMjAwLjAlMkMlMjAlMjJqb2JzJTIyJTNBJTIwMSUyQyUyMCUyMmNhbGxzJTIyJTNBJTIwMSU3RCUyQyUwQSUyMyUyMCUyMCUyMnJlc2VydmVkJTIyJTNBJTIwJTdCJTIydXNkJTIyJTNBJTIwMC4wJTdEJTJDJTBBJTIzJTIwJTIwJTIyZmluYWwlMjIlM0ElMjBUcnVlJTdE",highlighted:`obs.metadata[<span class="hljs-string">&quot;service_spend&quot;</span>]
<span class="hljs-comment"># {&quot;version&quot;: 1,</span>
<span class="hljs-comment">#  &quot;settled&quot;: {&quot;usd&quot;: 0.000334, &quot;gpu_s&quot;: 0.0, &quot;jobs&quot;: 1, &quot;calls&quot;: 1},</span>
<span class="hljs-comment">#  &quot;reserved&quot;: {&quot;usd&quot;: 0.0},</span>
<span class="hljs-comment">#  &quot;final&quot;: True}</span>`,lang:"python",wrap:!1});var M=e(h,4);t(M,{title:"Episode end",local:"episode-end",headingTag:"h2"});var u=e(M,8);t(u,{title:"Using it with a trainer",local:"using-it-with-a-trainer",headingTag:"h2"});var y=e(u,8);t(y,{title:"Tasks",local:"tasks",headingTag:"h2"});var w=e(y,4);t(w,{title:"Running the server",local:"running-the-server",headingTag:"h2"});var T=e(w,2);s(T,{code:"ZXhwb3J0JTIwSEZfVE9LRU4lM0QuLi4lMjAlMjAlMjAlMjMlMjB0aGUlMjBicm9rZXIncyUyMHRva2VuJTJDJTIwc2VlJTIwYmVsb3clMEFjZCUyMGVudnMlMkZwb3N0dHJhaW5fZW52JTBBdXYlMjBydW4lMjAtLXByb2plY3QlMjAuJTIwc2VydmVy",highlighted:`<span class="hljs-built_in">export</span> HF_TOKEN=...   <span class="hljs-comment"># the broker&#x27;s token, see below</span>
<span class="hljs-built_in">cd</span> envs/posttrain_env
uv run --project . server`,lang:"bash",wrap:!1});var J=e(T,10);t(J,{title:"Client",local:"client",headingTag:"h2"});var j=e(J,2);s(j,{code:"ZnJvbSUyMG9wZW5lbnYuY29yZS5lbnZfc2VydmVyLm1jcF90eXBlcyUyMGltcG9ydCUyMENhbGxUb29sQWN0aW9uJTBBZnJvbSUyMHBvc3R0cmFpbl9lbnYlMjBpbXBvcnQlMjBQb3N0VHJhaW5FbnYlMEElMEF3aXRoJTIwUG9zdFRyYWluRW52KCUwQSUyMCUyMCUyMCUyMGJhc2VfdXJsJTNEJTIyaHR0cCUzQSUyRiUyRmxvY2FsaG9zdCUzQTgwMDAlMjIlMkMlMjBtZXNzYWdlX3RpbWVvdXRfcyUzRDM2MDAlMEEpLnN5bmMoKSUyMGFzJTIwZW52JTNBJTBBJTIwJTIwJTIwJTIwcmVzdWx0JTIwJTNEJTIwZW52LnJlc2V0KHRhc2tfaWQlM0QlMjJ0MF9hcnRpZmFjdCUyMiklMEElMjAlMjAlMjAlMjBwcmludChyZXN1bHQub2JzZXJ2YXRpb24ubWV0YWRhdGElNUIlMjJwcm9tcHQlMjIlNUQpJTBBJTIwJTIwJTIwJTIwcmVzdWx0JTIwJTNEJTIwZW52LnN0ZXAoJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwQ2FsbFRvb2xBY3Rpb24oJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwdG9vbF9uYW1lJTNEJTIyam9ic19zdWJtaXQlMjIlMkMlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjBhcmd1bWVudHMlM0QlN0IlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjJzY3JpcHQlMjIlM0ElMjBvcGVuKCUyMnRyYWluLnB5JTIyKS5yZWFkKCklMkMlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjJmbGF2b3IlMjIlM0ElMjAlMjJjcHUtYmFzaWMlMjIlMkMlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjJ0aW1lb3V0X21pbnV0ZXMlMjIlM0ElMjA1JTJDJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTdEJTJDJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwKSUwQSUyMCUyMCUyMCUyMCklMEElMjAlMjAlMjAlMjBhc3NlcnQlMjByZXN1bHQub2JzZXJ2YXRpb24uZXJyb3IlMjBpcyUyME5vbmUlMEElMjAlMjAlMjAlMjBqb2IlMjAlM0QlMjByZXN1bHQub2JzZXJ2YXRpb24ucmVzdWx0JTVCJTIyZGF0YSUyMiU1RCUwQSUyMCUyMCUyMCUyMGFzc2VydCUyMCUyMmVycm9yJTIyJTIwbm90JTIwaW4lMjBqb2IlMkMlMjBqb2IlMEElMjAlMjAlMjAlMjBzdGF0dXMlMjAlM0QlMjAlMjJxdWV1ZWQlMjIlMEElMjAlMjAlMjAlMjB3aGlsZSUyMHN0YXR1cyUyMGluJTIwJTdCJTIycXVldWVkJTIyJTJDJTIwJTIycnVubmluZyUyMiU3RCUzQSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMHJlc3VsdCUyMCUzRCUyMGVudi5zdGVwKCUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMENhbGxUb29sQWN0aW9uKCUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMHRvb2xfbmFtZSUzRCUyMmpvYnNfd2FpdCUyMiUyQyUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMGFyZ3VtZW50cyUzRCU3QiUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMmpvYl9pZCUyMiUzQSUyMGpvYiU1QiUyMmpvYl9pZCUyMiU1RCUyQyUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMnRpbWVvdXRfcyUyMiUzQSUyMDMwMCUyQyUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCU3RCUyQyUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCklMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjApJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwYXNzZXJ0JTIwbm90JTIwcmVzdWx0LmRvbmUlMkMlMjByZXN1bHQub2JzZXJ2YXRpb24ubWV0YWRhdGElMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjBzdGF0dXMlMjAlM0QlMjByZXN1bHQub2JzZXJ2YXRpb24ucmVzdWx0JTVCJTIyZGF0YSUyMiU1RCU1QiUyMnN0YXRlJTIyJTVEJTBBJTIwJTIwJTIwJTIwYXNzZXJ0JTIwc3RhdHVzJTIwJTNEJTNEJTIwJTIyc3VjY2VlZGVkJTIyJTJDJTIwcmVzdWx0Lm9ic2VydmF0aW9uLnJlc3VsdCUwQSUyMCUyMCUyMCUyMGZpbmFsJTIwJTNEJTIwZW52LnN0ZXAoJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwQ2FsbFRvb2xBY3Rpb24oJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwdG9vbF9uYW1lJTNEJTIyc3VibWl0JTIyJTJDJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwYXJndW1lbnRzJTNEJTdCJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIyam9iX2lkJTIyJTNBJTIwam9iJTVCJTIyam9iX2lkJTIyJTVEJTJDJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTdEJTJDJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwKSUwQSUyMCUyMCUyMCUyMCklMEElMjAlMjAlMjAlMjBwcmludChmaW5hbC5kb25lJTJDJTIwZmluYWwucmV3YXJkKSUwQSUyMCUyMCUyMCUyMHByaW50KGZpbmFsLm9ic2VydmF0aW9uLm1ldGFkYXRhJTVCJTIyc2VydmljZV9zcGVuZCUyMiU1RCk=",highlighted:`<span class="hljs-keyword">from</span> openenv.core.env_server.mcp_types <span class="hljs-keyword">import</span> CallToolAction
<span class="hljs-keyword">from</span> posttrain_env <span class="hljs-keyword">import</span> PostTrainEnv

<span class="hljs-keyword">with</span> PostTrainEnv(
    base_url=<span class="hljs-string">&quot;http://localhost:8000&quot;</span>, message_timeout_s=<span class="hljs-number">3600</span>
).sync() <span class="hljs-keyword">as</span> env:
    result = env.reset(task_id=<span class="hljs-string">&quot;t0_artifact&quot;</span>)
    <span class="hljs-built_in">print</span>(result.observation.metadata[<span class="hljs-string">&quot;prompt&quot;</span>])
    result = env.step(
        CallToolAction(
            tool_name=<span class="hljs-string">&quot;jobs_submit&quot;</span>,
            arguments={
                <span class="hljs-string">&quot;script&quot;</span>: <span class="hljs-built_in">open</span>(<span class="hljs-string">&quot;train.py&quot;</span>).read(),
                <span class="hljs-string">&quot;flavor&quot;</span>: <span class="hljs-string">&quot;cpu-basic&quot;</span>,
                <span class="hljs-string">&quot;timeout_minutes&quot;</span>: <span class="hljs-number">5</span>,
            },
        )
    )
    <span class="hljs-keyword">assert</span> result.observation.error <span class="hljs-keyword">is</span> <span class="hljs-literal">None</span>
    job = result.observation.result[<span class="hljs-string">&quot;data&quot;</span>]
    <span class="hljs-keyword">assert</span> <span class="hljs-string">&quot;error&quot;</span> <span class="hljs-keyword">not</span> <span class="hljs-keyword">in</span> job, job
    status = <span class="hljs-string">&quot;queued&quot;</span>
    <span class="hljs-keyword">while</span> status <span class="hljs-keyword">in</span> {<span class="hljs-string">&quot;queued&quot;</span>, <span class="hljs-string">&quot;running&quot;</span>}:
        result = env.step(
            CallToolAction(
                tool_name=<span class="hljs-string">&quot;jobs_wait&quot;</span>,
                arguments={
                    <span class="hljs-string">&quot;job_id&quot;</span>: job[<span class="hljs-string">&quot;job_id&quot;</span>],
                    <span class="hljs-string">&quot;timeout_s&quot;</span>: <span class="hljs-number">300</span>,
                },
            )
        )
        <span class="hljs-keyword">assert</span> <span class="hljs-keyword">not</span> result.done, result.observation.metadata
        status = result.observation.result[<span class="hljs-string">&quot;data&quot;</span>][<span class="hljs-string">&quot;state&quot;</span>]
    <span class="hljs-keyword">assert</span> status == <span class="hljs-string">&quot;succeeded&quot;</span>, result.observation.result
    final = env.step(
        CallToolAction(
            tool_name=<span class="hljs-string">&quot;submit&quot;</span>,
            arguments={
                <span class="hljs-string">&quot;job_id&quot;</span>: job[<span class="hljs-string">&quot;job_id&quot;</span>],
            },
        )
    )
    <span class="hljs-built_in">print</span>(final.done, final.reward)
    <span class="hljs-built_in">print</span>(final.observation.metadata[<span class="hljs-string">&quot;service_spend&quot;</span>])`,lang:"python",wrap:!1});var b=e(j,4);t(b,{title:"Known gaps",local:"known-gaps",headingTag:"h2"});var U=e(b,4);t(U,{title:"Tests",local:"tests",headingTag:"h2"});var m=e(U,2);s(m,{code:"UFlUSE9OUEFUSCUzRHNyYyUzQWVudnMlMjB1diUyMHJ1biUyMHB5dGVzdCUyMHRlc3RzJTJGZW52cyUyRnRlc3RfcG9zdHRyYWluX2Vudmlyb25tZW50LnB5JTIwLXY=",highlighted:"PYTHONPATH=src:envs uv run pytest tests/envs/test_posttrain_environment.py -v",lang:"bash",wrap:!1});var _=e(m,4);B(_,{source:"https://github.com/openenv/OpenEnv/blob/main/docs/source/environments/posttrain.md"}),F(2),C(f,o),Q()}export{H as component};
