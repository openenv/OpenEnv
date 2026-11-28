import"../chunks/DsnmJJEf.js";import{i as k,h as g,C as E,H as l,a,E as G,s as V}from"../chunks/Dz7d4_v9.js";import{p as N,o as X,s,f as S,a as C,b as f,c as v,n as Q}from"../chunks/BSuxAqoA.js";const W='{"title":"Evaluating with Environments","local":"evaluating-with-environments","sections":[{"title":"How the pieces fit together","local":"how-the-pieces-fit-together","sections":[],"depth":2},{"title":"Install dependencies","local":"install-dependencies","sections":[],"depth":2},{"title":"Set your model provider","local":"set-your-model-provider","sections":[],"depth":2},{"title":"Define an Inspect AI task for an OpenEnv environment","local":"define-an-inspect-ai-task-for-an-openenv-environment","sections":[],"depth":2},{"title":"Run the eval with InspectAIHarness","local":"run-the-eval-with-inspectaiharness","sections":[],"depth":2},{"title":"Using a task file instead of a task object","local":"using-a-task-file-instead-of-a-task-object","sections":[],"depth":2},{"title":"Adapting to your own environment and task","local":"adapting-to-your-own-environment-and-task","sections":[],"depth":2},{"title":"Next steps","local":"next-steps","sections":[],"depth":2}],"depth":1}';var R=v('<meta name="hf:doc:metadata"/>'),Y=v(`<p></p> <!> <!> <p><a href="https://colab.research.google.com/github/openenv/OpenEnv/blob/main/examples/evaluation_inspect.ipynb" rel="nofollow"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a></p> <p>After training a model in an OpenEnv environment, you need to measure how it
actually performs on a held-out set of episodes. OpenEnv integrates with <a href="https://inspect.aisi.org.uk/" rel="nofollow">Inspect AI</a> — an open-source evaluation
framework by the UK AI Safety Institute — through <code>InspectAIHarness</code>.</p> <!> <p>Inspect AI and OpenEnv are complementary, not overlapping:</p> <ul><li><strong>OpenEnv</strong> provides the environment (reset, step, reward) and the training
infrastructure (GRPO via TRL).</li> <li><strong>Inspect AI</strong> provides the evaluation infrastructure: datasets, solvers,
scorers, and structured logs.</li></ul> <p><code>InspectAIHarness</code> is the bridge. It wraps <code>inspect_ai.eval()</code> inside
OpenEnv’s <code>EvalHarness</code> interface so that eval runs are tracked with the same
structured <code>EvalConfig</code> / <code>EvalResult</code> types you use across all harnesses.</p> <p>The typical workflow is:</p> <!> <!> <!> <p>The <code>inspect</code> extra pulls in <code>inspect-ai</code> (pinned to <code>&gt;=0.3.0</code>).</p> <p><code>inspect-ai</code> is an optional dependency — <code>InspectAIHarness</code> is importable
without it, but raises a clear <code>ImportError</code> at call time if it is missing.</p> <!> <p>Uncomment exactly one option. All three feed into the same task and harness —
no other cells need to change.</p> <!> <p>The <code>model</code> string uses <code>provider/model-name</code> format for API providers.
For local models, the <code>hf/</code> prefix loads the model with <code>transformers</code> — point
it at a Hub ID to download, or a local path (<code>hf/./path/to/checkpoint</code>) to use
weights you already have on disk (e.g. from TRL training).</p> <!> <p>An Inspect AI <code>Task</code> has three parts: a <strong>dataset</strong> of samples to evaluate,
a <strong>solver</strong> that runs the model (and optionally the environment), and a <strong>scorer</strong> that grades each sample.</p> <p>The example below evaluates a model against <code>echo_env</code> — the reference
OpenEnv environment. The model is asked to repeat a phrase; the solver sends
the phrase to the environment and records the echoed response; the scorer
checks it matches the expected output.</p> <p>The solver calls Inspect AI’s <code>generate()</code> to get the model’s output, then
sends it to the environment. The dataset, scorer, and harness are identical
for both providers.</p> <!> <blockquote class="note"><p><code>echo_env</code> is a pure MCP environment. Interact with it via <code>MCPToolClient</code> and <code>call_tool("echo_message", ...)</code>. For non-MCP environments, use <code>GenericEnvClient</code> instead.</p></blockquote> <!> <p>Pass the task to <code>InspectAIHarness</code> via <code>EvalConfig</code>. The <code>task</code> key in <code>eval_parameters</code> takes a task object or a registered task name string.</p> <!> <p>The <code>EvalResult</code> carries both the config and the scores, making it easy to
log, compare across runs, or serialize to JSON:</p> <!> <!> <p>Inspect AI tasks can also be defined in standalone <code>.py</code> files and referenced
by path. This is useful for CI pipelines where the task definition lives in
the repo and the harness is called from a script:</p> <!> <!> <p>Replace <code>echo_env_solver</code> with a solver that uses your env and model:</p> <ol><li><strong>Dataset</strong> — collect held-out episodes from your env (or a static
benchmark); each <code>Sample</code> needs <code>input</code> and <code>target</code> fields.</li> <li><strong>Solver</strong> — call your trained model against the env via <code>generate()</code>.
If you used GRPO training with an <code>environment_factory</code>, reuse the same
factory here so the eval env matches training exactly.</li> <li><strong>Scorer</strong> — use the env’s reward signal directly, or write an Inspect AI <code>@scorer</code> that checks the final observation against a ground-truth target.</li></ol> <blockquote class="tip"><p>Run this eval <strong>before training</strong> on your base model to establish a baseline,
then again after training to measure the improvement. The delta (post − pre)
is more informative than either number alone — a model that scores 60% after
training tells you little without knowing it started at 4%.</p></blockquote> <!> <!> <ul><li><a href="end-to-end-walkthrough">End-to-end walkthrough</a> — full GRPO training loop that produces a model you can evaluate with this tutorial</li> <li><a href="sft-warmup">SFT warm-up tutorial</a> — collect rollouts, filter by reward, and fine-tune a student model before running GRPO</li> <li><a href="rubrics">Rubrics tutorial</a> — define reward functions inside
the environment using composable rubrics</li> <li><a href="https://inspect.aisi.org.uk/" rel="nofollow">Inspect AI documentation</a> — full reference
for tasks, solvers, scorers, and the log viewer</li></ul> <!> <p></p>`,1);function H(B,A){N(A,!1),X(()=>{new URLSearchParams(window.location.search).get("fw")}),k();var e=Y();g("1a5wvd4",m=>{var b=R();V(b,"content",W),C(m,b)});var n=s(S(e),2);E(n,{containerStyle:"float: inline-end; margin-inline-start: 10px; display: inline-flex; position: relative; z-index: 10;"});var t=s(n,2);l(t,{title:"Evaluating with Environments",local:"evaluating-with-environments",headingTag:"h1"});var o=s(t,6);l(o,{title:"How the pieces fit together",local:"how-the-pieces-fit-together",headingTag:"h2"});var M=s(o,10);a(M,{code:"VHJhaW4lMjB3aXRoJTIwT3BlbkVudiUyMChHUlBPJTIwJTJGJTIwU0ZUKSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCVFMiU4NiU5MyUwQURlZmluZSUyMGFuJTIwSW5zcGVjdCUyMEFJJTIwVGFzayUwQSUyMCUyMC0lMjBkYXRhc2V0JTNBJTIwaGVsZC1vdXQlMjBlcGlzb2RlcyUyMG9yJTIwcHJvbXB0cyUwQSUyMCUyMC0lMjBzb2x2ZXIlM0ElMjBjYWxscyUyMHlvdXIlMjBtb2RlbCUyMCUyQiUyMHRoZSUyME9wZW5FbnYlMjBlbnYlMEElMjAlMjAtJTIwc2NvcmVyJTNBJTIwZ3JhZGVzJTIwY29ycmVjdG5lc3MlMjB1c2luZyUyMGVudiUyMHJld2FyZCUyMG9yJTIwZXhhY3QlMjBtYXRjaCUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCVFMiU4NiU5MyUwQVJ1biUyMHZpYSUyMEluc3BlY3RBSUhhcm5lc3MlMjAlRTIlODYlOTIlMjBFdmFsUmVzdWx0JTIwd2l0aCUyMHN0cnVjdHVyZWQlMjBzY29yZXM=",highlighted:`Train <span class="hljs-built_in">with</span> OpenEnv (GRPO / SFT)
        ↓
Define an <span class="hljs-keyword">Inspect</span> AI Task
  - dataset: held-out episodes or prompts
  - solver: calls your model + the OpenEnv env
  - scorer: grades correctness <span class="hljs-built_in">using</span> env reward or <span class="hljs-built_in">exact</span> <span class="hljs-keyword">match</span>
        ↓
Run via InspectAIHarness → EvalResult <span class="hljs-built_in">with</span> structured scores`,lang:"",wrap:!1});var c=s(M,2);l(c,{title:"Install dependencies",local:"install-dependencies",headingTag:"h2"});var p=s(c,2);a(p,{code:"cGlwJTIwaW5zdGFsbCUyMCUyMm9wZW5lbnYlNUJpbnNwZWN0JTVEJTIwJTQwJTIwZ2l0JTJCaHR0cHMlM0ElMkYlMkZnaXRodWIuY29tJTJGb3BlbmVudiUyRk9wZW5FbnYuZ2l0JTIy",highlighted:'pip install <span class="hljs-string">&quot;openenv[inspect] @ git+https://github.com/openenv/OpenEnv.git&quot;</span>',lang:"bash",wrap:!1});var r=s(p,6);l(r,{title:"Set your model provider",local:"set-your-model-provider",headingTag:"h2"});var y=s(r,4);a(y,{code:"aW1wb3J0JTIwZ2V0cGFzcyUyQyUyMG9zJTBBJTBBJTIzJTIwLS0tJTIwT3B0aW9uJTIwQSUzQSUyME9wZW5BSSUyMC0tLSUwQW9zLmVudmlyb24uc2V0ZGVmYXVsdCglMjJPUEVOQUlfQVBJX0tFWSUyMiUyQyUyMGdldHBhc3MuZ2V0cGFzcyglMjJPcGVuQUklMjBBUEklMjBrZXklM0ElMjAlMjIpKSUwQU1PREVMJTIwJTNEJTIwJTIyb3BlbmFpJTJGZ3B0LTUtbWluaSUyMiUwQSUwQSUyMyUyMC0tLSUyME9wdGlvbiUyMEIlM0ElMjBBbnRocm9waWMlMjAtLS0lMEElMjMlMjBvcy5lbnZpcm9uLnNldGRlZmF1bHQoJTIyQU5USFJPUElDX0FQSV9LRVklMjIlMkMlMjBnZXRwYXNzLmdldHBhc3MoJTIyQW50aHJvcGljJTIwQVBJJTIwa2V5JTNBJTIwJTIyKSklMEElMjMlMjBNT0RFTCUyMCUzRCUyMCUyMmFudGhyb3BpYyUyRmNsYXVkZS1oYWlrdS00LTUtMjAyNTEwMDElMjIlMEElMEElMjMlMjAtLS0lMjBPcHRpb24lMjBDJTNBJTIwbG9jYWwlMjB0cmFuc2Zvcm1lcnMlMjBtb2RlbCUyMChubyUyMEFQSSUyMGtleSUyMG5lZWRlZCklMjAtLS0lMEElMjMlMjBSZXF1aXJlcyUyMGElMjBHUFUlMjBmb3IlMjByZWFzb25hYmxlJTIwc3BlZWQuJTIwT21pdCUyMCd0ZW1wZXJhdHVyZSclMjBmcm9tJTIwZXZhbF9wYXJhbWV0ZXJzJTIwYmVsb3cuJTBBJTIzJTIwIXBpcCUyMGluc3RhbGwlMjAtVSUyMHRyYW5zZm9ybWVycyUwQSUyMyUyME1PREVMJTIwJTNEJTIwJTIyaGYlMkZRd2VuJTJGUXdlbjMuNS0wLjhCJTIyJTBBJTIzJTIwVXNlJTIwYSUyMGxvY2FsJTIwY2hlY2twb2ludCUyMHBhdGglMjB0byUyMHNraXAlMjB0aGUlMjBkb3dubG9hZCUzQSUwQSUyMyUyME1PREVMJTIwJTNEJTIwJTIyaGYlMkYuJTJGb3V0cHV0cyUyRm15LXRyYWluZWQtbW9kZWwlMjI=",highlighted:`<span class="hljs-keyword">import</span> getpass, os

<span class="hljs-comment"># --- Option A: OpenAI ---</span>
os.environ.setdefault(<span class="hljs-string">&quot;OPENAI_API_KEY&quot;</span>, getpass.getpass(<span class="hljs-string">&quot;OpenAI API key: &quot;</span>))
MODEL = <span class="hljs-string">&quot;openai/gpt-5-mini&quot;</span>

<span class="hljs-comment"># --- Option B: Anthropic ---</span>
<span class="hljs-comment"># os.environ.setdefault(&quot;ANTHROPIC_API_KEY&quot;, getpass.getpass(&quot;Anthropic API key: &quot;))</span>
<span class="hljs-comment"># MODEL = &quot;anthropic/claude-haiku-4-5-20251001&quot;</span>

<span class="hljs-comment"># --- Option C: local transformers model (no API key needed) ---</span>
<span class="hljs-comment"># Requires a GPU for reasonable speed. Omit &#x27;temperature&#x27; from eval_parameters below.</span>
<span class="hljs-comment"># !pip install -U transformers</span>
<span class="hljs-comment"># MODEL = &quot;hf/Qwen/Qwen3.5-0.8B&quot;</span>
<span class="hljs-comment"># Use a local checkpoint path to skip the download:</span>
<span class="hljs-comment"># MODEL = &quot;hf/./outputs/my-trained-model&quot;</span>`,lang:"python",wrap:!1});var i=s(y,4);l(i,{title:"Define an Inspect AI task for an OpenEnv environment",local:"define-an-inspect-ai-task-for-an-openenv-environment",headingTag:"h2"});var J=s(i,8);a(J,{code:"aW1wb3J0JTIwYXN5bmNpbyUwQSUwQWZyb20lMjBpbnNwZWN0X2FpJTIwaW1wb3J0JTIwVGFzayUyQyUyMHRhc2slMEFmcm9tJTIwaW5zcGVjdF9haS5kYXRhc2V0JTIwaW1wb3J0JTIwU2FtcGxlJTBBZnJvbSUyMGluc3BlY3RfYWkuc2NvcmVyJTIwaW1wb3J0JTIwQ09SUkVDVCUyQyUyMElOQ09SUkVDVCUyQyUyMFNjb3JlJTJDJTIwVGFyZ2V0JTJDJTIwYWNjdXJhY3klMkMlMjBzY29yZXIlMEFmcm9tJTIwaW5zcGVjdF9haS5zb2x2ZXIlMjBpbXBvcnQlMjBHZW5lcmF0ZSUyQyUyMFRhc2tTdGF0ZSUyQyUyMHNvbHZlciUwQSUwQWZyb20lMjBvcGVuZW52LmNvcmUlMjBpbXBvcnQlMjBNQ1BUb29sQ2xpZW50JTBBJTBBRUNIT19FTlZfVVJMJTIwJTNEJTIwJTIyaHR0cHMlM0ElMkYlMkZvcGVuZW52LWVjaG8tZW52LmhmLnNwYWNlJTIyJTBBJTBBJTIzJTIwTGltaXQlMjBjb25jdXJyZW50JTIwZW52JTIwY29ubmVjdGlvbnMlMjB0byUyMG1hdGNoJTIwdGhlJTIwc2VydmVyJ3MlMjBNQVhfQ09OQ1VSUkVOVF9FTlZTLiUwQV9lbnZfc2VtJTIwJTNEJTIwYXN5bmNpby5TZW1hcGhvcmUoMSklMjAlMjAlMjMlMjBpbmNyZWFzZSUyMGlmJTIweW91ciUyMFNwYWNlJTIwc3VwcG9ydHMlMjBtb3JlJTIwc2Vzc2lvbnMlMEElMEElMEElNDB0YXNrJTBBZGVmJTIwb3BlbmVudl9lY2hvX2V2YWwoYmFzZV91cmwlM0ElMjBzdHIlMjAlM0QlMjBFQ0hPX0VOVl9VUkwpJTNBJTBBJTIwJTIwJTIwJTIwcmV0dXJuJTIwVGFzayglMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjBkYXRhc2V0JTNEJTVCJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwU2FtcGxlKGlucHV0JTNEJTIyUmVwZWF0JTIwZXhhY3RseSUzQSUyMGhlbGxvJTIwd29ybGQlMjIlMkMlMjB0YXJnZXQlM0QlMjJoZWxsbyUyMHdvcmxkJTIyKSUyQyUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMFNhbXBsZShpbnB1dCUzRCUyMlJlcGVhdCUyMGV4YWN0bHklM0ElMjBpbnNwZWN0JTIwYWklMjIlMkMlMjB0YXJnZXQlM0QlMjJpbnNwZWN0JTIwYWklMjIpJTJDJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwU2FtcGxlKGlucHV0JTNEJTIyUmVwZWF0JTIwZXhhY3RseSUzQSUyMG9wZW5lbnYlMjBldmFsJTIyJTJDJTIwdGFyZ2V0JTNEJTIyb3BlbmVudiUyMGV2YWwlMjIpJTJDJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwU2FtcGxlKGlucHV0JTNEJTIyUmVwZWF0JTIwZXhhY3RseSUzQSUyMHJlaW5mb3JjZW1lbnQlMjBsZWFybmluZyUyMiUyQyUyMHRhcmdldCUzRCUyMnJlaW5mb3JjZW1lbnQlMjBsZWFybmluZyUyMiklMkMlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjBTYW1wbGUoaW5wdXQlM0QlMjJSZXBlYXQlMjBleGFjdGx5JTNBJTIwaHVnZ2luZyUyMGZhY2UlMjIlMkMlMjB0YXJnZXQlM0QlMjJodWdnaW5nJTIwZmFjZSUyMiklMkMlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlNUQlMkMlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjBzb2x2ZXIlM0RlY2hvX2Vudl9zb2x2ZXIoYmFzZV91cmwlM0RiYXNlX3VybCklMkMlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjBzY29yZXIlM0RlY2hvX3Njb3JlcigpJTJDJTBBJTIwJTIwJTIwJTIwKSUwQSUwQSUwQSU0MHNvbHZlciUwQWRlZiUyMGVjaG9fZW52X3NvbHZlcihiYXNlX3VybCUzQSUyMHN0ciklM0ElMEElMjAlMjAlMjAlMjAlMjIlMjIlMjJBc2slMjB0aGUlMjBtb2RlbCUyMHRvJTIwcmVwZWF0JTIwdGhlJTIwcGhyYXNlJTJDJTIwdGhlbiUyMGVjaG8lMjBpdCUyMHRocm91Z2glMjB0aGUlMjBlbnYuJTIyJTIyJTIyJTBBJTBBJTIwJTIwJTIwJTIwYXN5bmMlMjBkZWYlMjBzb2x2ZShzdGF0ZSUzQSUyMFRhc2tTdGF0ZSUyQyUyMGdlbmVyYXRlJTNBJTIwR2VuZXJhdGUpJTIwLSUzRSUyMFRhc2tTdGF0ZSUzQSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMHN0YXRlJTIwJTNEJTIwYXdhaXQlMjBnZW5lcmF0ZShzdGF0ZSklMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjBtb2RlbF9vdXRwdXQlMjAlM0QlMjBzdGF0ZS5vdXRwdXQuY29tcGxldGlvbi5zdHJpcCgpJTBBJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwYXN5bmMlMjB3aXRoJTIwX2Vudl9zZW0lM0ElMjAlMjAlMjMlMjBvbmUlMjBlbnYlMjBjb25uZWN0aW9uJTIwYXQlMjBhJTIwdGltZSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMGVudiUyMCUzRCUyME1DUFRvb2xDbGllbnQoYmFzZV91cmwlM0RiYXNlX3VybCklMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjB0cnklM0ElMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjBhd2FpdCUyMGVudi5yZXNldCgpJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwZWNob2VkJTIwJTNEJTIwYXdhaXQlMjBlbnYuY2FsbF90b29sKCUyMmVjaG9fbWVzc2FnZSUyMiUyQyUyMG1lc3NhZ2UlM0Rtb2RlbF9vdXRwdXQpJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwc3RhdGUubWV0YWRhdGElNUIlMjJlY2hvZWQlMjIlNUQlMjAlM0QlMjBzdHIoZWNob2VkKSUyMGlmJTIwZWNob2VkJTIwaXMlMjBub3QlMjBOb25lJTIwZWxzZSUyMCUyMiUyMiUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMGZpbmFsbHklM0ElMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjBhd2FpdCUyMGVudi5jbG9zZSgpJTBBJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwcmV0dXJuJTIwc3RhdGUlMEElMEElMjAlMjAlMjAlMjByZXR1cm4lMjBzb2x2ZSUwQSUwQSUwQSU0MHNjb3JlcihtZXRyaWNzJTNEJTVCYWNjdXJhY3koKSU1RCklMEFkZWYlMjBlY2hvX3Njb3JlcigpJTNBJTBBJTIwJTIwJTIwJTIwJTIyJTIyJTIyQ09SUkVDVCUyMGlmJTIwdGhlJTIwZW52JTIwZWNob2VkJTIwYmFjayUyMGV4YWN0bHklMjB3aGF0JTIwdGhlJTIwdGFyZ2V0JTIwcGhyYXNlJTIwd2FzLiUyMiUyMiUyMiUwQSUwQSUyMCUyMCUyMCUyMGFzeW5jJTIwZGVmJTIwc2NvcmUoc3RhdGUlM0ElMjBUYXNrU3RhdGUlMkMlMjB0YXJnZXQlM0ElMjBUYXJnZXQpJTIwLSUzRSUyMFNjb3JlJTNBJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwZWNob2VkJTIwJTNEJTIwc3RhdGUubWV0YWRhdGEuZ2V0KCUyMmVjaG9lZCUyMiUyQyUyMCUyMiUyMikuc3RyaXAoKSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMGV4cGVjdGVkJTIwJTNEJTIwdGFyZ2V0LnRleHQuc3RyaXAoKSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMHJldHVybiUyMFNjb3JlKCUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMHZhbHVlJTNEQ09SUkVDVCUyMGlmJTIwZWNob2VkJTIwJTNEJTNEJTIwZXhwZWN0ZWQlMjBlbHNlJTIwSU5DT1JSRUNUJTJDJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwZXhwbGFuYXRpb24lM0RmJTIyRW52JTIwZWNob2VkJTIwJTdCZWNob2VkIXIlN0QlMkMlMjBleHBlY3RlZCUyMCU3QmV4cGVjdGVkIXIlN0QlMjIlMkMlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjApJTBBJTBBJTIwJTIwJTIwJTIwcmV0dXJuJTIwc2NvcmU=",highlighted:`<span class="hljs-keyword">import</span> asyncio

<span class="hljs-keyword">from</span> inspect_ai <span class="hljs-keyword">import</span> Task, task
<span class="hljs-keyword">from</span> inspect_ai.dataset <span class="hljs-keyword">import</span> Sample
<span class="hljs-keyword">from</span> inspect_ai.scorer <span class="hljs-keyword">import</span> CORRECT, INCORRECT, Score, Target, accuracy, scorer
<span class="hljs-keyword">from</span> inspect_ai.solver <span class="hljs-keyword">import</span> Generate, TaskState, solver

<span class="hljs-keyword">from</span> openenv.core <span class="hljs-keyword">import</span> MCPToolClient

ECHO_ENV_URL = <span class="hljs-string">&quot;https://openenv-echo-env.hf.space&quot;</span>

<span class="hljs-comment"># Limit concurrent env connections to match the server&#x27;s MAX_CONCURRENT_ENVS.</span>
_env_sem = asyncio.Semaphore(<span class="hljs-number">1</span>)  <span class="hljs-comment"># increase if your Space supports more sessions</span>


<span class="hljs-meta">@task</span>
<span class="hljs-keyword">def</span> <span class="hljs-title function_">openenv_echo_eval</span>(<span class="hljs-params">base_url: <span class="hljs-built_in">str</span> = ECHO_ENV_URL</span>):
    <span class="hljs-keyword">return</span> Task(
        dataset=[
            Sample(<span class="hljs-built_in">input</span>=<span class="hljs-string">&quot;Repeat exactly: hello world&quot;</span>, target=<span class="hljs-string">&quot;hello world&quot;</span>),
            Sample(<span class="hljs-built_in">input</span>=<span class="hljs-string">&quot;Repeat exactly: inspect ai&quot;</span>, target=<span class="hljs-string">&quot;inspect ai&quot;</span>),
            Sample(<span class="hljs-built_in">input</span>=<span class="hljs-string">&quot;Repeat exactly: openenv eval&quot;</span>, target=<span class="hljs-string">&quot;openenv eval&quot;</span>),
            Sample(<span class="hljs-built_in">input</span>=<span class="hljs-string">&quot;Repeat exactly: reinforcement learning&quot;</span>, target=<span class="hljs-string">&quot;reinforcement learning&quot;</span>),
            Sample(<span class="hljs-built_in">input</span>=<span class="hljs-string">&quot;Repeat exactly: hugging face&quot;</span>, target=<span class="hljs-string">&quot;hugging face&quot;</span>),
        ],
        solver=echo_env_solver(base_url=base_url),
        scorer=echo_scorer(),
    )


<span class="hljs-meta">@solver</span>
<span class="hljs-keyword">def</span> <span class="hljs-title function_">echo_env_solver</span>(<span class="hljs-params">base_url: <span class="hljs-built_in">str</span></span>):
    <span class="hljs-string">&quot;&quot;&quot;Ask the model to repeat the phrase, then echo it through the env.&quot;&quot;&quot;</span>

    <span class="hljs-keyword">async</span> <span class="hljs-keyword">def</span> <span class="hljs-title function_">solve</span>(<span class="hljs-params">state: TaskState, generate: Generate</span>) -&gt; TaskState:
        state = <span class="hljs-keyword">await</span> generate(state)
        model_output = state.output.completion.strip()

        <span class="hljs-keyword">async</span> <span class="hljs-keyword">with</span> _env_sem:  <span class="hljs-comment"># one env connection at a time</span>
            env = MCPToolClient(base_url=base_url)
            <span class="hljs-keyword">try</span>:
                <span class="hljs-keyword">await</span> env.reset()
                echoed = <span class="hljs-keyword">await</span> env.call_tool(<span class="hljs-string">&quot;echo_message&quot;</span>, message=model_output)
                state.metadata[<span class="hljs-string">&quot;echoed&quot;</span>] = <span class="hljs-built_in">str</span>(echoed) <span class="hljs-keyword">if</span> echoed <span class="hljs-keyword">is</span> <span class="hljs-keyword">not</span> <span class="hljs-literal">None</span> <span class="hljs-keyword">else</span> <span class="hljs-string">&quot;&quot;</span>
            <span class="hljs-keyword">finally</span>:
                <span class="hljs-keyword">await</span> env.close()

        <span class="hljs-keyword">return</span> state

    <span class="hljs-keyword">return</span> solve


<span class="hljs-meta">@scorer(<span class="hljs-params">metrics=[accuracy(<span class="hljs-params"></span>)]</span>)</span>
<span class="hljs-keyword">def</span> <span class="hljs-title function_">echo_scorer</span>():
    <span class="hljs-string">&quot;&quot;&quot;CORRECT if the env echoed back exactly what the target phrase was.&quot;&quot;&quot;</span>

    <span class="hljs-keyword">async</span> <span class="hljs-keyword">def</span> <span class="hljs-title function_">score</span>(<span class="hljs-params">state: TaskState, target: Target</span>) -&gt; Score:
        echoed = state.metadata.get(<span class="hljs-string">&quot;echoed&quot;</span>, <span class="hljs-string">&quot;&quot;</span>).strip()
        expected = target.text.strip()
        <span class="hljs-keyword">return</span> Score(
            value=CORRECT <span class="hljs-keyword">if</span> echoed == expected <span class="hljs-keyword">else</span> INCORRECT,
            explanation=<span class="hljs-string">f&quot;Env echoed <span class="hljs-subst">{echoed!r}</span>, expected <span class="hljs-subst">{expected!r}</span>&quot;</span>,
        )

    <span class="hljs-keyword">return</span> score`,lang:"python",wrap:!1});var w=s(J,4);l(w,{title:"Run the eval with InspectAIHarness",local:"run-the-eval-with-inspectaiharness",headingTag:"h2"});var T=s(w,4);a(T,{code:"aW1wb3J0JTIwaW5zcGVjdF9haSUwQWltcG9ydCUyMG9wZW5lbnYlMEElMEFmcm9tJTIwb3BlbmVudi5jb3JlLmV2YWxzJTIwaW1wb3J0JTIwRXZhbENvbmZpZyUyQyUyMEV2YWxSZXN1bHQlMkMlMjBJbnNwZWN0QUlIYXJuZXNzJTBBJTBBaGFybmVzcyUyMCUzRCUyMEluc3BlY3RBSUhhcm5lc3MobG9nX2RpciUzRCUyMi4lMkZldmFsLWxvZ3MlMjIpJTBBJTBBY29uZmlnJTIwJTNEJTIwRXZhbENvbmZpZyglMEElMjAlMjAlMjAlMjBoYXJuZXNzX25hbWUlM0QlMjJJbnNwZWN0QUlIYXJuZXNzJTIyJTJDJTBBJTIwJTIwJTIwJTIwaGFybmVzc192ZXJzaW9uJTNEaW5zcGVjdF9haS5fX3ZlcnNpb25fXyUyQyUwQSUyMCUyMCUyMCUyMGxpYnJhcnlfdmVyc2lvbnMlM0QlN0IlMjJvcGVuZW52JTIyJTNBJTIwb3BlbmVudi5fX3ZlcnNpb25fXyU3RCUyQyUwQSUyMCUyMCUyMCUyMGRhdGFzZXQlM0QlMjJvcGVuZW52X2VjaG9fZXZhbCUyMiUyQyUwQSUyMCUyMCUyMCUyMGV2YWxfcGFyYW1ldGVycyUzRCU3QiUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMm1vZGVsJTIyJTNBJTIwTU9ERUwlMkMlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjJ0YXNrJTIyJTNBJTIwb3BlbmVudl9lY2hvX2V2YWwoYmFzZV91cmwlM0RFQ0hPX0VOVl9VUkwpJTJDJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIzJTIwdGVtcGVyYXR1cmUlMjBpcyUyMHN1cHBvcnRlZCUyMGZvciUyMEFQSSUyMHByb3ZpZGVycyUyMChPcHRpb25zJTIwQSUyRkIpLiUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMyUyME9taXQlMjBpdCUyMGZvciUyMGxvY2FsJTIwdHJhbnNmb3JtZXJzJTIwbW9kZWxzJTIwKE9wdGlvbiUyMEMpLiUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMnRlbXBlcmF0dXJlJTIyJTNBJTIwMC4wJTJDJTBBJTIwJTIwJTIwJTIwJTdEJTJDJTBBKSUwQSUwQXJlc3VsdCUzQSUyMEV2YWxSZXN1bHQlMjAlM0QlMjBoYXJuZXNzLnJ1bl9mcm9tX2NvbmZpZyhjb25maWcpJTBBcHJpbnQocmVzdWx0LnNjb3JlcyklMEElMjMlMjAlN0InYWNjdXJhY3knJTNBJTIwMS4wJTdE",highlighted:`<span class="hljs-keyword">import</span> inspect_ai
<span class="hljs-keyword">import</span> openenv

<span class="hljs-keyword">from</span> openenv.core.evals <span class="hljs-keyword">import</span> EvalConfig, EvalResult, InspectAIHarness

harness = InspectAIHarness(log_dir=<span class="hljs-string">&quot;./eval-logs&quot;</span>)

config = EvalConfig(
    harness_name=<span class="hljs-string">&quot;InspectAIHarness&quot;</span>,
    harness_version=inspect_ai.__version__,
    library_versions={<span class="hljs-string">&quot;openenv&quot;</span>: openenv.__version__},
    dataset=<span class="hljs-string">&quot;openenv_echo_eval&quot;</span>,
    eval_parameters={
        <span class="hljs-string">&quot;model&quot;</span>: MODEL,
        <span class="hljs-string">&quot;task&quot;</span>: openenv_echo_eval(base_url=ECHO_ENV_URL),
        <span class="hljs-comment"># temperature is supported for API providers (Options A/B).</span>
        <span class="hljs-comment"># Omit it for local transformers models (Option C).</span>
        <span class="hljs-string">&quot;temperature&quot;</span>: <span class="hljs-number">0.0</span>,
    },
)

result: EvalResult = harness.run_from_config(config)
<span class="hljs-built_in">print</span>(result.scores)
<span class="hljs-comment"># {&#x27;accuracy&#x27;: 1.0}</span>`,lang:"python",wrap:!1});var j=s(T,4);a(j,{code:"aW1wb3J0JTIwanNvbiUwQSUwQWNsYXNzJTIwX1N0ckZhbGxiYWNrKGpzb24uSlNPTkVuY29kZXIpJTNBJTBBJTIwJTIwJTIwJTIwZGVmJTIwZGVmYXVsdChzZWxmJTJDJTIwbyklM0ElMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjByZXR1cm4lMjBzdHIobyklMEElMEFwcmludChqc29uLmR1bXBzKHJlc3VsdC5tb2RlbF9kdW1wKCklMkMlMjBpbmRlbnQlM0QyJTJDJTIwY2xzJTNEX1N0ckZhbGxiYWNrKSk=",highlighted:`<span class="hljs-keyword">import</span> json

<span class="hljs-keyword">class</span> <span class="hljs-title class_">_StrFallback</span>(json.JSONEncoder):
    <span class="hljs-keyword">def</span> <span class="hljs-title function_">default</span>(<span class="hljs-params">self, o</span>):
        <span class="hljs-keyword">return</span> <span class="hljs-built_in">str</span>(o)

<span class="hljs-built_in">print</span>(json.dumps(result.model_dump(), indent=<span class="hljs-number">2</span>, cls=_StrFallback))`,lang:"python",wrap:!1});var d=s(j,2);l(d,{title:"Using a task file instead of a task object",local:"using-a-task-file-instead-of-a-task-object",headingTag:"h2"});var h=s(d,4);a(h,{code:"JTIzJTIwdGFza3MlMkZlY2hvX2V2YWwucHklMjAlMjAoY29udGFpbnMlMjB0aGUlMjAlNDB0YXNrJTIwZGVmaW5pdGlvbiUyMGFib3ZlKSUwQSUwQXJlc3VsdCUyMCUzRCUyMGhhcm5lc3MucnVuX2Zyb21fY29uZmlnKEV2YWxDb25maWcoJTBBJTIwJTIwJTIwJTIwaGFybmVzc19uYW1lJTNEJTIySW5zcGVjdEFJSGFybmVzcyUyMiUyQyUwQSUyMCUyMCUyMCUyMGhhcm5lc3NfdmVyc2lvbiUzRGluc3BlY3RfYWkuX192ZXJzaW9uX18lMkMlMEElMjAlMjAlMjAlMjBsaWJyYXJ5X3ZlcnNpb25zJTNEJTdCJTIyb3BlbmVudiUyMiUzQSUyMG9wZW5lbnYuX192ZXJzaW9uX18lN0QlMkMlMEElMjAlMjAlMjAlMjBkYXRhc2V0JTNEJTIydGFza3MlMkZlY2hvX2V2YWwucHklNDBvcGVuZW52X2VjaG9fZXZhbCUyMiUyQyUwQSUyMCUyMCUyMCUyMGV2YWxfcGFyYW1ldGVycyUzRCU3QiUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMm1vZGVsJTIyJTNBJTIwJTIyb3BlbmFpJTJGZ3B0LTUtbWluaSUyMiUyQyUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMnRhc2slMjIlM0ElMjAlMjJ0YXNrcyUyRmVjaG9fZXZhbC5weSU0MG9wZW5lbnZfZWNob19ldmFsJTIyJTJDJTBBJTIwJTIwJTIwJTIwJTdEJTJDJTBBKSk=",highlighted:`<span class="hljs-comment"># tasks/echo_eval.py  (contains the @task definition above)</span>

result = harness.run_from_config(EvalConfig(
    harness_name=<span class="hljs-string">&quot;InspectAIHarness&quot;</span>,
    harness_version=inspect_ai.__version__,
    library_versions={<span class="hljs-string">&quot;openenv&quot;</span>: openenv.__version__},
    dataset=<span class="hljs-string">&quot;tasks/echo_eval.py@openenv_echo_eval&quot;</span>,
    eval_parameters={
        <span class="hljs-string">&quot;model&quot;</span>: <span class="hljs-string">&quot;openai/gpt-5-mini&quot;</span>,
        <span class="hljs-string">&quot;task&quot;</span>: <span class="hljs-string">&quot;tasks/echo_eval.py@openenv_echo_eval&quot;</span>,
    },
))`,lang:"python",wrap:!1});var U=s(h,2);l(U,{title:"Adapting to your own environment and task",local:"adapting-to-your-own-environment-and-task",headingTag:"h2"});var I=s(U,8);a(I,{code:"aW1wb3J0JTIwYXN5bmNpbyUwQSUwQWZyb20lMjBpbnNwZWN0X2FpLnNvbHZlciUyMGltcG9ydCUyMEdlbmVyYXRlJTJDJTIwVGFza1N0YXRlJTJDJTIwc29sdmVyJTBBZnJvbSUyMG9wZW5lbnYuY29yZSUyMGltcG9ydCUyME1DUFRvb2xDbGllbnQlMEElMEFfZW52X3NlbSUyMCUzRCUyMGFzeW5jaW8uU2VtYXBob3JlKDEpJTIwJTIwJTIzJTIwcmFpc2UlMjBpZiUyMHlvdXIlMjBTcGFjZSUyMHN1cHBvcnRzJTIwbW9yZSUyMHNlc3Npb25zJTBBJTBBJTBBJTQwc29sdmVyJTBBZGVmJTIwbXlfZW52X3NvbHZlcihiYXNlX3VybCUzQSUyMHN0ciklM0ElMEElMjAlMjAlMjAlMjBhc3luYyUyMGRlZiUyMHNvbHZlKHN0YXRlJTNBJTIwVGFza1N0YXRlJTJDJTIwZ2VuZXJhdGUlM0ElMjBHZW5lcmF0ZSklMjAtJTNFJTIwVGFza1N0YXRlJTNBJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwc3RhdGUlMjAlM0QlMjBhd2FpdCUyMGdlbmVyYXRlKHN0YXRlKSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMG1vZGVsX291dHB1dCUyMCUzRCUyMHN0YXRlLm91dHB1dC5jb21wbGV0aW9uLnN0cmlwKCklMEElMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjBhc3luYyUyMHdpdGglMjBfZW52X3NlbSUzQSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMGVudiUyMCUzRCUyME1DUFRvb2xDbGllbnQoYmFzZV91cmwlM0RiYXNlX3VybCklMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjB0cnklM0ElMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjBhd2FpdCUyMGVudi5yZXNldCgpJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwcmVzdWx0JTIwJTNEJTIwYXdhaXQlMjBlbnYuY2FsbF90b29sKCUyMnlvdXJfdG9vbF9uYW1lJTIyJTJDJTIwbWVzc2FnZSUzRG1vZGVsX291dHB1dCklMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjBzdGF0ZS5tZXRhZGF0YSU1QiUyMmVudl9yZXN1bHQlMjIlNUQlMjAlM0QlMjByZXN1bHQlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjBmaW5hbGx5JTNBJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwYXdhaXQlMjBlbnYuY2xvc2UoKSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMHJldHVybiUyMHN0YXRlJTBBJTBBJTIwJTIwJTIwJTIwcmV0dXJuJTIwc29sdmU=",highlighted:`<span class="hljs-keyword">import</span> asyncio

<span class="hljs-keyword">from</span> inspect_ai.solver <span class="hljs-keyword">import</span> Generate, TaskState, solver
<span class="hljs-keyword">from</span> openenv.core <span class="hljs-keyword">import</span> MCPToolClient

_env_sem = asyncio.Semaphore(<span class="hljs-number">1</span>)  <span class="hljs-comment"># raise if your Space supports more sessions</span>


<span class="hljs-meta">@solver</span>
<span class="hljs-keyword">def</span> <span class="hljs-title function_">my_env_solver</span>(<span class="hljs-params">base_url: <span class="hljs-built_in">str</span></span>):
    <span class="hljs-keyword">async</span> <span class="hljs-keyword">def</span> <span class="hljs-title function_">solve</span>(<span class="hljs-params">state: TaskState, generate: Generate</span>) -&gt; TaskState:
        state = <span class="hljs-keyword">await</span> generate(state)
        model_output = state.output.completion.strip()

        <span class="hljs-keyword">async</span> <span class="hljs-keyword">with</span> _env_sem:
            env = MCPToolClient(base_url=base_url)
            <span class="hljs-keyword">try</span>:
                <span class="hljs-keyword">await</span> env.reset()
                result = <span class="hljs-keyword">await</span> env.call_tool(<span class="hljs-string">&quot;your_tool_name&quot;</span>, message=model_output)
                state.metadata[<span class="hljs-string">&quot;env_result&quot;</span>] = result
            <span class="hljs-keyword">finally</span>:
                <span class="hljs-keyword">await</span> env.close()
        <span class="hljs-keyword">return</span> state

    <span class="hljs-keyword">return</span> solve`,lang:"python",wrap:!1});var u=s(I,2);l(u,{title:"Next steps",local:"next-steps",headingTag:"h2"});var Z=s(u,4);G(Z,{source:"https://github.com/openenv/OpenEnv/blob/main/docs/source/tutorials/evaluation-inspect.md"}),Q(2),C(B,e),f()}export{H as component};
