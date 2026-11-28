import"../chunks/DsnmJJEf.js";import{i as F,h as q,C as z,H as a,a as l,E as Y,s as x}from"../chunks/Dz7d4_v9.js";import{p as H,o as L,s,f as O,a as W,b as D,c as k,n as K}from"../chunks/BSuxAqoA.js";const P='{"title":"Collect Rollouts for SFT","local":"collect-rollouts-for-sft","sections":[{"title":"Why use an environment to collect training data","local":"why-use-an-environment-to-collect-training-data","sections":[],"depth":2},{"title":"What you’ll use","local":"what-youll-use","sections":[],"depth":2},{"title":"1. Install dependencies","local":"1-install-dependencies","sections":[],"depth":2},{"title":"2. Set your credentials","local":"2-set-your-credentials","sections":[],"depth":2},{"title":"3. Define the system prompt","local":"3-define-the-system-prompt","sections":[],"depth":2},{"title":"4. Configure data collection","local":"4-configure-data-collection","sections":[],"depth":2},{"title":"5. Collect rollouts with openenv collect","local":"5-collect-rollouts-with-openenv-collect","sections":[],"depth":2},{"title":"6. Filter the dataset","local":"6-filter-the-dataset","sections":[],"depth":2},{"title":"7. Inspect the dataset before training","local":"7-inspect-the-dataset-before-training","sections":[],"depth":2},{"title":"8. Measure token lengths","local":"8-measure-token-lengths","sections":[],"depth":2},{"title":"9. Fine-tune with SFTTrainer","local":"9-fine-tune-with-sfttrainer","sections":[],"depth":2},{"title":"10. Evaluate: before vs after","local":"10-evaluate-before-vs-after","sections":[],"depth":2},{"title":"11. Where to go next: GRPO","local":"11-where-to-go-next-grpo","sections":[],"depth":2}],"depth":1}';var $=k('<meta name="hf:doc:metadata"/>'),ss=k(`<p></p> <!> <!> <p><a href="https://colab.research.google.com/github/openenv/OpenEnv/blob/main/examples/sft_warmup.ipynb" rel="nofollow"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a></p> <p>OpenEnv environments are not only useful for RL training — they are also a natural tool for <strong>collecting
rollouts that become supervised training data</strong>. The environment handles episode management, automatic scoring,
and reproducibility, so you get a reward-labeled dataset without writing any of that infrastructure yourself.</p> <p>This tutorial shows the full pipeline:</p> <ol><li>Run a strong teacher model inside an OpenEnv environment to collect rollouts.</li> <li>Use the environment’s reward signal to filter out incorrect examples automatically.</li> <li>Train a smaller student model on the filtered rollouts with TRL’s <code>SFTTrainer</code>.</li></ol> <p>As a concrete application, the resulting checkpoint is used as a warm-start for GRPO: once the student
reliably produces valid tool calls, GRPO’s <code>reward_std</code> is non-zero from the first batch and the reward
curve climbs immediately.</p> <!> <p>Building a supervised dataset usually means writing a custom collection loop, a scorer, and episode
bookkeeping. An OpenEnv environment gives you all three out of the box:</p> <ul><li><strong>Automatic scoring</strong> — every <code>step()</code> returns a reward. Filter by <code>reward == 1.0</code> and you have a
clean, correct dataset with no manual labelling.</li> <li><strong>Reproducible episodes</strong> — <code>reset(seed=42, size=N)</code> produces the same sequence of problems every
run. Anyone can regenerate the exact dataset.</li> <li><strong>Configurable difficulty</strong> — adjust <code>DATASET_CONFIG</code> to control problem complexity without changing
any collection code.</li> <li><strong>Portable across environments</strong> — the same collect → filter → train pipeline works for any OpenEnv
environment. Swap the env and the tool definition; everything else stays the same.</li></ul> <!> <table><thead><tr><th></th><th></th></tr></thead><tbody><tr><td><strong>Student model</strong></td><td><a href="https://huggingface.co/Qwen/Qwen3-1.7B" rel="nofollow"><code>Qwen/Qwen3-1.7B</code></a></td></tr><tr><td><strong>Teacher model</strong></td><td><code>gpt-5-mini</code> via the OpenAI API</td></tr><tr><td><strong>Environment</strong></td><td><a href="https://github.com/openenv/OpenEnv/tree/main/envs/reasoning_gym_env" rel="nofollow"><code>reasoning_gym_env</code></a> / <code>chain_sum</code></td></tr><tr><td><strong>SFT trainer</strong></td><td><a href="https://huggingface.co/docs/trl/main/en/sft_trainer" rel="nofollow">TRL <code>SFTTrainer</code></a></td></tr><tr><td><strong>Next step</strong></td><td><a href="end-to-end-walkthrough">End-to-end walkthrough with GRPO</a></td></tr></tbody></table> <hr/> <!> <!> <hr/> <!> <!> <p>You’ll also need a Hugging Face login to download the base model and push both the collected dataset
and the fine-tuned checkpoint:</p> <!> <!> <hr/> <!> <p>Use the same prompt as the <a href="end-to-end-walkthrough">GRPO tutorial</a> so the SFT-trained model is a drop-in replacement when you continue with GRPO.</p> <!> <hr/> <!> <p><code>DATASET_CONFIG</code> controls the difficulty of the <code>chain_sum</code> problems the environment generates: <code>min_terms</code>/<code>max_terms</code> set how many integers are added together, and <code>min_digits</code>/<code>max_digits</code> set
how many digits each integer has. At these settings each problem is a sum of 2–3 two-digit numbers
— easy enough for <code>gpt-5-mini</code> to answer correctly ~90% of the time, which gives a clean training
signal after filtering.</p> <p><code>N_EPISODES</code> is the number of problems to collect. 300 is enough to get ~270 correct examples after
filtering, which is sufficient for format compliance training.</p> <!> <hr/> <!> <p><code>openenv collect</code> runs the teacher model inside the environment and records every episode — the
environment’s <code>step()</code> reward is written alongside the messages, so filtering by correctness requires
no additional scoring code. By default it drops rollouts with a negative reward (<code>--keep-losses</code> keeps
them). Reasoning Gym scores a wrong answer <code>0.0</code>, so wrong answers are kept here and filtered in section 6.</p> <!> <p>To use a self-hosted teacher instead of a hosted provider, point <code>--llm-endpoint</code> at any
OpenAI-compatible server (vLLM, TGI, Ollama) and pass the model id it serves:</p> <!> <p>The teacher drives the environment through tool calls, so the server must have tool calling
enabled (for vLLM: <code>--enable-auto-tool-choice --tool-call-parser &lt;parser for your model&gt;</code>).</p> <p>The command prints a live progress summary and pushes the collected episodes to the Hub as <code>&#123;YOUR_HF_USERNAME&#125;/chain-sum-rollouts</code>. Pull them back to start filtering:</p> <!> <p>The <code>messages</code> field stores the full conversation in standard OpenAI format (assistant messages have
a <code>tool_calls</code> list). Convert to Qwen3’s <code>&lt;tool_call&gt;</code> text format before training — GRPOTrainer
produces this same format during RL, so the SFT checkpoint becomes a direct drop-in:</p> <!> <hr/> <!> <p>Keep only episodes where the teacher answered correctly. <code>openenv collect</code> already dropped negative
rewards, and this filter also removes the wrong answers, which score <code>0.0</code>. The environment’s reward
signal does the labelling — no manual annotation needed.</p> <!> <p><code>gpt-5-mini</code> typically scores above 90% on <code>chain_sum</code> at this difficulty, so you should end up with
~270 examples from 300 rollouts.</p> <hr/> <!> <p>Always look at your data before training. Automated collection can introduce unexpected patterns that the
student model will learn to imitate.</p> <!> <p>Things to check:</p> <ul><li>Does every response contain a valid <code>&lt;tool_call&gt;</code> block?</li> <li>Are the answers integers with no extra text?</li> <li>Is there any reasoning in the assistant message that you don’t want the student to learn?
(For example: an internal monologue, disclaimers, or repeated phrasing that the teacher leaked
from its own system prompt.)</li></ul> <hr/> <!> <p>Set <code>max_length</code> in <code>SFTConfig</code> to cover nearly all examples without wasting GPU memory on padding.
The 99th percentile is a good target: you truncate fewer than 1% of examples while keeping batches tight.</p> <!> <hr/> <!> <p><code>assistant_only_loss=True</code> in <code>SFTConfig</code> masks the prompt tokens so the loss is computed only on the
assistant response — the <code>&lt;tool_call&gt;</code> block. This is more efficient than full-sequence training and avoids
accidentally reinforcing the system prompt wording.</p> <!> <blockquote class="note"><p>Training ~270 examples for 3 epochs takes around 5 minutes on a single A100 (40 GB). The goal is format
compliance, not task mastery — a handful of epochs is enough. Mastery comes from GRPO.</p></blockquote> <hr/> <!> <p>Run both the base model and the SFT checkpoint on a held-out set and compare. The key metric for a
warm-up evaluation is <strong>format compliance</strong> — how often the model uses <code>&lt;tool_call&gt;</code> correctly — as
well as overall accuracy.</p> <!> <p>A successful warm-up looks like this:</p> <table><thead><tr><th>Metric</th><th>Base model</th><th>After SFT</th><th>Delta</th></tr></thead><tbody><tr><td>Format compliance</td><td>~0%</td><td>~68%</td><td>+68 pp</td></tr><tr><td>Accuracy</td><td>~4%</td><td>~68%</td><td>+64 pp</td></tr></tbody></table> <p>Format compliance should jump sharply from near-zero — that’s the primary goal. <code>Qwen3-1.7B</code> produces
essentially no valid <code>&lt;tool_call&gt;</code> blocks out of the box. After SFT on ~270 examples, the model reliably
uses the format, and accuracy rises in lockstep because correct format is a prerequisite for the
environment’s scorer to award any credit.</p> <hr/> <!> <p>The SFT checkpoint is ready to use as the starting model for GRPO. In the <a href="end-to-end-walkthrough">end-to-end walkthrough</a>,
change one line in section 7:</p> <!> <p>With most completions already in the right format, GRPO’s <code>reward_std</code> will be non-zero from the very first
batch and the reward curve will climb immediately — no cold-start stall.</p> <p><strong>Other directions:</strong></p> <ul><li><strong>Harder tasks.</strong> Increase <code>max_terms</code> or <code>max_digits</code> in <code>DATASET_CONFIG</code> and collect a new SFT set.
Once the student handles easier examples reliably, a harder GRPO phase can push further.</li> <li><strong>Different environments.</strong> The same pipeline — teacher collects → filter → SFT → GRPO — applies to
any OpenEnv environment. Swap <code>reasoning_gym_env</code> and the <code>answer</code> tool definition for your env’s
tool surface.</li> <li><strong>Larger teacher.</strong> A stronger model, such as <code>claude-opus-5-5</code> (<code>--provider anthropic</code>) or <code>Qwen/Qwen3-32B</code> served behind <code>--llm-endpoint</code>, yields higher-quality examples on tasks where <code>gpt-5-mini</code> struggles.</li></ul> <!> <p></p>`,1);function es(S,X){H(X,!1),L(()=>{new URLSearchParams(window.location.search).get("fw")}),F();var n=ss();q("1p1vfr3",V=>{var v=$();x(v,"content",P),W(V,v)});var e=s(O(n),2);z(e,{containerStyle:"float: inline-end; margin-inline-start: 10px; display: inline-flex; position: relative; z-index: 10;"});var t=s(e,2);a(t,{title:"Collect Rollouts for SFT",local:"collect-rollouts-for-sft",headingTag:"h1"});var o=s(t,12);a(o,{title:"Why use an environment to collect training data",local:"why-use-an-environment-to-collect-training-data",headingTag:"h2"});var M=s(o,6);a(M,{title:"What you’ll use",local:"what-youll-use",headingTag:"h2"});var r=s(M,6);a(r,{title:"1. Install dependencies",local:"1-install-dependencies",headingTag:"h2"});var c=s(r,2);l(c,{code:"IXBpcCUyMGluc3RhbGwlMjAtcSUyMG9wZW5haSUyMHRybCUwQSFwaXAlMjBpbnN0YWxsJTIwLXElMjBvcGVuZW52JTBBIXBpcCUyMGluc3RhbGwlMjAtcSUyMC0tbm8tZGVwcyUyMGdpdCUyQmh0dHBzJTNBJTJGJTJGaHVnZ2luZ2ZhY2UuY28lMkZzcGFjZXMlMkZzZXJnaW9wYW5pZWdvJTJGcmVhc29uaW5nX2d5bSUwQSFwaXAlMjBpbnN0YWxsJTIwLVVxJTIwJTIydHJhbnNmb3JtZXJzJTNFJTNENS4zLjAlMjI=",highlighted:`!pip install -q openai trl
!pip install -q openenv
!pip install -q --no-deps git+https://huggingface.co/spaces/sergiopaniego/reasoning_gym
!pip install -Uq <span class="hljs-string">&quot;transformers&gt;=5.3.0&quot;</span>`,lang:"python",wrap:!1});var p=s(c,4);a(p,{title:"2. Set your credentials",local:"2-set-your-credentials",headingTag:"h2"});var i=s(p,2);l(i,{code:"aW1wb3J0JTIwZ2V0cGFzcyUyQyUyMG9zJTBBJTBBaWYlMjAlMjJPUEVOQUlfQVBJX0tFWSUyMiUyMG5vdCUyMGluJTIwb3MuZW52aXJvbiUzQSUwQSUyMCUyMCUyMCUyMG9zLmVudmlyb24lNUIlMjJPUEVOQUlfQVBJX0tFWSUyMiU1RCUyMCUzRCUyMGdldHBhc3MuZ2V0cGFzcyglMjJPcGVuQUklMjBBUEklMjBrZXklM0ElMjAlMjIp",highlighted:`<span class="hljs-keyword">import</span> getpass, os

<span class="hljs-keyword">if</span> <span class="hljs-string">&quot;OPENAI_API_KEY&quot;</span> <span class="hljs-keyword">not</span> <span class="hljs-keyword">in</span> os.environ:
    os.environ[<span class="hljs-string">&quot;OPENAI_API_KEY&quot;</span>] = getpass.getpass(<span class="hljs-string">&quot;OpenAI API key: &quot;</span>)`,lang:"python",wrap:!1});var y=s(i,4);l(y,{code:"ZnJvbSUyMGh1Z2dpbmdmYWNlX2h1YiUyMGltcG9ydCUyMG5vdGVib29rX2xvZ2luJTBBJTBBbm90ZWJvb2tfbG9naW4oKQ==",highlighted:`<span class="hljs-keyword">from</span> huggingface_hub <span class="hljs-keyword">import</span> notebook_login

notebook_login()`,lang:"python",wrap:!1});var d=s(y,2);l(d,{code:"WU9VUl9IRl9VU0VSTkFNRSUyMCUzRCUyMCUyMnlvdXItdXNlcm5hbWUlMjIlMjAlMjAlMjMlMjByZXBsYWNlJTIwd2l0aCUyMHlvdXIlMjBIdWdnaW5nJTIwRmFjZSUyMHVzZXJuYW1lJTBBYXNzZXJ0JTIwWU9VUl9IRl9VU0VSTkFNRSUyMCElM0QlMjAlMjJ5b3VyLXVzZXJuYW1lJTIyJTJDJTIwJTIyUmVwbGFjZSUyMFlPVVJfSEZfVVNFUk5BTUUlMjB3aXRoJTIweW91ciUyMEh1Z2dpbmclMjBGYWNlJTIwdXNlcm5hbWUlMjI=",highlighted:`YOUR_HF_USERNAME = <span class="hljs-string">&quot;your-username&quot;</span>  <span class="hljs-comment"># replace with your Hugging Face username</span>
<span class="hljs-keyword">assert</span> YOUR_HF_USERNAME != <span class="hljs-string">&quot;your-username&quot;</span>, <span class="hljs-string">&quot;Replace YOUR_HF_USERNAME with your Hugging Face username&quot;</span>`,lang:"python",wrap:!1});var h=s(d,4);a(h,{title:"3. Define the system prompt",local:"3-define-the-system-prompt",headingTag:"h2"});var j=s(h,4);l(j,{code:"U1lTVEVNX1BST01QVCUyMCUzRCUyMCUyMiUyMiUyMllvdSUyMGFyZSUyMGElMjBjYXJlZnVsJTIwYXJpdGhtZXRpYyUyMGFzc2lzdGFudC4lMEElMEFZb3UlMjB3aWxsJTIwYmUlMjBnaXZlbiUyMGElMjBjaGFpbiUyMG9mJTIwaW50ZWdlciUyMGFkZGl0aW9ucy4lMjBDb21wdXRlJTIwdGhlJTIwcmVzdWx0JTIwYW5kJTIwc3VibWl0JTIwaXQlMjBhcyUyMGElMjBzaW5nbGUlMjBudW1iZXIuJTBBJTBBUnVsZXMlM0ElMEExLiUyMFJlYWQlMjB0aGUlMjBxdWVzdGlvbiUyMGNhcmVmdWxseS4lMEEyLiUyMFVzZSUyMHRoZSUyMHRvb2wlMjAlNjBhbnN3ZXIlNjAlMjBleGFjdGx5JTIwb25jZSUyMHdpdGglMjB5b3VyJTIwZmluYWwlMjBudW1iZXIuJTBBMy4lMjBUaGUlMjBhbnN3ZXIlMjBtdXN0JTIwYmUlMjBhJTIwc2luZ2xlJTIwaW50ZWdlciUyMHdpdGglMjBubyUyMHVuaXRzJTIwb3IlMjBleHBsYW5hdGlvbi4lMEElMjIlMjIlMjI=",highlighted:`SYSTEM_PROMPT = <span class="hljs-string">&quot;&quot;&quot;You are a careful arithmetic assistant.

You will be given a chain of integer additions. Compute the result and submit it as a single number.

Rules:
1. Read the question carefully.
2. Use the tool \`answer\` exactly once with your final number.
3. The answer must be a single integer with no units or explanation.
&quot;&quot;&quot;</span>`,lang:"python",wrap:!1});var U=s(j,4);a(U,{title:"4. Configure data collection",local:"4-configure-data-collection",headingTag:"h2"});var J=s(U,6);l(J,{code:"REFUQVNFVF9DT05GSUclMjAlM0QlMjAlN0IlMEElMjAlMjAlMjAlMjAlMjJtaW5fdGVybXMlMjIlM0ElMjAyJTJDJTBBJTIwJTIwJTIwJTIwJTIybWF4X3Rlcm1zJTIyJTNBJTIwMyUyQyUwQSUyMCUyMCUyMCUyMCUyMm1pbl9kaWdpdHMlMjIlM0ElMjAyJTJDJTBBJTIwJTIwJTIwJTIwJTIybWF4X2RpZ2l0cyUyMiUzQSUyMDIlMkMlMEElN0QlMEElMEFOX0VQSVNPREVTJTIwJTNEJTIwMzAw",highlighted:`DATASET_CONFIG = {
    <span class="hljs-string">&quot;min_terms&quot;</span>: <span class="hljs-number">2</span>,
    <span class="hljs-string">&quot;max_terms&quot;</span>: <span class="hljs-number">3</span>,
    <span class="hljs-string">&quot;min_digits&quot;</span>: <span class="hljs-number">2</span>,
    <span class="hljs-string">&quot;max_digits&quot;</span>: <span class="hljs-number">2</span>,
}

N_EPISODES = <span class="hljs-number">300</span>`,lang:"python",wrap:!1});var T=s(J,4);a(T,{title:"5. Collect rollouts with openenv collect",local:"5-collect-rollouts-with-openenv-collect",headingTag:"h2"});var w=s(T,4);l(w,{code:"aW1wb3J0JTIwanNvbiUyQyUyMHNobGV4JTBBJTBBZGF0YXNldF9jb25maWdfYXJnJTIwJTNEJTIwc2hsZXgucXVvdGUoanNvbi5kdW1wcyhEQVRBU0VUX0NPTkZJRykpJTBBc3lzdGVtX3Byb21wdF9hcmclMjAlM0QlMjBzaGxleC5xdW90ZShTWVNURU1fUFJPTVBUKSUwQWh1Yl9yZXBvX2FyZyUyMCUzRCUyMHNobGV4LnF1b3RlKGYlMjIlN0JZT1VSX0hGX1VTRVJOQU1FJTdEJTJGY2hhaW4tc3VtLXJvbGxvdXRzJTIyKSUwQSUwQSFvcGVuZW52JTIwY29sbGVjdCUyMHJlYXNvbmluZ19neW0lM0FjaGFpbl9zdW0lMjAlNUMlMEElMjAlMjAtLWJhc2UtdXJsJTIwaHR0cHMlM0ElMkYlMkZzZXJnaW9wYW5pZWdvLXJlYXNvbmluZy1neW0uaGYuc3BhY2UlMjAlNUMlMEElMjAlMjAtLXByb3ZpZGVyJTIwb3BlbmFpJTIwJTVDJTBBJTIwJTIwLS1tb2RlbCUyMGdwdC01LW1pbmklMjAlNUMlMEElMjAlMjAtLW51bS1lcGlzb2RlcyUyMCU3Qk5fRVBJU09ERVMlN0QlMjAlNUMlMEElMjAlMjAtLW1heC10b2tlbnMlMjAxMDI0JTIwJTVDJTBBJTIwJTIwLS1kYXRhc2V0LWNvbmZpZyUyMCU3QmRhdGFzZXRfY29uZmlnX2FyZyU3RCUyMCU1QyUwQSUyMCUyMC0tc3lzdGVtLXByb21wdCUyMCU3QnN5c3RlbV9wcm9tcHRfYXJnJTdEJTIwJTVDJTBBJTIwJTIwLS1wdXNoLXRvLWh1YiUyMCU3Qmh1Yl9yZXBvX2FyZyU3RCUyMCU1QyUwQSUyMCUyMC0tb3V0cHV0LWRpciUyMC4lMkZyb2xsb3V0cw==",highlighted:`<span class="hljs-keyword">import</span> json, shlex

dataset_config_arg = shlex.quote(json.dumps(DATASET_CONFIG))
system_prompt_arg = shlex.quote(SYSTEM_PROMPT)
hub_repo_arg = shlex.quote(<span class="hljs-string">f&quot;<span class="hljs-subst">{YOUR_HF_USERNAME}</span>/chain-sum-rollouts&quot;</span>)

!openenv collect reasoning_gym:chain_sum \\
  --base-url https://sergiopaniego-reasoning-gym.hf.space \\
  --provider openai \\
  --model gpt-<span class="hljs-number">5</span>-mini \\
  --num-episodes {N_EPISODES} \\
  --<span class="hljs-built_in">max</span>-tokens <span class="hljs-number">1024</span> \\
  --dataset-config {dataset_config_arg} \\
  --system-prompt {system_prompt_arg} \\
  --push-to-hub {hub_repo_arg} \\
  --output-<span class="hljs-built_in">dir</span> ./rollouts`,lang:"python",wrap:!1});var u=s(w,4);l(u,{code:"b3BlbmVudiUyMGNvbGxlY3QlMjByZWFzb25pbmdfZ3ltJTNBY2hhaW5fc3VtJTIwJTVDJTBBJTIwJTIwLS1iYXNlLXVybCUyMGh0dHBzJTNBJTJGJTJGc2VyZ2lvcGFuaWVnby1yZWFzb25pbmctZ3ltLmhmLnNwYWNlJTIwJTVDJTBBJTIwJTIwLS1sbG0tZW5kcG9pbnQlMjBodHRwJTNBJTJGJTJGbG9jYWxob3N0JTNBODAwMCUyMCU1QyUwQSUyMCUyMC0tbW9kZWwlMjBRd2VuJTJGUXdlbjMtOEIlMjAlNUMlMEElMjAlMjAtLW51bS1lcGlzb2RlcyUyMDMwMCUyMCU1QyUwQSUyMCUyMC0tb3V0cHV0LWRpciUyMC4lMkZyb2xsb3V0cw==",highlighted:`openenv collect reasoning_gym:chain_sum \\
  --base-url https://sergiopaniego-reasoning-gym.hf.space \\
  --llm-endpoint http://localhost:8000 \\
  --model Qwen/Qwen3-8B \\
  --num-episodes 300 \\
  --output-dir ./rollouts`,lang:"bash",wrap:!1});var m=s(u,6);l(m,{code:"ZnJvbSUyMGRhdGFzZXRzJTIwaW1wb3J0JTIwbG9hZF9kYXRhc2V0JTBBJTBBZHMlMjAlM0QlMjBsb2FkX2RhdGFzZXQoZiUyMiU3QllPVVJfSEZfVVNFUk5BTUUlN0QlMkZjaGFpbi1zdW0tcm9sbG91dHMlMjIlMkMlMjBzcGxpdCUzRCUyMnRyYWluJTIyKSUwQXJhd19yb2xsb3V0cyUyMCUzRCUyMGxpc3QoZHMpJTBBcHJpbnQoZiUyMkNvbGxlY3RlZCUyMCU3QmxlbihyYXdfcm9sbG91dHMpJTdEJTIwZXBpc29kZXMlMjIp",highlighted:`<span class="hljs-keyword">from</span> datasets <span class="hljs-keyword">import</span> load_dataset

ds = load_dataset(<span class="hljs-string">f&quot;<span class="hljs-subst">{YOUR_HF_USERNAME}</span>/chain-sum-rollouts&quot;</span>, split=<span class="hljs-string">&quot;train&quot;</span>)
raw_rollouts = <span class="hljs-built_in">list</span>(ds)
<span class="hljs-built_in">print</span>(<span class="hljs-string">f&quot;Collected <span class="hljs-subst">{<span class="hljs-built_in">len</span>(raw_rollouts)}</span> episodes&quot;</span>)`,lang:"python",wrap:!1});var b=s(m,4);l(b,{code:"ZGVmJTIwdG9fcXdlbjNfbWVzc2FnZXMocmVjb3JkKSUzQSUwQSUyMCUyMCUyMCUyMGNvbnZlcnRlZCUyMCUzRCUyMCU1QiU1RCUwQSUyMCUyMCUyMCUyMGZvciUyMG1zZyUyMGluJTIwcmVjb3JkJTVCJTIybWVzc2FnZXMlMjIlNUQlM0ElMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjBpZiUyMG1zZyU1QiUyMnJvbGUlMjIlNUQlMjAlM0QlM0QlMjAlMjJ0b29sJTIyJTNBJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwY29udGludWUlMjAlMjAlMjMlMjBzdHJpcCUyMGVudmlyb25tZW50JTIwcmVzcG9uc2VzJTNCJTIwU0ZUJTIwb25seSUyMG5lZWRzJTIwdGhlJTIwYXNzaXN0YW50JTIwdHVybiUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMGlmJTIwbXNnJTVCJTIycm9sZSUyMiU1RCUyMCUzRCUzRCUyMCUyMmFzc2lzdGFudCUyMiUyMGFuZCUyMG1zZy5nZXQoJTIydG9vbF9jYWxscyUyMiklM0ElMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjB0YyUyMCUzRCUyMG1zZyU1QiUyMnRvb2xfY2FsbHMlMjIlNUQlNUIwJTVEJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwYXJncyUyMCUzRCUyMGpzb24ubG9hZHModGMlNUIlMjJmdW5jdGlvbiUyMiU1RCU1QiUyMmFyZ3VtZW50cyUyMiU1RCklMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjBhbnN3ZXJfc3RyJTIwJTNEJTIwYXJncy5nZXQoJTIyYW5zd2VyJTIyJTJDJTIwJTIyJTIyKSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMHRvb2xfY2FsbF90ZXh0JTIwJTNEJTIwKCUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMiUzQ3Rvb2xfY2FsbCUzRSU1Q24lMjIlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMkIlMjBqc29uLmR1bXBzKCU3QiUyMm5hbWUlMjIlM0ElMjAlMjJhbnN3ZXIlMjIlMkMlMjAlMjJhcmd1bWVudHMlMjIlM0ElMjAlN0IlMjJhbnN3ZXIlMjIlM0ElMjBhbnN3ZXJfc3RyJTdEJTdEKSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyQiUyMCUyMiU1Q24lM0MlMkZ0b29sX2NhbGwlM0UlMjIlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjApJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwY29udmVydGVkLmFwcGVuZCglN0IlMjJyb2xlJTIyJTNBJTIwJTIyYXNzaXN0YW50JTIyJTJDJTIwJTIyY29udGVudCUyMiUzQSUyMHRvb2xfY2FsbF90ZXh0JTdEKSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMGVsc2UlM0ElMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjBjb252ZXJ0ZWQuYXBwZW5kKG1zZyklMEElMjAlMjAlMjAlMjByZXR1cm4lMjAlN0IlMjJtZXNzYWdlcyUyMiUzQSUyMGNvbnZlcnRlZCUyQyUyMCUyMnJld2FyZCUyMiUzQSUyMHJlY29yZCU1QiUyMnJld2FyZCUyMiU1RCU3RCUwQSUwQXJvbGxvdXRzJTIwJTNEJTIwJTVCdG9fcXdlbjNfbWVzc2FnZXMociklMjBmb3IlMjByJTIwaW4lMjByYXdfcm9sbG91dHMlNUQ=",highlighted:`<span class="hljs-keyword">def</span> <span class="hljs-title function_">to_qwen3_messages</span>(<span class="hljs-params">record</span>):
    converted = []
    <span class="hljs-keyword">for</span> msg <span class="hljs-keyword">in</span> record[<span class="hljs-string">&quot;messages&quot;</span>]:
        <span class="hljs-keyword">if</span> msg[<span class="hljs-string">&quot;role&quot;</span>] == <span class="hljs-string">&quot;tool&quot;</span>:
            <span class="hljs-keyword">continue</span>  <span class="hljs-comment"># strip environment responses; SFT only needs the assistant turn</span>
        <span class="hljs-keyword">if</span> msg[<span class="hljs-string">&quot;role&quot;</span>] == <span class="hljs-string">&quot;assistant&quot;</span> <span class="hljs-keyword">and</span> msg.get(<span class="hljs-string">&quot;tool_calls&quot;</span>):
            tc = msg[<span class="hljs-string">&quot;tool_calls&quot;</span>][<span class="hljs-number">0</span>]
            args = json.loads(tc[<span class="hljs-string">&quot;function&quot;</span>][<span class="hljs-string">&quot;arguments&quot;</span>])
            answer_str = args.get(<span class="hljs-string">&quot;answer&quot;</span>, <span class="hljs-string">&quot;&quot;</span>)
            tool_call_text = (
                <span class="hljs-string">&quot;&lt;tool_call&gt;\\n&quot;</span>
                + json.dumps({<span class="hljs-string">&quot;name&quot;</span>: <span class="hljs-string">&quot;answer&quot;</span>, <span class="hljs-string">&quot;arguments&quot;</span>: {<span class="hljs-string">&quot;answer&quot;</span>: answer_str}})
                + <span class="hljs-string">&quot;\\n&lt;/tool_call&gt;&quot;</span>
            )
            converted.append({<span class="hljs-string">&quot;role&quot;</span>: <span class="hljs-string">&quot;assistant&quot;</span>, <span class="hljs-string">&quot;content&quot;</span>: tool_call_text})
        <span class="hljs-keyword">else</span>:
            converted.append(msg)
    <span class="hljs-keyword">return</span> {<span class="hljs-string">&quot;messages&quot;</span>: converted, <span class="hljs-string">&quot;reward&quot;</span>: record[<span class="hljs-string">&quot;reward&quot;</span>]}

rollouts = [to_qwen3_messages(r) <span class="hljs-keyword">for</span> r <span class="hljs-keyword">in</span> raw_rollouts]`,lang:"python",wrap:!1});var I=s(b,4);a(I,{title:"6. Filter the dataset",local:"6-filter-the-dataset",headingTag:"h2"});var g=s(I,4);l(g,{code:"Y29ycmVjdCUyMCUzRCUyMCU1QnIlMjBmb3IlMjByJTIwaW4lMjByb2xsb3V0cyUyMGlmJTIwciU1QiUyMnJld2FyZCUyMiU1RCUyMCUzRCUzRCUyMDEuMCU1RCUwQXByaW50KGYlMjJDb3JyZWN0JTNBJTIwJTdCbGVuKGNvcnJlY3QpJTdEJTIwJTJGJTIwJTdCbGVuKHJvbGxvdXRzKSU3RCUyMCglN0JsZW4oY29ycmVjdCklMkZsZW4ocm9sbG91dHMpJTNBLjElMjUlN0QpJTIyKQ==",highlighted:`correct = [r <span class="hljs-keyword">for</span> r <span class="hljs-keyword">in</span> rollouts <span class="hljs-keyword">if</span> r[<span class="hljs-string">&quot;reward&quot;</span>] == <span class="hljs-number">1.0</span>]
<span class="hljs-built_in">print</span>(<span class="hljs-string">f&quot;Correct: <span class="hljs-subst">{<span class="hljs-built_in">len</span>(correct)}</span> / <span class="hljs-subst">{<span class="hljs-built_in">len</span>(rollouts)}</span> (<span class="hljs-subst">{<span class="hljs-built_in">len</span>(correct)/<span class="hljs-built_in">len</span>(rollouts):<span class="hljs-number">.1</span>%}</span>)&quot;</span>)`,lang:"python",wrap:!1});var C=s(g,6);a(C,{title:"7. Inspect the dataset before training",local:"7-inspect-the-dataset-before-training",headingTag:"h2"});var A=s(C,4);l(A,{code:"aW1wb3J0JTIwcmFuZG9tJTBBJTBBZm9yJTIwcm93JTIwaW4lMjByYW5kb20uc2FtcGxlKGNvcnJlY3QlMkMlMjAzKSUzQSUwQSUyMCUyMCUyMCUyMHF1ZXN0aW9uJTIwJTNEJTIwcm93JTVCJTIybWVzc2FnZXMlMjIlNUQlNUIwJTVEJTVCJTIyY29udGVudCUyMiU1RCUwQSUyMCUyMCUyMCUyMHJlc3BvbnNlJTIwJTNEJTIwcm93JTVCJTIybWVzc2FnZXMlMjIlNUQlNUIxJTVEJTVCJTIyY29udGVudCUyMiU1RCUwQSUyMCUyMCUyMCUyMHByaW50KGYlMjJRJTNBJTIwJTdCcXVlc3Rpb24lN0QlMjIpJTBBJTIwJTIwJTIwJTIwcHJpbnQoZiUyMkElM0ElMjAlN0JyZXNwb25zZSU3RCUyMiklMEElMjAlMjAlMjAlMjBwcmludCgp",highlighted:`<span class="hljs-keyword">import</span> random

<span class="hljs-keyword">for</span> row <span class="hljs-keyword">in</span> random.sample(correct, <span class="hljs-number">3</span>):
    question = row[<span class="hljs-string">&quot;messages&quot;</span>][<span class="hljs-number">0</span>][<span class="hljs-string">&quot;content&quot;</span>]
    response = row[<span class="hljs-string">&quot;messages&quot;</span>][<span class="hljs-number">1</span>][<span class="hljs-string">&quot;content&quot;</span>]
    <span class="hljs-built_in">print</span>(<span class="hljs-string">f&quot;Q: <span class="hljs-subst">{question}</span>&quot;</span>)
    <span class="hljs-built_in">print</span>(<span class="hljs-string">f&quot;A: <span class="hljs-subst">{response}</span>&quot;</span>)
    <span class="hljs-built_in">print</span>()`,lang:"python",wrap:!1});var B=s(A,8);a(B,{title:"8. Measure token lengths",local:"8-measure-token-lengths",headingTag:"h2"});var Z=s(B,4);l(Z,{code:"aW1wb3J0JTIwbnVtcHklMjBhcyUyMG5wJTBBZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMEElMEF0b2tlbml6ZXIlMjAlM0QlMjBBdXRvVG9rZW5pemVyLmZyb21fcHJldHJhaW5lZCglMjJRd2VuJTJGUXdlbjMtMS43QiUyMiklMEElMEFsZW5ndGhzJTIwJTNEJTIwJTVCJTVEJTBBZm9yJTIwcm93JTIwaW4lMjBjb3JyZWN0JTNBJTBBJTIwJTIwJTIwJTIwdGV4dCUyMCUzRCUyMHRva2VuaXplci5hcHBseV9jaGF0X3RlbXBsYXRlKCUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMHJvdyU1QiUyMm1lc3NhZ2VzJTIyJTVEJTJDJTIwdG9rZW5pemUlM0RGYWxzZSUyQyUyMGFkZF9nZW5lcmF0aW9uX3Byb21wdCUzREZhbHNlJTBBJTIwJTIwJTIwJTIwKSUwQSUyMCUyMCUyMCUyMGlkcyUyMCUzRCUyMHRva2VuaXplci5lbmNvZGUodGV4dCklMEElMjAlMjAlMjAlMjBsZW5ndGhzLmFwcGVuZChsZW4oaWRzKSklMEElMEFsZW5ndGhzJTIwJTNEJTIwbnAuYXJyYXkobGVuZ3RocyklMEFNQVhfU0VRX0xFTiUyMCUzRCUyMGludChucC5wZXJjZW50aWxlKGxlbmd0aHMlMkMlMjA5OSkpJTIwJTJCJTIwMTYlMEElMEFwcmludCglMEElMjAlMjAlMjAlMjBmJTIycDUwJTNEJTdCbnAucGVyY2VudGlsZShsZW5ndGhzJTJDJTIwNTApJTNBLjBmJTdEJTIwJTIwJTIyJTBBJTIwJTIwJTIwJTIwZiUyMnA5NSUzRCU3Qm5wLnBlcmNlbnRpbGUobGVuZ3RocyUyQyUyMDk1KSUzQS4wZiU3RCUyMCUyMCUyMiUwQSUyMCUyMCUyMCUyMGYlMjJwOTklM0QlN0JucC5wZXJjZW50aWxlKGxlbmd0aHMlMkMlMjA5OSklM0EuMGYlN0QlMjAlMjAlMjIlMEElMjAlMjAlMjAlMjBmJTIybWF4JTNEJTdCbGVuZ3Rocy5tYXgoKSU3RCUyMiUwQSklMEFwcmludChmJTIyU2V0dGluZyUyME1BWF9TRVFfTEVOJTIwJTNEJTIwJTdCTUFYX1NFUV9MRU4lN0QlMjIp",highlighted:`<span class="hljs-keyword">import</span> numpy <span class="hljs-keyword">as</span> np
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;Qwen/Qwen3-1.7B&quot;</span>)

lengths = []
<span class="hljs-keyword">for</span> row <span class="hljs-keyword">in</span> correct:
    text = tokenizer.apply_chat_template(
        row[<span class="hljs-string">&quot;messages&quot;</span>], tokenize=<span class="hljs-literal">False</span>, add_generation_prompt=<span class="hljs-literal">False</span>
    )
    ids = tokenizer.encode(text)
    lengths.append(<span class="hljs-built_in">len</span>(ids))

lengths = np.array(lengths)
MAX_SEQ_LEN = <span class="hljs-built_in">int</span>(np.percentile(lengths, <span class="hljs-number">99</span>)) + <span class="hljs-number">16</span>

<span class="hljs-built_in">print</span>(
    <span class="hljs-string">f&quot;p50=<span class="hljs-subst">{np.percentile(lengths, <span class="hljs-number">50</span>):<span class="hljs-number">.0</span>f}</span>  &quot;</span>
    <span class="hljs-string">f&quot;p95=<span class="hljs-subst">{np.percentile(lengths, <span class="hljs-number">95</span>):<span class="hljs-number">.0</span>f}</span>  &quot;</span>
    <span class="hljs-string">f&quot;p99=<span class="hljs-subst">{np.percentile(lengths, <span class="hljs-number">99</span>):<span class="hljs-number">.0</span>f}</span>  &quot;</span>
    <span class="hljs-string">f&quot;max=<span class="hljs-subst">{lengths.<span class="hljs-built_in">max</span>()}</span>&quot;</span>
)
<span class="hljs-built_in">print</span>(<span class="hljs-string">f&quot;Setting MAX_SEQ_LEN = <span class="hljs-subst">{MAX_SEQ_LEN}</span>&quot;</span>)`,lang:"python",wrap:!1});var f=s(Z,4);a(f,{title:"9. Fine-tune with SFTTrainer",local:"9-fine-tune-with-sfttrainer",headingTag:"h2"});var G=s(f,4);l(G,{code:"ZnJvbSUyMGRhdGFzZXRzJTIwaW1wb3J0JTIwRGF0YXNldCUwQWZyb20lMjB0cmFuc2Zvcm1lcnMlMjBpbXBvcnQlMjBBdXRvTW9kZWxGb3JDYXVzYWxMTSUwQWZyb20lMjB0cmwlMjBpbXBvcnQlMjBTRlRDb25maWclMkMlMjBTRlRUcmFpbmVyJTBBJTBBZGF0YXNldCUyMCUzRCUyMERhdGFzZXQuZnJvbV9saXN0KCU1QiU3QiUyMm1lc3NhZ2VzJTIyJTNBJTIwciU1QiUyMm1lc3NhZ2VzJTIyJTVEJTdEJTIwZm9yJTIwciUyMGluJTIwY29ycmVjdCU1RCklMEElMEFtb2RlbCUyMCUzRCUyMEF1dG9Nb2RlbEZvckNhdXNhbExNLmZyb21fcHJldHJhaW5lZCglMjJRd2VuJTJGUXdlbjMtMS43QiUyMiklMEElMEFzZnRfY29uZmlnJTIwJTNEJTIwU0ZUQ29uZmlnKCUwQSUyMCUyMCUyMCUyMG91dHB1dF9kaXIlM0QlMjJyZWFzb25pbmctZ3ltLWNoYWluLXN1bS1Rd2VuMy0xLjdCLXNmdCUyMiUyQyUwQSUyMCUyMCUyMCUyMG1heF9sZW5ndGglM0RNQVhfU0VRX0xFTiUyQyUwQSUyMCUyMCUyMCUyMG51bV90cmFpbl9lcG9jaHMlM0QzJTJDJTBBJTIwJTIwJTIwJTIwcGVyX2RldmljZV90cmFpbl9iYXRjaF9zaXplJTNENCUyQyUwQSUyMCUyMCUyMCUyMGdyYWRpZW50X2FjY3VtdWxhdGlvbl9zdGVwcyUzRDIlMkMlMEElMjAlMjAlMjAlMjBsZWFybmluZ19yYXRlJTNEMmUtNSUyQyUwQSUyMCUyMCUyMCUyMHdhcm11cF9zdGVwcyUzRDEwJTJDJTBBJTIwJTIwJTIwJTIwbHJfc2NoZWR1bGVyX3R5cGUlM0QlMjJjb3NpbmUlMjIlMkMlMEElMjAlMjAlMjAlMjBsb2dnaW5nX3N0ZXBzJTNENSUyQyUwQSUyMCUyMCUyMCUyMHNhdmVfc3RyYXRlZ3klM0QlMjJubyUyMiUyQyUwQSUyMCUyMCUyMCUyMGFzc2lzdGFudF9vbmx5X2xvc3MlM0RUcnVlJTJDJTBBKSUwQSUwQXRyYWluZXIlMjAlM0QlMjBTRlRUcmFpbmVyKCUwQSUyMCUyMCUyMCUyMG1vZGVsJTNEbW9kZWwlMkMlMEElMjAlMjAlMjAlMjB0cmFpbl9kYXRhc2V0JTNEZGF0YXNldCUyQyUwQSUyMCUyMCUyMCUyMHByb2Nlc3NpbmdfY2xhc3MlM0R0b2tlbml6ZXIlMkMlMEElMjAlMjAlMjAlMjBhcmdzJTNEc2Z0X2NvbmZpZyUyQyUwQSklMEElMEF0cmFpbmVyLnRyYWluKCklMEF0cmFpbmVyLnB1c2hfdG9faHViKGNvbW1pdF9tZXNzYWdlJTNEJTIyU0ZUJTIwd2FybS11cCUyMG9uJTIwcmVhc29uaW5nX2d5bSUyMGNoYWluX3N1bSUyMik=",highlighted:`<span class="hljs-keyword">from</span> datasets <span class="hljs-keyword">import</span> Dataset
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoModelForCausalLM
<span class="hljs-keyword">from</span> trl <span class="hljs-keyword">import</span> SFTConfig, SFTTrainer

dataset = Dataset.from_list([{<span class="hljs-string">&quot;messages&quot;</span>: r[<span class="hljs-string">&quot;messages&quot;</span>]} <span class="hljs-keyword">for</span> r <span class="hljs-keyword">in</span> correct])

model = AutoModelForCausalLM.from_pretrained(<span class="hljs-string">&quot;Qwen/Qwen3-1.7B&quot;</span>)

sft_config = SFTConfig(
    output_dir=<span class="hljs-string">&quot;reasoning-gym-chain-sum-Qwen3-1.7B-sft&quot;</span>,
    max_length=MAX_SEQ_LEN,
    num_train_epochs=<span class="hljs-number">3</span>,
    per_device_train_batch_size=<span class="hljs-number">4</span>,
    gradient_accumulation_steps=<span class="hljs-number">2</span>,
    learning_rate=<span class="hljs-number">2e-5</span>,
    warmup_steps=<span class="hljs-number">10</span>,
    lr_scheduler_type=<span class="hljs-string">&quot;cosine&quot;</span>,
    logging_steps=<span class="hljs-number">5</span>,
    save_strategy=<span class="hljs-string">&quot;no&quot;</span>,
    assistant_only_loss=<span class="hljs-literal">True</span>,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    processing_class=tokenizer,
    args=sft_config,
)

trainer.train()
trainer.push_to_hub(commit_message=<span class="hljs-string">&quot;SFT warm-up on reasoning_gym chain_sum&quot;</span>)`,lang:"python",wrap:!1});var E=s(G,6);a(E,{title:"10. Evaluate: before vs after",local:"10-evaluate-before-vs-after",headingTag:"h2"});var R=s(E,4);l(R,{code:"aW1wb3J0JTIwcmUlMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwcGlwZWxpbmUlMEFmcm9tJTIwcmVhc29uaW5nX2d5bV9lbnYuY2xpZW50JTIwaW1wb3J0JTIwUmVhc29uaW5nR3ltRW52JTBBZnJvbSUyMHJlYXNvbmluZ19neW1fZW52Lm1vZGVscyUyMGltcG9ydCUyMFJlYXNvbmluZ0d5bUFjdGlvbiUwQSUwQSUwQWFzeW5jJTIwZGVmJTIwZXZhbHVhdGVfbW9kZWwobW9kZWxfbmFtZSUyQyUyMG5fZXZhbCUzRDUwJTJDJTIwc2VlZCUzRDk5OSklM0ElMEElMjAlMjAlMjAlMjBnZW4lMjAlM0QlMjBwaXBlbGluZSglMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjJ0ZXh0LWdlbmVyYXRpb24lMjIlMkMlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjBtb2RlbCUzRG1vZGVsX25hbWUlMkMlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjB0b2tlbml6ZXIlM0Rtb2RlbF9uYW1lJTJDJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwZGV2aWNlX21hcCUzRCUyMmF1dG8lMjIlMkMlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjBkdHlwZSUzRCUyMmF1dG8lMjIlMkMlMEElMjAlMjAlMjAlMjApJTBBJTIwJTIwJTIwJTIwZ2VuLm1vZGVsLmdlbmVyYXRpb25fY29uZmlnLm1heF9sZW5ndGglMjAlM0QlMjBOb25lJTBBJTIwJTIwJTIwJTIwdG9rJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQobW9kZWxfbmFtZSklMEElMjAlMjAlMjAlMjBlbnYlMjAlM0QlMjBSZWFzb25pbmdHeW1FbnYoYmFzZV91cmwlM0QlMjJodHRwcyUzQSUyRiUyRnNlcmdpb3BhbmllZ28tcmVhc29uaW5nLWd5bS5oZi5zcGFjZSUyMiklMEElMEElMjAlMjAlMjAlMjBvYnMlMjAlM0QlMjBhd2FpdCUyMGVudi5yZXNldCglMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjBkYXRhc2V0X25hbWUlM0QlMjJjaGFpbl9zdW0lMjIlMkMlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjBkYXRhc2V0X2NvbmZpZyUzRERBVEFTRVRfQ09ORklHJTJDJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwc2VlZCUzRHNlZWQlMkMlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjBzaXplJTNEbl9ldmFsJTJDJTBBJTIwJTIwJTIwJTIwKSUwQSUwQSUyMCUyMCUyMCUyMHJld2FyZHMlMkMlMjBmb3JtYXRfaGl0cyUyMCUzRCUyMCU1QiU1RCUyQyUyMDAlMEElMEElMjAlMjAlMjAlMjBmb3IlMjBpJTIwaW4lMjByYW5nZShuX2V2YWwpJTNBJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwaWYlMjBpJTIwJTNFJTIwMCUzQSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMG9icyUyMCUzRCUyMGF3YWl0JTIwZW52LnJlc2V0KCklMEElMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjBxdWVzdGlvbiUyMCUzRCUyMG9icy5vYnNlcnZhdGlvbi5xdWVzdGlvbiUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMG1lc3NhZ2VzJTIwJTNEJTIwJTVCJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTdCJTIycm9sZSUyMiUzQSUyMCUyMnN5c3RlbSUyMiUyQyUyMCUyMmNvbnRlbnQlMjIlM0ElMjBTWVNURU1fUFJPTVBUJTdEJTJDJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTdCJTIycm9sZSUyMiUzQSUyMCUyMnVzZXIlMjIlMkMlMjAlMjJjb250ZW50JTIyJTNBJTIwcXVlc3Rpb24lN0QlMkMlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlNUQlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjBwcm9tcHQlMjAlM0QlMjB0b2suYXBwbHlfY2hhdF90ZW1wbGF0ZSglMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjBtZXNzYWdlcyUyQyUyMHRva2VuaXplJTNERmFsc2UlMkMlMjBhZGRfZ2VuZXJhdGlvbl9wcm9tcHQlM0RUcnVlJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwKSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMGNvbXBsZXRpb24lMjAlM0QlMjBnZW4ocHJvbXB0JTJDJTIwbWF4X25ld190b2tlbnMlM0Q2NCklNUIwJTVEJTVCJTIyZ2VuZXJhdGVkX3RleHQlMjIlNUQlNUJsZW4ocHJvbXB0KSUzQSU1RCUwQSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMG0lMjAlM0QlMjByZS5zZWFyY2gociclMjJhbnN3ZXIlMjIlNUNzKiUzQSU1Q3MqJTIyJTNGKCU1Q2QlMkIpJTIyJTNGJyUyQyUyMGNvbXBsZXRpb24pJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwaWYlMjBtJTNBJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwZm9ybWF0X2hpdHMlMjAlMkIlM0QlMjAxJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwYW5zd2VyJTIwJTNEJTIwbS5ncm91cCgxKSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMGVsc2UlM0ElMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjBudW1zJTIwJTNEJTIwcmUuZmluZGFsbChyJTIyJTVDYiglNUNkJTJCKSU1Q2IlMjIlMkMlMjBjb21wbGV0aW9uKSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMGFuc3dlciUyMCUzRCUyMG51bXMlNUItMSU1RCUyMGlmJTIwbnVtcyUyMGVsc2UlMjAlMjIwJTIyJTBBJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwcmVzdWx0JTIwJTNEJTIwYXdhaXQlMjBlbnYuc3RlcChSZWFzb25pbmdHeW1BY3Rpb24oYW5zd2VyJTNEYW5zd2VyKSklMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjByZXdhcmRzLmFwcGVuZChmbG9hdChyZXN1bHQub2JzZXJ2YXRpb24uc2NvcmUlMjBvciUyMDAuMCkpJTBBJTBBJTIwJTIwJTIwJTIwYXdhaXQlMjBlbnYuY2xvc2UoKSUwQSUyMCUyMCUyMCUyMGRlbCUyMGdlbiUyMCUyMCUyMyUyMGZyZWUlMjBHUFUlMjBtZW1vcnklMjBiZWZvcmUlMjBsb2FkaW5nJTIwdGhlJTIwbmV4dCUyMG1vZGVsJTBBJTBBJTIwJTIwJTIwJTIwcmV0dXJuJTIwJTdCJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIyYWNjdXJhY3klMjIlM0ElMjBzdW0ocmV3YXJkcyklMjAlMkYlMjBsZW4ocmV3YXJkcyklMkMlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjJmb3JtYXRfY29tcGxpYW5jZSUyMiUzQSUyMGZvcm1hdF9oaXRzJTIwJTJGJTIwbl9ldmFsJTJDJTBBJTIwJTIwJTIwJTIwJTdEJTBBJTBBJTBBYmFzZV9tZXRyaWNzJTIwJTNEJTIwYXdhaXQlMjBldmFsdWF0ZV9tb2RlbCglMjJRd2VuJTJGUXdlbjMtMS43QiUyMiklMEFzZnRfbWV0cmljcyUyMCUzRCUyMGF3YWl0JTIwZXZhbHVhdGVfbW9kZWwoZiUyMiU3QllPVVJfSEZfVVNFUk5BTUUlN0QlMkZyZWFzb25pbmctZ3ltLWNoYWluLXN1bS1Rd2VuMy0xLjdCLXNmdCUyMiklMEElMEFwcmludChmJTIyJTVDbiU3QidNZXRyaWMnJTNBJTNDMjUlN0QlMjAlN0InQmFzZSUyMG1vZGVsJyUzQSUzRTEyJTdEJTIwJTdCJ0FmdGVyJTIwU0ZUJyUzQSUzRTEyJTdEJTIwJTdCJ0RlbHRhJyUzQSUzRTEwJTdEJTIyKSUwQXByaW50KCUyMi0lMjIlMjAqJTIwNjIpJTBBZm9yJTIwa2V5JTJDJTIwbGFiZWwlMjBpbiUyMCU1QiglMjJmb3JtYXRfY29tcGxpYW5jZSUyMiUyQyUyMCUyMkZvcm1hdCUyMGNvbXBsaWFuY2UlMjIpJTJDJTIwKCUyMmFjY3VyYWN5JTIyJTJDJTIwJTIyQWNjdXJhY3klMjIpJTVEJTNBJTBBJTIwJTIwJTIwJTIwYiUyQyUyMHMlMjAlM0QlMjBiYXNlX21ldHJpY3MlNUJrZXklNUQlMkMlMjBzZnRfbWV0cmljcyU1QmtleSU1RCUwQSUyMCUyMCUyMCUyMHByaW50KGYlMjIlN0JsYWJlbCUzQSUzQzI1JTdEJTIwJTdCYiUzQSUzRTEyLjElMjUlN0QlMjAlN0JzJTNBJTNFMTIuMSUyNSU3RCUyMCU3QihzJTIwLSUyMGIpJTIwKiUyMDEwMCUzQSUzRSUyQjkuMWYlN0QlMjBwcCUyMik=",highlighted:`<span class="hljs-keyword">import</span> re
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> pipeline
<span class="hljs-keyword">from</span> reasoning_gym_env.client <span class="hljs-keyword">import</span> ReasoningGymEnv
<span class="hljs-keyword">from</span> reasoning_gym_env.models <span class="hljs-keyword">import</span> ReasoningGymAction


<span class="hljs-keyword">async</span> <span class="hljs-keyword">def</span> <span class="hljs-title function_">evaluate_model</span>(<span class="hljs-params">model_name, n_eval=<span class="hljs-number">50</span>, seed=<span class="hljs-number">999</span></span>):
    gen = pipeline(
        <span class="hljs-string">&quot;text-generation&quot;</span>,
        model=model_name,
        tokenizer=model_name,
        device_map=<span class="hljs-string">&quot;auto&quot;</span>,
        dtype=<span class="hljs-string">&quot;auto&quot;</span>,
    )
    gen.model.generation_config.max_length = <span class="hljs-literal">None</span>
    tok = AutoTokenizer.from_pretrained(model_name)
    env = ReasoningGymEnv(base_url=<span class="hljs-string">&quot;https://sergiopaniego-reasoning-gym.hf.space&quot;</span>)

    obs = <span class="hljs-keyword">await</span> env.reset(
        dataset_name=<span class="hljs-string">&quot;chain_sum&quot;</span>,
        dataset_config=DATASET_CONFIG,
        seed=seed,
        size=n_eval,
    )

    rewards, format_hits = [], <span class="hljs-number">0</span>

    <span class="hljs-keyword">for</span> i <span class="hljs-keyword">in</span> <span class="hljs-built_in">range</span>(n_eval):
        <span class="hljs-keyword">if</span> i &gt; <span class="hljs-number">0</span>:
            obs = <span class="hljs-keyword">await</span> env.reset()

        question = obs.observation.question
        messages = [
            {<span class="hljs-string">&quot;role&quot;</span>: <span class="hljs-string">&quot;system&quot;</span>, <span class="hljs-string">&quot;content&quot;</span>: SYSTEM_PROMPT},
            {<span class="hljs-string">&quot;role&quot;</span>: <span class="hljs-string">&quot;user&quot;</span>, <span class="hljs-string">&quot;content&quot;</span>: question},
        ]
        prompt = tok.apply_chat_template(
            messages, tokenize=<span class="hljs-literal">False</span>, add_generation_prompt=<span class="hljs-literal">True</span>
        )
        completion = gen(prompt, max_new_tokens=<span class="hljs-number">64</span>)[<span class="hljs-number">0</span>][<span class="hljs-string">&quot;generated_text&quot;</span>][<span class="hljs-built_in">len</span>(prompt):]

        m = re.search(<span class="hljs-string">r&#x27;&quot;answer&quot;\\s*:\\s*&quot;?(\\d+)&quot;?&#x27;</span>, completion)
        <span class="hljs-keyword">if</span> m:
            format_hits += <span class="hljs-number">1</span>
            answer = m.group(<span class="hljs-number">1</span>)
        <span class="hljs-keyword">else</span>:
            nums = re.findall(<span class="hljs-string">r&quot;\\b(\\d+)\\b&quot;</span>, completion)
            answer = nums[-<span class="hljs-number">1</span>] <span class="hljs-keyword">if</span> nums <span class="hljs-keyword">else</span> <span class="hljs-string">&quot;0&quot;</span>

        result = <span class="hljs-keyword">await</span> env.step(ReasoningGymAction(answer=answer))
        rewards.append(<span class="hljs-built_in">float</span>(result.observation.score <span class="hljs-keyword">or</span> <span class="hljs-number">0.0</span>))

    <span class="hljs-keyword">await</span> env.close()
    <span class="hljs-keyword">del</span> gen  <span class="hljs-comment"># free GPU memory before loading the next model</span>

    <span class="hljs-keyword">return</span> {
        <span class="hljs-string">&quot;accuracy&quot;</span>: <span class="hljs-built_in">sum</span>(rewards) / <span class="hljs-built_in">len</span>(rewards),
        <span class="hljs-string">&quot;format_compliance&quot;</span>: format_hits / n_eval,
    }


base_metrics = <span class="hljs-keyword">await</span> evaluate_model(<span class="hljs-string">&quot;Qwen/Qwen3-1.7B&quot;</span>)
sft_metrics = <span class="hljs-keyword">await</span> evaluate_model(<span class="hljs-string">f&quot;<span class="hljs-subst">{YOUR_HF_USERNAME}</span>/reasoning-gym-chain-sum-Qwen3-1.7B-sft&quot;</span>)

<span class="hljs-built_in">print</span>(<span class="hljs-string">f&quot;\\n<span class="hljs-subst">{<span class="hljs-string">&#x27;Metric&#x27;</span>:&lt;<span class="hljs-number">25</span>}</span> <span class="hljs-subst">{<span class="hljs-string">&#x27;Base model&#x27;</span>:&gt;<span class="hljs-number">12</span>}</span> <span class="hljs-subst">{<span class="hljs-string">&#x27;After SFT&#x27;</span>:&gt;<span class="hljs-number">12</span>}</span> <span class="hljs-subst">{<span class="hljs-string">&#x27;Delta&#x27;</span>:&gt;<span class="hljs-number">10</span>}</span>&quot;</span>)
<span class="hljs-built_in">print</span>(<span class="hljs-string">&quot;-&quot;</span> * <span class="hljs-number">62</span>)
<span class="hljs-keyword">for</span> key, label <span class="hljs-keyword">in</span> [(<span class="hljs-string">&quot;format_compliance&quot;</span>, <span class="hljs-string">&quot;Format compliance&quot;</span>), (<span class="hljs-string">&quot;accuracy&quot;</span>, <span class="hljs-string">&quot;Accuracy&quot;</span>)]:
    b, s = base_metrics[key], sft_metrics[key]
    <span class="hljs-built_in">print</span>(<span class="hljs-string">f&quot;<span class="hljs-subst">{label:&lt;<span class="hljs-number">25</span>}</span> <span class="hljs-subst">{b:&gt;<span class="hljs-number">12.1</span>%}</span> <span class="hljs-subst">{s:&gt;<span class="hljs-number">12.1</span>%}</span> <span class="hljs-subst">{(s - b) * <span class="hljs-number">100</span>:&gt;+<span class="hljs-number">9.1</span>f}</span> pp&quot;</span>)`,lang:"python",wrap:!1});var N=s(R,10);a(N,{title:"11. Where to go next: GRPO",local:"11-where-to-go-next-grpo",headingTag:"h2"});var Q=s(N,4);l(Q,{code:"JTIzJTIwQmVmb3JlJTIwKGNvbGQtc3RhcnQlMjBmcm9tJTIwdGhlJTIwYmFzZSUyMG1vZGVsKSUzQSUwQU1PREVMX05BTUUlMjAlM0QlMjAlMjJRd2VuJTJGUXdlbjMtMS43QiUyMiUwQSUwQSUyMyUyMEFmdGVyJTIwKHdhcm0tc3RhcnQlMjBmcm9tJTIweW91ciUyMFNGVCUyMGNoZWNrcG9pbnQpJTNBJTBBTU9ERUxfTkFNRSUyMCUzRCUyMGYlMjIlN0JZT1VSX0hGX1VTRVJOQU1FJTdEJTJGcmVhc29uaW5nLWd5bS1jaGFpbi1zdW0tUXdlbjMtMS43Qi1zZnQlMjI=",highlighted:`<span class="hljs-comment"># Before (cold-start from the base model):</span>
MODEL_NAME = <span class="hljs-string">&quot;Qwen/Qwen3-1.7B&quot;</span>

<span class="hljs-comment"># After (warm-start from your SFT checkpoint):</span>
MODEL_NAME = <span class="hljs-string">f&quot;<span class="hljs-subst">{YOUR_HF_USERNAME}</span>/reasoning-gym-chain-sum-Qwen3-1.7B-sft&quot;</span>`,lang:"python",wrap:!1});var _=s(Q,8);Y(_,{source:"https://github.com/openenv/OpenEnv/blob/main/docs/source/tutorials/sft-warmup.md"}),K(2),W(S,n),D()}export{es as component};
