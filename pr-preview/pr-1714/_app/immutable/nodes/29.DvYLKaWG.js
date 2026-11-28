import"../chunks/DsnmJJEf.js";import{i as _,h as I,C as j,H as t,a,E as z,s as U}from"../chunks/Dz7d4_v9.js";import{p as V,o as A,s as e,f as B,a as M,b as k,c as f,n as L}from"../chunks/BSuxAqoA.js";const W='{"title":"LLM Capacity Reservation","local":"llm-capacity-reservation","sections":[{"title":"Task and interaction","local":"task-and-interaction","sections":[],"depth":2},{"title":"Cost and reward","local":"cost-and-reward","sections":[],"depth":2},{"title":"Data, split and provenance","local":"data-split-and-provenance","sections":[],"depth":2},{"title":"Install and run","local":"install-and-run","sections":[],"depth":2},{"title":"Baselines","local":"baselines","sections":[],"depth":2},{"title":"Optional trajectory recording","local":"optional-trajectory-recording","sections":[],"depth":2}],"depth":1}';var x=f('<meta name="hf:doc:metadata"/>'),N=f(`<p></p> <!> <!> <p>An OpenEnv MCP environment for sequential capacity reservation using real, publicly
released BurstGPT conversation-service workload logs. Before seeing the next day’s
requests, an agent commits 24 hourly token reservations for each of two deployments.
The environment then reveals demand and charges reservation and overflow costs.</p> <p>The environment replays recorded demand and applies a fixed reservation and
shortfall cost model.</p> <!> <p>One episode contains 14 decision days. The observation provides the previous 14
realized days, the next relative log day, cost-model coefficients, and the most
recent outcome. Compare <code>cumulative_modeled_cost</code> only across completed episodes
on the same start. Summed reward treats a day that is never committed as zero
savings.</p> <p>Each action is a dictionary with exactly <code>ChatGPT</code> and <code>GPT-4</code>,
each containing 24 finite nonnegative hourly token reservations in <code>[0, 1000000]</code>.
Reservations are measured in tokens per hour; the upper bound limits the action space.</p> <p>Agent-facing MCP tools:</p> <ul><li><code>observe_demand_history()</code>: inspect revealed history and costs; does not advance time.</li> <li><code>reserve_capacity(reservations)</code>: irreversibly commit all 48 hourly quantities,
advance one day, and return the realized demand, cost and next observation.</li></ul> <p>The training orchestrator owns <code>reset(seed=...)</code>, <code>step</code>, and <code>state</code>; these are
not agent-facing MCP tools. Reading history pays zero reward. A successful
reservation emits its day’s reward exactly once. Invalid actions do not change
costs or advance the decision day. Calls after the final decision cannot submit
additional reservations.</p> <!> <p>For model <code>m</code> and hour <code>h</code>, let <code>r</code> be reserved tokens and <code>d</code> realized total tokens:</p> <!> <p>Minimize total modeled episode cost, or equivalently maximize summed reward.
The coefficients are fixed modeling assumptions for this capacity-reservation
task. Reward is continuous. The factor 1000 is a fixed numerical scale that
preserves the optimal policy.</p> <p>The per-decision reward range is <code>[-0.948, 1.422]</code>. The data loader checks
that hourly demand is at most 500,000 tokens. Reserving zero earns exactly zero;
reserving useful capacity can earn positive savings, while excess reservation can
produce negative reward. The lower bound occurs when demand is zero and all
48 reservations are 1,000,000 tokens. The upper bound occurs when demand and
reservations are both 500,000 tokens in every hour. Reset, invalid actions and
observation-only calls earn zero and do not count as completed decisions.
For a fixed completed episode, the zero-reservation cost is policy-independent,
so maximizing total reward is equivalent to minimizing total modeled cost.</p> <p>The 0.75 empirical quantile is a useful baseline because the cost has a 4x
shortfall multiplier. Drift and limited observation history create forecasting
uncertainty as workload levels change.</p> <!> <p>Data: Yuxin Wang et al., <em>BurstGPT: A Real-World Workload Dataset to Optimize LLM
Serving Systems</em>, KDD 2025, DOI <code>10.1145/3711896.3737413</code>, arXiv <code>2401.17644</code>.
Upstream: <a href="https://github.com/HPMLL/BurstGPT" rel="nofollow">https://github.com/HPMLL/BurstGPT</a>, release <code>v2.0</code>.</p> <p>We use hourly sums of <code>Total tokens</code> from <code>Conversation log</code> records, for the <code>ChatGPT</code> and <code>GPT-4</code> labels. There are 231 original relative days: 0–120 and
225–334. Days 121–224 are absent in the upstream release. Episodes never cross
this gap. Missing hours within a logged day have zero demand.
Day 77 has zero ChatGPT demand in every hour; test days 101–106 have
7–20 zero-demand hours per deployment. These may reflect upstream logging gaps
rather than absent traffic. They remain scored as zero demand, so costs partly
reflect forecasting these gaps; the trace does not establish real service outages.</p> <p>All 231 days from the two selected conversation-service streams are retained.
Results describe these two streams and do not represent all LLM workloads.
No request text, completions, or user identifiers are distributed.</p> <ul><li>Train decision-day segments: 14–89 and 239–304.</li> <li>Test decision-day segments: 90–120 and 305–334.</li> <li>The split is chronological within each recorded segment, not across segments:
the second train segment comes after the first test segment.</li> <li>Start points require 14 preceding observed days and 14 consecutive target days.</li> <li>There are 116 train starts and 35 test starts. Adjacent episodes overlap; they
must not be treated as independent experimental replicates.</li> <li>Test episodes may use earlier revealed history, including earlier days of the
test time segment; this models online deployment with expanding observations.</li></ul> <p>Set <code>CAPACITY_SPLIT=train</code> (default) or <code>test</code> when starting the server. Seeded
reset chooses uniformly from eligible starts. The split is orchestrator-owned.
The public trace is shipped on the server side. Agents interact through the MCP
tools; the server filesystem contains the full trace and is reserved for the
environment operator. No live data API or provider account is required.</p> <p>To independently reconstruct the aggregate, download <code>BurstGPT_1.csv</code>, <code>BurstGPT_2.csv</code>, and <code>BurstGPT_3.csv</code> from upstream release v2.0 into a raw directory
(gzip versions are also accepted), then run:</p> <!> <p>Expected SHA-256: <code>503ba7712c204aea5295427d6cef49fd0f662338fb62598453dfbe2b35877f77</code>.
The reconstruction script aggregates all three original release assets using the
selection and zero-filling rules above.</p> <p>Data license: CC-BY-4.0, preserved in <code>server/data/LICENSE-BurstGPT.txt</code>.
Modification details and aggregate hashes are in <code>server/data/PROVENANCE.json</code>.
Environment code license: BSD-3-Clause. See <code>THIRD_PARTY_NOTICES.md</code>.</p> <!> <p>The environment supports Python 3.11 and later and requires OpenEnv 0.11.0 or later.
From <code>envs/llm_capacity_env</code>, install the frozen dependency set and run:</p> <!> <p>In another shell in the same directory:</p> <!> <p>The integration tests live in the OpenEnv repository. From its root, install
its development dependencies and run:</p> <!> <p><code>uv.lock</code> supplies the same frozen runtime dependency set for the container
and the repository dependency check. Development extras are not installed in the
runtime image.</p> <p>Docker:</p> <!> <p><code>/health</code>, <code>/schema</code>, and <code>/openapi.json</code> are provided by OpenEnv. The debug UI is
available at <code>/web/</code> when <code>ENABLE_WEB_INTERFACE=true</code> (enabled in the Docker image).
MCP and WebSocket clients use OpenEnv’s standard endpoints. The server supports
independent concurrent sessions through an environment factory.</p> <!> <p><code>examples/evaluate.py</code> evaluates zero reservation, previous-day demand, a fixed
initial-history 0.75 quantile, and an adaptive rolling 0.75 quantile on all eligible
starts. It also reports a hindsight diagnostic that reserves the realized demand;
this diagnostic reads future data and is not an agent policy. Compare completed
episodes on the same starts using mean modeled cost (lower is better) or summed
savings reward (higher is better). Overlapping episodes are not independent samples.</p> <!> <p>Recording is disabled by default. To write one JSONL file per server episode,
set <code>CAPACITY_TRACE_DIR=/path/to/trajectories</code> before starting the server, or pass <code>trace_dir</code> when constructing <code>CapacityEnvironment</code>. The operator is responsible
for retention and disk limits when recording is enabled.</p> <p>To generate and independently replay complete heuristic-policy examples:</p> <!> <p>Generated traces and validation logs are not included in the environment package.</p> <!> <p></p>`,1);function H(J,C){V(C,!1),A(()=>{new URLSearchParams(window.location.search).get("fw")}),_();var s=N();I("kh0mf0",w=>{var b=x();U(b,"content",W),M(w,b)});var n=e(B(s),2);j(n,{containerStyle:"float: inline-end; margin-inline-start: 10px; display: inline-flex; position: relative; z-index: 10;"});var o=e(n,2);t(o,{title:"LLM Capacity Reservation",local:"llm-capacity-reservation",headingTag:"h1"});var r=e(o,6);t(r,{title:"Task and interaction",local:"task-and-interaction",headingTag:"h2"});var i=e(r,12);t(i,{title:"Cost and reward",local:"cost-and-reward",headingTag:"h2"});var d=e(i,4);a(d,{code:"aG91cmx5X21vZGVsZWRfY29zdCUyMCUzRCUyMHVuaXRfY29zdCU1Qm0lNUQlMjAqJTIwKHIlMjAlMkIlMjA0JTIwKiUyMG1heChkJTIwLSUyMHIlMkMlMjAwKSklMjAlMkYlMjAxMDAwJTBBdW5pdF9jb3N0JTIwJTNEJTIwJTdCQ2hhdEdQVCUzQSUyMDAuMDAyJTJDJTIwR1BULTQlM0ElMjAwLjAzNzUlN0QlMEFkYWlseV9jb3N0JTIwJTNEJTIwc3VtKGhvdXJseV9tb2RlbGVkX2Nvc3QpJTBBemVyb19yZXNlcnZhdGlvbl9jb3N0JTIwJTNEJTIwc3VtKHVuaXRfY29zdCU1Qm0lNUQlMjAqJTIwNCUyMColMjBkJTIwJTJGJTIwMTAwMCklMEFkYWlseV9yZXdhcmQlMjAlM0QlMjAoemVyb19yZXNlcnZhdGlvbl9jb3N0JTIwLSUyMGRhaWx5X2Nvc3QpJTIwJTJGJTIwMTAwMA==",highlighted:`hourly_modeled_cost = unit_cost[m] * (r + 4 * max(d - r, 0)) / 1000
unit_cost = {ChatGPT: 0.002, GPT-4: 0.0375}
daily_cost = sum(hourly_modeled_cost)
zero_reservation_cost = sum(unit_cost[m] * 4 * d / 1000)
daily_reward = (zero_reservation_cost - daily_cost) / 1000`,lang:"text",wrap:!1});var l=e(d,8);t(l,{title:"Data, split and provenance",local:"data-split-and-provenance",headingTag:"h2"});var c=e(l,14);a(c,{code:"cHl0aG9uJTIwc2NyaXB0cyUyRnJlYnVpbGRfYWdncmVnYXRlcy5weSUyMC0tcmF3LWRpciUyMCUyRnBhdGglMkZ0byUyRnJhdyUyMC0tb3V0cHV0JTIwcmVidWlsdC1ob3VybHkuY3N2JTBBc2hhMjU2c3VtJTIwcmVidWlsdC1ob3VybHkuY3N2",highlighted:`python scripts/rebuild_aggregates.py --raw-dir /path/to/raw --output rebuilt-hourly.csv
<span class="hljs-built_in">sha256sum</span> rebuilt-hourly.csv`,lang:"bash",wrap:!1});var p=e(c,6);t(p,{title:"Install and run",local:"install-and-run",headingTag:"h2"});var h=e(p,4);a(h,{code:"dXYlMjBzeW5jJTIwLS1weXRob24lMjAzLjExJTIwLS1mcm96ZW4lMjAtLWFsbC1ncm91cHMlMjAtLWFsbC1leHRyYXMlMEF1diUyMHJ1biUyMC0tZnJvemVuJTIwcHl0aG9uJTIwZXhhbXBsZXMlMkZldmFsdWF0ZS5weSUyMGJhc2VsaW5lLXJlc3VsdHMuanNvbiUwQXV2JTIwcnVuJTIwLS1mcm96ZW4lMjB1dmljb3JuJTIwc2VydmVyLmFwcCUzQWFwcCUyMC0taG9zdCUyMDAuMC4wLjAlMjAtLXBvcnQlMjA4MDAw",highlighted:`uv <span class="hljs-built_in">sync</span> --python 3.11 --frozen --all-groups --all-extras
uv run --frozen python examples/evaluate.py baseline-results.json
uv run --frozen uvicorn server.app:app --host 0.0.0.0 --port 8000`,lang:"bash",wrap:!1});var m=e(h,4);a(m,{code:"dXYlMjBydW4lMjAtLWZyb3plbiUyMHB5dGhvbiUyMGV4YW1wbGVzJTJGY2xpZW50X2RlbW8ucHklMjBodHRwJTNBJTJGJTJGbG9jYWxob3N0JTNBODAwMCUwQXV2JTIwcnVuJTIwLS1mcm96ZW4lMjBvcGVuZW52JTIwdmFsaWRhdGUlMjAuJTIwLS1sZXZlbCUyMHN0YXRpYyUyMC0tc2tpcC1idWlsZCUwQXV2JTIwcnVuJTIwLS1mcm96ZW4lMjBvcGVuZW52JTIwdmFsaWRhdGUlMjAtLXVybCUyMGh0dHAlM0ElMkYlMkZsb2NhbGhvc3QlM0E4MDAw",highlighted:`uv run --frozen python examples/client_demo.py http://localhost:8000
uv run --frozen openenv validate . --level static --skip-build
uv run --frozen openenv validate --url http://localhost:8000`,lang:"bash",wrap:!1});var u=e(m,4);a(u,{code:"dXYlMjBzeW5jJTIwLS1hbGwtZXh0cmFzJTBBUFlUSE9OUEFUSCUzRHNyYyUzQWVudnMlMjB1diUyMHJ1biUyMHB5dGVzdCUyMC1xJTIwJTVDJTBBJTIwJTIwdGVzdHMlMkZlbnZzJTJGdGVzdF9sbG1fY2FwYWNpdHlfbWNwLnB5JTIwJTVDJTBBJTIwJTIwdGVzdHMlMkZlbnZzJTJGdGVzdF9sbG1fY2FwYWNpdHlfcmVwbGF5LnB5JTIwJTVDJTBBJTIwJTIwdGVzdHMlMkZlbnZzJTJGdGVzdF9sbG1fY2FwYWNpdHlfdHJhamVjdG9yeS5weQ==",highlighted:`uv <span class="hljs-built_in">sync</span> --all-extras
PYTHONPATH=src:envs uv run pytest -q \\
  tests/envs/test_llm_capacity_mcp.py \\
  tests/envs/test_llm_capacity_replay.py \\
  tests/envs/test_llm_capacity_trajectory.py`,lang:"bash",wrap:!1});var v=e(u,6);a(v,{code:"ZG9ja2VyJTIwYnVpbGQlMjAtZiUyMHNlcnZlciUyRkRvY2tlcmZpbGUlMjAtdCUyMGxsbS1jYXBhY2l0eS1lbnYlM0EwLjEuMCUyMC4lMEFkb2NrZXIlMjBydW4lMjAtLXJtJTIwLXAlMjA4MDAwJTNBODAwMCUyMC1lJTIwQ0FQQUNJVFlfU1BMSVQlM0R0ZXN0JTIwbGxtLWNhcGFjaXR5LWVudiUzQTAuMS4w",highlighted:`docker build -f server/Dockerfile -t llm-capacity-env:0.1.0 .
docker run --<span class="hljs-built_in">rm</span> -p 8000:8000 -e CAPACITY_SPLIT=<span class="hljs-built_in">test</span> llm-capacity-env:0.1.0`,lang:"bash",wrap:!1});var y=e(v,4);t(y,{title:"Baselines",local:"baselines",headingTag:"h2"});var T=e(y,4);t(T,{title:"Optional trajectory recording",local:"optional-trajectory-recording",headingTag:"h2"});var g=e(T,6);a(g,{code:"dXYlMjBydW4lMjAtLWZyb3plbiUyMHB5dGhvbiUyMGV4YW1wbGVzJTJGcmVjb3JkX3RyYWNlLnB5JTIwb3V0cHV0cyUyRmV4YW1wbGUtdHJhY2Vz",highlighted:"uv run --frozen python examples/record_trace.py outputs/example-traces",lang:"bash",wrap:!1});var G=e(g,4);z(G,{source:"https://github.com/openenv/OpenEnv/blob/main/docs/source/environments/llm_capacity.md"}),L(2),M(J,s),k()}export{H as component};
