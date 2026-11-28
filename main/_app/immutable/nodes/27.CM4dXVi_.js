import"../chunks/DsnmJJEf.js";import{i as k,h as _,C as j,H as t,a as o,E as C,s as x}from"../chunks/Dz7d4_v9.js";import{p as A,o as B,s as e,f as Z,a as b,b as E,c as g,n as V}from"../chunks/BSuxAqoA.js";const G='{"title":"Jupyter Environment","local":"jupyter-environment","sections":[{"title":"Tools","local":"tools","sections":[],"depth":2},{"title":"Quick Start","local":"quick-start","sections":[],"depth":2},{"title":"Local Server","local":"local-server","sections":[],"depth":2},{"title":"Docker","local":"docker","sections":[],"depth":2},{"title":"Configuration","local":"configuration","sections":[],"depth":2},{"title":"Setup and Verify Commands","local":"setup-and-verify-commands","sections":[],"depth":2},{"title":"Verifying outside the agent’s sandbox","local":"verifying-outside-the-agents-sandbox","sections":[],"depth":2},{"title":"Notes","local":"notes","sections":[],"depth":2}],"depth":1}';var Q=g('<meta name="hf:doc:metadata"/>'),S=g(`<p></p> <!> <!> <p><code>jupyter_env</code> exposes a stateful Jupyter-style Python notebook through MCP
tools. Each episode creates a fresh E2B Code Interpreter sandbox, so Python
variables, imports, files, and generated plots persist across tool calls until
the next reset.</p> <p>Reset can also receive setup and verify scripts. Setup commands run immediately
after the sandbox is created. Verify commands are stored for the episode and run
when the agent calls <code>final_answer</code>.</p> <!> <ul><li><code>add_and_execute_code_cell(code)</code>: execute a new Python cell.</li> <li><code>edit_and_execute_current_cell(code)</code>: replace and re-run the latest code cell.</li> <li><code>execute_shell_command(command)</code>: run shell commands inside the sandbox.</li> <li><code>get_notebook_state(include_images=False)</code>: summarize recent notebook cells.</li> <li><code>final_answer(answer)</code>: record a final answer and run any configured verify
commands.</li></ul> <!> <!> <!> <!> <p>The API and custom web UI are served on port 8000. The notebook UI is mounted
at <code>/web</code>.</p> <!> <!> <!> <ul><li><code>E2B_API_KEY</code>: required when resetting an episode.</li> <li><code>MAX_CONCURRENT_ENVS</code>: maximum concurrent WebSocket sessions. Defaults to <code>4</code>.</li> <li><code>KAGGLE_DATA_DIR</code>: optional root directory for reset-time Kaggle file staging.</li></ul> <!> <p><code>reset()</code> accepts either <code>setup</code> / <code>verify</code> or <code>setup_scripts</code> / <code>verify_scripts</code>.</p> <!> <p>Setup failure ends the reset response with <code>done=True</code> and returns the captured
setup results. Verify commands run when <code>final_answer(answer)</code> is called. Reward
defaults to <code>passed_verify_commands / total_verify_commands</code>. A verify command
can override this by writing a float to:</p> <!> <p>The file is deleted before the verify commands run, so only a verify command
can set it; anything the agent writes there earlier is discarded. The value
must be a finite number; a verify command is written by the task author, so
negative rewards and values above 1 are accepted too. A value that is not a
finite number is ignored, the pass rate is used, and the reason is recorded in <code>state.reward_override_ignored</code>.</p> <p>Verify commands, the clearing of that file and the read of it run as their own
sandbox processes through E2B’s process API, as <code>root</code> in <code>/home/user</code>, rather
than inside the notebook kernel. A cell can rebind <code>subprocess.run</code>, change the
working directory or edit <code>os.environ</code>, and none of that reaches verification.
This removes the coupling to the kernel; it does not isolate verification from
the agent. The notebook kernel runs as root in E2B’s default code-interpreter
template, so agent code can still change anything in the sandbox, including the
files a verify command reads and the startup files of the shell it runs in.
E2B starts each command as a login shell, so a profile the agent writes decides
what a verify command reports: an <code>exit 0</code> or an <code>EXIT</code> trap in <code>/root/.bash_profile</code> makes a failing command look like a passing one. A test
pins that behaviour rather than leaving it to be found in a reward curve, and <code>submit_paths</code> closes that, per environment and per task.</p> <!> <p>A task that passes <code>submit_paths</code> to <code>reset</code> is verified in a second sandbox the
agent never touches. The list defaults to empty, and an empty list keeps the
behaviour above, so existing tasks score exactly as before.</p> <p>With paths declared, submitting runs this order, which is what makes it a
boundary rather than hardening:</p> <ol><li>A fresh sandbox is created from the same template.</li> <li>The task’s <code>setup</code> commands run there, before any of the agent’s work exists.</li> <li>The declared paths are copied out of the agent’s sandbox into <code>/submission</code>,
which is not the working directory and is not on <code>PATH</code> or <code>PYTHONPATH</code>.</li> <li>The verify commands run, and the reward file is cleared and read there.</li> <li>The sandbox is killed.</li></ol> <p>Paths are relative to <code>/home/user</code>. An absolute path, one that leaves the
workspace with <code>..</code>, a symlink, a socket, a fifo, a device, or a submission over
200 files or 10 MiB is refused, and the reason is recorded in <code>state.submission_refused</code>. Files that a shell or a test runner reads before the
first assertion are left behind even when a declared directory contains them:
the login files a shell reads (<code>.bash_profile</code>, <code>.bash_login</code>, <code>.profile</code>, <code>.bashrc</code>, <code>.bash_logout</code>), what Python runs before an import
(<code>sitecustomize.py</code>, <code>usercustomize.py</code>, <code>*.pth</code>, <code>conftest.py</code>), and every
file pytest takes options from (<code>pytest.ini</code>, <code>.pytest.ini</code>, <code>pytest.toml</code>, <code>.pytest.toml</code>, <code>pyproject.toml</code>, <code>setup.cfg</code>, <code>tox.ini</code>), since any of them can carry <code>addopts = --co</code> and report success
without running a test. Declaring one of those exactly still submits it, since
naming it is the task author’s decision.</p> <p>Because the submission is no longer in the workspace, a task that opts in writes
its verify commands against <code>/submission</code>. The task’s own test files keep coming
from its <code>setup</code> commands, so the copy cannot replace them.</p> <p>The boundary covers what the agent leaves in its own sandbox, not the code it
submits. A verify command that imports or runs the submission runs that code as
root in the verifier sandbox, where it can do anything root can, including
ending the test process with status 0 and writing the reward file.</p> <p>The cost is one extra sandbox per scored rollout, which is why it is opt-in
rather than the default.</p> <p>Only a command that ran and exited decides a verify result. If the sandbox
itself fails, because it cannot be reached, the command cannot be started, or
the wait for it ends without an exit status, the error is raised instead:
verification stops, no reward is produced, and the episode does not finish. A sandbox that
is unreachable, expired or unauthenticated says nothing about the agent’s work,
and a command with no exit status may still be running, since E2B’s kill
signals the command’s own process and not the children it started.</p> <!> <p>This first version intentionally keeps sandbox provider selection local to the
environment and uses E2B as the concrete backend. A shared OpenEnv sandbox API
can be introduced later once the provider contract is RFC-backed.</p> <!> <p></p>`,1);function q(v,U){A(U,!1),B(()=>{new URLSearchParams(window.location.search).get("fw")}),k();var s=S();_("11b6rhp",J=>{var T=Q();x(T,"content",G),b(J,T)});var a=e(Z(s),2);j(a,{containerStyle:"float: inline-end; margin-inline-start: 10px; display: inline-flex; position: relative; z-index: 10;"});var n=e(a,2);t(n,{title:"Jupyter Environment",local:"jupyter-environment",headingTag:"h1"});var i=e(n,6);t(i,{title:"Tools",local:"tools",headingTag:"h2"});var r=e(i,4);t(r,{title:"Quick Start",local:"quick-start",headingTag:"h2"});var l=e(r,2);o(l,{code:"ZnJvbSUyMGp1cHl0ZXJfZW52JTIwaW1wb3J0JTIwSnVweXRlckVudiUwQSUwQXdpdGglMjBKdXB5dGVyRW52KGJhc2VfdXJsJTNEJTIyaHR0cCUzQSUyRiUyRmxvY2FsaG9zdCUzQTgwMDAlMjIpLnN5bmMoKSUyMGFzJTIwZW52JTNBJTBBJTIwJTIwJTIwJTIwZW52LnJlc2V0KCUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMHNldHVwJTNEJTVCJTIycGlwJTIwaW5zdGFsbCUyMC1xJTIwcGFuZGFzJTIyJTVEJTJDJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwdmVyaWZ5JTNEJTVCJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIycHl0aG9uJTIwLSUyMCUzQyUzQydQWSclNUNuZnJvbSUyMHBhdGhsaWIlMjBpbXBvcnQlMjBQYXRoJTVDbmFzc2VydCUyMFBhdGgoJyUyRmhvbWUlMkZ1c2VyJTJGd29yayUyRmFuc3dlci50eHQnKS5leGlzdHMoKSU1Q25QWSUyMiUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCU1RCUyQyUwQSUyMCUyMCUyMCUyMCklMEElMjAlMjAlMjAlMjBwcmludChlbnYuY2FsbF90b29sKCUyMmFkZF9hbmRfZXhlY3V0ZV9jb2RlX2NlbGwlMjIlMkMlMjBjb2RlJTNEJTIyeCUyMCUzRCUyMDIxJTIwKiUyMDIlNUNueCUyMikpJTBBJTIwJTIwJTIwJTIwcHJpbnQoZW52LmNhbGxfdG9vbCglMjJmaW5hbF9hbnN3ZXIlMjIlMkMlMjBhbnN3ZXIlM0QlMjJkb25lJTIyKSk=",highlighted:`<span class="hljs-keyword">from</span> jupyter_env <span class="hljs-keyword">import</span> JupyterEnv

<span class="hljs-keyword">with</span> JupyterEnv(base_url=<span class="hljs-string">&quot;http://localhost:8000&quot;</span>).sync() <span class="hljs-keyword">as</span> env:
    env.reset(
        setup=[<span class="hljs-string">&quot;pip install -q pandas&quot;</span>],
        verify=[
            <span class="hljs-string">&quot;python - &lt;&lt;&#x27;PY&#x27;\\nfrom pathlib import Path\\nassert Path(&#x27;/home/user/work/answer.txt&#x27;).exists()\\nPY&quot;</span>
        ],
    )
    <span class="hljs-built_in">print</span>(env.call_tool(<span class="hljs-string">&quot;add_and_execute_code_cell&quot;</span>, code=<span class="hljs-string">&quot;x = 21 * 2\\nx&quot;</span>))
    <span class="hljs-built_in">print</span>(env.call_tool(<span class="hljs-string">&quot;final_answer&quot;</span>, answer=<span class="hljs-string">&quot;done&quot;</span>))`,lang:"python",wrap:!1});var d=e(l,2);t(d,{title:"Local Server",local:"local-server",headingTag:"h2"});var c=e(d,2);o(c,{code:"Y2QlMjBlbnZzJTJGanVweXRlcl9lbnYlMEFFMkJfQVBJX0tFWSUzRGUyYl8uLi4lMjB1diUyMHJ1biUyMC0tcHJvamVjdCUyMC4lMjBzZXJ2ZXI=",highlighted:`<span class="hljs-built_in">cd</span> envs/jupyter_env
E2B_API_KEY=e2b_... uv run --project . server`,lang:"bash",wrap:!1});var h=e(c,4);t(h,{title:"Docker",local:"docker",headingTag:"h2"});var p=e(h,2);o(p,{code:"Y2QlMjBlbnZzJTJGanVweXRlcl9lbnYlMEFvcGVuZW52JTIwYnVpbGQlMjAtdCUyMGp1cHl0ZXItZW52JTBBZG9ja2VyJTIwcnVuJTIwLXAlMjA4MDAwJTNBODAwMCUyMC1lJTIwRTJCX0FQSV9LRVklM0RlMmJfLi4uJTIwanVweXRlci1lbnY=",highlighted:`<span class="hljs-built_in">cd</span> envs/jupyter_env
openenv build -t jupyter-env
docker run -p 8000:8000 -e E2B_API_KEY=e2b_... jupyter-env`,lang:"bash",wrap:!1});var u=e(p,2);t(u,{title:"Configuration",local:"configuration",headingTag:"h2"});var y=e(u,4);t(y,{title:"Setup and Verify Commands",local:"setup-and-verify-commands",headingTag:"h2"});var m=e(y,4);o(m,{code:"ZW52LnJlc2V0KCUwQSUyMCUyMCUyMCUyMHNldHVwJTNEJTVCJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIybWtkaXIlMjAtcCUyMCUyRmhvbWUlMkZ1c2VyJTJGd29yayUyMiUyQyUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMnByaW50ZiUyMCdzZWVkJTIwZGF0YSclMjAlM0UlMjAlMkZob21lJTJGdXNlciUyRndvcmslMkZpbnB1dC50eHQlMjIlMkMlMEElMjAlMjAlMjAlMjAlNUQlMkMlMEElMjAlMjAlMjAlMjB2ZXJpZnklM0QlNUIlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjJ0ZXN0JTIwLWYlMjAlMkZob21lJTJGdXNlciUyRndvcmslMkZhbnN3ZXIudHh0JTIyJTJDJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIycHl0aG9uJTIwLW0lMjBweXRlc3QlMjAlMkZob21lJTJGdXNlciUyRndvcmslMkZ0ZXN0cyUyMiUyQyUwQSUyMCUyMCUyMCUyMCU1RCUyQyUwQSk=",highlighted:`env.reset(
    setup=[
        <span class="hljs-string">&quot;mkdir -p /home/user/work&quot;</span>,
        <span class="hljs-string">&quot;printf &#x27;seed data&#x27; &gt; /home/user/work/input.txt&quot;</span>,
    ],
    verify=[
        <span class="hljs-string">&quot;test -f /home/user/work/answer.txt&quot;</span>,
        <span class="hljs-string">&quot;python -m pytest /home/user/work/tests&quot;</span>,
    ],
)`,lang:"python",wrap:!1});var w=e(m,4);o(w,{code:"JTJGaG9tZSUyRnVzZXIlMkZsb2dzJTJGdmVyaWZpZXIlMkZyZXdhcmQudHh0",highlighted:"/home/user/logs/verifier/reward.txt",lang:"text",wrap:!1});var M=e(w,6);t(M,{title:"Verifying outside the agent’s sandbox",local:"verifying-outside-the-agents-sandbox",headingTag:"h2"});var f=e(M,18);t(f,{title:"Notes",local:"notes",headingTag:"h2"});var I=e(f,4);C(I,{source:"https://github.com/openenv/OpenEnv/blob/main/docs/source/environments/jupyter.md"}),V(2),b(v,s),E()}export{q as component};
