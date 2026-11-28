import"../chunks/DsnmJJEf.js";import{i as W,h as B,C as G,H as e,a as s,E as f,s as Z}from"../chunks/Dz7d4_v9.js";import{p as k,o as Y,s as n,f as V,a as g,b as N,c as C,n as Q}from"../chunks/BSuxAqoA.js";const S='{"title":"Getting Started","local":"getting-started","sections":[{"title":"Install OpenEnv","local":"install-openenv","sections":[{"title":"Optional dependencies","local":"optional-dependencies","sections":[],"depth":3}],"depth":2},{"title":"Try an Environment","local":"try-an-environment","sections":[],"depth":2},{"title":"Connect to a Running Environment","local":"connect-to-a-running-environment","sections":[],"depth":2},{"title":"Use Containers or Local Servers","local":"use-containers-or-local-servers","sections":[],"depth":2},{"title":"Next Steps","local":"next-steps","sections":[],"depth":2}],"depth":1}';var _=C('<meta name="hf:doc:metadata"/>'),x=C(`<p></p> <!> <!> <p>Install OpenEnv, load an environment, and run your first step.</p> <!> <!> <blockquote class="note"><p>This installs the full OpenEnv runtime: the environment server, the client,
the <code>openenv</code> CLI, the web interface, and MCP support. Environments depend on <code>openenv</code> directly.</p></blockquote> <!> <p>A few integrations ship as optional extras. Install them with <code>pip install openenv[&lt;extra&gt;]</code>:</p> <table><thead><tr><th>Extra</th><th>Pulls in</th></tr></thead><tbody><tr><td><code>inspect</code></td><td>The Inspect AI evaluation harness</td></tr><tr><td><code>harbor</code></td><td>The <a href="environments/harbor">Harbor integration</a> and its sandbox backends (Python 3.12+)</td></tr><tr><td><code>daytona</code>, <code>aca</code>, <code>modal</code>, <code>novita</code></td><td>Cloud sandbox providers (see <a href="guides/runtime-providers">Runtime Providers</a>)</td></tr></tbody></table> <!> <p>Use <code>AutoEnv</code> to load the Echo client from its Hugging Face Space. It asks for
confirmation before installing the environment package, then connects to the
running Space. Echo exposes its actions as MCP tools.</p> <!> <p>For installed environment packages, <code>AutoEnv.from_env()</code> also accepts the common
name forms below. Without a <code>base_url</code>, these start a local Docker container and
require Docker and the environment image:</p> <!> <!> <p>OpenEnv clients are async by default. Use the async client for production code,
parallel environment runs, and integrations with async frameworks.</p> <!> <p>For scripts and notebooks, use <code>.sync()</code>:</p> <!> <!> <p>You can run an environment from a Docker image:</p> <!> <p>If you run the Hugging Face Space image yourself with <code>docker run</code>, expose port <code>7860</code> and connect to that port:</p> <!> <!> <p>Or connect to a local server:</p> <!> <!> <p>The same client also runs environments on cloud sandboxes (Daytona, Modal, Novita, Azure Container Apps, Hugging Face) and on a Kubernetes cluster through runtime providers. <code>HFSandboxProvider</code> ships with core OpenEnv and needs a Hugging Face token for an account that can run Jobs (the sandboxes are billed as Jobs): it runs an environment’s Space image in a Hugging Face sandbox (<a href="guides/runtime-providers#hfsandboxprovider">example</a>). <code>KubernetesProvider</code> is an optional extra (<code>pip install openenv[kubernetes]</code>, <a href="guides/runtime-providers#kubernetesprovider">example</a>). See the <a href="guides/runtime-providers">Runtime Providers guide</a> to pick one.</p> <!> <ul><li><a href="guides/training">Train an agent</a> with your training framework, or train a coding agent through <a href="environments/harbor">Harbor</a></li> <li><a href="environments">Explore environments</a></li> <li><a href="guides/first-environment">Build your first environment</a></li> <li><a href="guides/concepts">Concepts</a></li></ul> <!> <p></p>`,1);function q(I,E){k(E,!1),Y(()=>{new URLSearchParams(window.location.search).get("fw")}),W();var l=x();B("1xzfz5b",m=>{var T=_();Z(T,"content",S),g(m,T)});var a=n(V(l),2);G(a,{containerStyle:"float: inline-end; margin-inline-start: 10px; display: inline-flex; position: relative; z-index: 10;"});var t=n(a,2);e(t,{title:"Getting Started",local:"getting-started",headingTag:"h1"});var o=n(t,4);e(o,{title:"Install OpenEnv",local:"install-openenv",headingTag:"h2"});var i=n(o,2);s(i,{code:"cGlwJTIwaW5zdGFsbCUyMG9wZW5lbnY=",highlighted:"pip install openenv",lang:"bash",wrap:!1});var c=n(i,4);e(c,{title:"Optional dependencies",local:"optional-dependencies",headingTag:"h3"});var r=n(c,6);e(r,{title:"Try an Environment",local:"try-an-environment",headingTag:"h2"});var p=n(r,4);s(p,{code:"ZnJvbSUyMG9wZW5lbnYlMjBpbXBvcnQlMjBBdXRvRW52JTBBJTBBZW52JTIwJTNEJTIwQXV0b0Vudi5mcm9tX2VudiglMEElMjAlMjAlMjAlMjAlMjJvcGVuZW52JTJGZWNob19lbnYlMjIlMkMlMEElMjAlMjAlMjAlMjBiYXNlX3VybCUzRCUyMmh0dHBzJTNBJTJGJTJGb3BlbmVudi1lY2hvLWVudi5oZi5zcGFjZSUyMiUyQyUwQSklMEElMEF3aXRoJTIwZW52LnN5bmMoKSUyMGFzJTIwY2xpZW50JTNBJTBBJTIwJTIwJTIwJTIwcmVzdWx0JTIwJTNEJTIwY2xpZW50LnJlc2V0KCklMEElMjAlMjAlMjAlMjBwcmludChyZXN1bHQub2JzZXJ2YXRpb24ubWV0YWRhdGElNUIlMjJtZXNzYWdlJTIyJTVEKSUyMCUyMCUyMyUyMCUyMkVjaG8lMjBlbnZpcm9ubWVudCUyMHJlYWR5ISUyMiUwQSUwQSUyMCUyMCUyMCUyMG1lc3NhZ2UlMjAlM0QlMjBjbGllbnQuY2FsbF90b29sKCUyMmVjaG9fbWVzc2FnZSUyMiUyQyUyMG1lc3NhZ2UlM0QlMjJIZWxsbyUyQyUyME9wZW5FbnYhJTIyKSUwQSUyMCUyMCUyMCUyMHByaW50KG1lc3NhZ2UpJTIwJTIwJTIzJTIwJTIySGVsbG8lMkMlMjBPcGVuRW52ISUyMg==",highlighted:`<span class="hljs-keyword">from</span> openenv <span class="hljs-keyword">import</span> AutoEnv

env = AutoEnv.from_env(
    <span class="hljs-string">&quot;openenv/echo_env&quot;</span>,
    base_url=<span class="hljs-string">&quot;https://openenv-echo-env.hf.space&quot;</span>,
)

<span class="hljs-keyword">with</span> env.sync() <span class="hljs-keyword">as</span> client:
    result = client.reset()
    <span class="hljs-built_in">print</span>(result.observation.metadata[<span class="hljs-string">&quot;message&quot;</span>])  <span class="hljs-comment"># &quot;Echo environment ready!&quot;</span>

    message = client.call_tool(<span class="hljs-string">&quot;echo_message&quot;</span>, message=<span class="hljs-string">&quot;Hello, OpenEnv!&quot;</span>)
    <span class="hljs-built_in">print</span>(message)  <span class="hljs-comment"># &quot;Hello, OpenEnv!&quot;</span>`,lang:"python",wrap:!1});var M=n(p,4);s(M,{code:"ZnJvbSUyMG9wZW5lbnYlMjBpbXBvcnQlMjBBdXRvRW52JTBBJTBBQXV0b0Vudi5mcm9tX2VudiglMjJlY2hvJTIyKSUwQUF1dG9FbnYuZnJvbV9lbnYoJTIyZWNoby1lbnYlMjIpJTBBQXV0b0Vudi5mcm9tX2VudiglMjJlY2hvX2VudiUyMik=",highlighted:`<span class="hljs-keyword">from</span> openenv <span class="hljs-keyword">import</span> AutoEnv

AutoEnv.from_env(<span class="hljs-string">&quot;echo&quot;</span>)
AutoEnv.from_env(<span class="hljs-string">&quot;echo-env&quot;</span>)
AutoEnv.from_env(<span class="hljs-string">&quot;echo_env&quot;</span>)`,lang:"python",wrap:!1});var d=n(M,2);e(d,{title:"Connect to a Running Environment",local:"connect-to-a-running-environment",headingTag:"h2"});var h=n(d,4);s(h,{code:"aW1wb3J0JTIwYXN5bmNpbyUwQSUwQWZyb20lMjBlY2hvX2VudiUyMGltcG9ydCUyMENhbGxUb29sQWN0aW9uJTJDJTIwRWNob0VudiUwQSUwQSUwQWFzeW5jJTIwZGVmJTIwbWFpbigpJTNBJTBBJTIwJTIwJTIwJTIwYXN5bmMlMjB3aXRoJTIwRWNob0VudihiYXNlX3VybCUzRCUyMmh0dHBzJTNBJTJGJTJGb3BlbmVudi1lY2hvLWVudi5oZi5zcGFjZSUyMiklMjBhcyUyMGNsaWVudCUzQSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMGF3YWl0JTIwY2xpZW50LnJlc2V0KCklMEElMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjByZXN1bHQlMjAlM0QlMjBhd2FpdCUyMGNsaWVudC5zdGVwKCUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMENhbGxUb29sQWN0aW9uKCUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMHRvb2xfbmFtZSUzRCUyMmVjaG9fbWVzc2FnZSUyMiUyQyUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMGFyZ3VtZW50cyUzRCU3QiUyMm1lc3NhZ2UlMjIlM0ElMjAlMjJIZWxsbyUyQyUyMFdvcmxkISUyMiU3RCUyQyUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCklMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjApJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwcHJpbnQocmVzdWx0LnJld2FyZCklMEElMEElMEFhc3luY2lvLnJ1bihtYWluKCkp",highlighted:`<span class="hljs-keyword">import</span> asyncio

<span class="hljs-keyword">from</span> echo_env <span class="hljs-keyword">import</span> CallToolAction, EchoEnv


<span class="hljs-keyword">async</span> <span class="hljs-keyword">def</span> <span class="hljs-title function_">main</span>():
    <span class="hljs-keyword">async</span> <span class="hljs-keyword">with</span> EchoEnv(base_url=<span class="hljs-string">&quot;https://openenv-echo-env.hf.space&quot;</span>) <span class="hljs-keyword">as</span> client:
        <span class="hljs-keyword">await</span> client.reset()

        result = <span class="hljs-keyword">await</span> client.step(
            CallToolAction(
                tool_name=<span class="hljs-string">&quot;echo_message&quot;</span>,
                arguments={<span class="hljs-string">&quot;message&quot;</span>: <span class="hljs-string">&quot;Hello, World!&quot;</span>},
            )
        )
        <span class="hljs-built_in">print</span>(result.reward)


asyncio.run(main())`,lang:"python",wrap:!1});var y=n(h,4);s(y,{code:"ZnJvbSUyMGVjaG9fZW52JTIwaW1wb3J0JTIwQ2FsbFRvb2xBY3Rpb24lMkMlMjBFY2hvRW52JTBBJTBBd2l0aCUyMEVjaG9FbnYoYmFzZV91cmwlM0QlMjJodHRwcyUzQSUyRiUyRm9wZW5lbnYtZWNoby1lbnYuaGYuc3BhY2UlMjIpLnN5bmMoKSUyMGFzJTIwY2xpZW50JTNBJTBBJTIwJTIwJTIwJTIwY2xpZW50LnJlc2V0KCklMEElMjAlMjAlMjAlMjByZXN1bHQlMjAlM0QlMjBjbGllbnQuc3RlcCglMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjBDYWxsVG9vbEFjdGlvbiglMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjB0b29sX25hbWUlM0QlMjJlY2hvX21lc3NhZ2UlMjIlMkMlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjBhcmd1bWVudHMlM0QlN0IlMjJtZXNzYWdlJTIyJTNBJTIwJTIySGVsbG8lMkMlMjBXb3JsZCElMjIlN0QlMkMlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjApJTBBJTIwJTIwJTIwJTIwKSUwQSUyMCUyMCUyMCUyMHByaW50KHJlc3VsdC5vYnNlcnZhdGlvbi5yZXN1bHQp",highlighted:`<span class="hljs-keyword">from</span> echo_env <span class="hljs-keyword">import</span> CallToolAction, EchoEnv

<span class="hljs-keyword">with</span> EchoEnv(base_url=<span class="hljs-string">&quot;https://openenv-echo-env.hf.space&quot;</span>).sync() <span class="hljs-keyword">as</span> client:
    client.reset()
    result = client.step(
        CallToolAction(
            tool_name=<span class="hljs-string">&quot;echo_message&quot;</span>,
            arguments={<span class="hljs-string">&quot;message&quot;</span>: <span class="hljs-string">&quot;Hello, World!&quot;</span>},
        )
    )
    <span class="hljs-built_in">print</span>(result.observation.result)`,lang:"python",wrap:!1});var j=n(y,2);e(j,{title:"Use Containers or Local Servers",local:"use-containers-or-local-servers",headingTag:"h2"});var u=n(j,4);s(u,{code:"aW1wb3J0JTIwYXN5bmNpbyUwQSUwQWZyb20lMjBlY2hvX2VudiUyMGltcG9ydCUyMEVjaG9FbnYlMEElMEElMEFhc3luYyUyMGRlZiUyMG1haW4oKSUzQSUwQSUyMCUyMCUyMCUyMGNsaWVudCUyMCUzRCUyMGF3YWl0JTIwRWNob0Vudi5mcm9tX2RvY2tlcl9pbWFnZSglMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjJyZWdpc3RyeS5oZi5zcGFjZSUyRm9wZW5lbnYtZWNoby1lbnYlM0FsYXRlc3QlMjIlMEElMjAlMjAlMjAlMjApJTBBJTIwJTIwJTIwJTIwYXN5bmMlMjB3aXRoJTIwY2xpZW50JTNBJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwcmVzdWx0JTIwJTNEJTIwYXdhaXQlMjBjbGllbnQucmVzZXQoKSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMHByaW50KHJlc3VsdC5vYnNlcnZhdGlvbiklMEElMEElMEFhc3luY2lvLnJ1bihtYWluKCkp",highlighted:`<span class="hljs-keyword">import</span> asyncio

<span class="hljs-keyword">from</span> echo_env <span class="hljs-keyword">import</span> EchoEnv


<span class="hljs-keyword">async</span> <span class="hljs-keyword">def</span> <span class="hljs-title function_">main</span>():
    client = <span class="hljs-keyword">await</span> EchoEnv.from_docker_image(
        <span class="hljs-string">&quot;registry.hf.space/openenv-echo-env:latest&quot;</span>
    )
    <span class="hljs-keyword">async</span> <span class="hljs-keyword">with</span> client:
        result = <span class="hljs-keyword">await</span> client.reset()
        <span class="hljs-built_in">print</span>(result.observation)


asyncio.run(main())`,lang:"python",wrap:!1});var U=n(u,4);s(U,{code:"ZG9ja2VyJTIwcnVuJTIwLWl0JTIwLXAlMjA3ODYwJTNBNzg2MCUyMC0tcGxhdGZvcm0lM0RsaW51eCUyRmFtZDY0JTIwJTVDJTBBJTIwJTIwJTIwJTIwcmVnaXN0cnkuaGYuc3BhY2UlMkZvcGVuZW52LWVjaG8tZW52JTNBbGF0ZXN0",highlighted:`docker run -it -p 7860:7860 --platform=linux/amd64 \\
    registry.hf.space/openenv-echo-env:latest`,lang:"bash",wrap:!1});var w=n(U,2);s(w,{code:"ZnJvbSUyMGVjaG9fZW52JTIwaW1wb3J0JTIwRWNob0VudiUwQSUwQXdpdGglMjBFY2hvRW52KGJhc2VfdXJsJTNEJTIyaHR0cCUzQSUyRiUyRmxvY2FsaG9zdCUzQTc4NjAlMjIpLnN5bmMoKSUyMGFzJTIwY2xpZW50JTNBJTBBJTIwJTIwJTIwJTIwcmVzdWx0JTIwJTNEJTIwY2xpZW50LnJlc2V0KCk=",highlighted:`<span class="hljs-keyword">from</span> echo_env <span class="hljs-keyword">import</span> EchoEnv

<span class="hljs-keyword">with</span> EchoEnv(base_url=<span class="hljs-string">&quot;http://localhost:7860&quot;</span>).sync() <span class="hljs-keyword">as</span> client:
    result = client.reset()`,lang:"python",wrap:!1});var J=n(w,4);s(J,{code:"Y2QlMjBwYXRoJTJGdG8lMkZlY2hvLWVudiUwQXV2JTIwdmVudiUwQXNvdXJjZSUyMC52ZW52JTJGYmluJTJGYWN0aXZhdGUlMEF1diUyMHBpcCUyMGluc3RhbGwlMjAtZSUyMC4lMEF1diUyMHJ1biUyMHNlcnZlciUyMC0taG9zdCUyMDAuMC4wLjAlMjAtLXBvcnQlMjA4MDAw",highlighted:`<span class="hljs-built_in">cd</span> path/to/echo-env
uv venv
<span class="hljs-built_in">source</span> .venv/bin/activate
uv pip install -e .
uv run server --host 0.0.0.0 --port 8000`,lang:"bash",wrap:!1});var v=n(J,2);s(v,{code:"ZnJvbSUyMGVjaG9fZW52JTIwaW1wb3J0JTIwRWNob0VudiUwQSUwQXdpdGglMjBFY2hvRW52KGJhc2VfdXJsJTNEJTIyaHR0cCUzQSUyRiUyRmxvY2FsaG9zdCUzQTgwMDAlMjIpLnN5bmMoKSUyMGFzJTIwY2xpZW50JTNBJTBBJTIwJTIwJTIwJTIwcmVzdWx0JTIwJTNEJTIwY2xpZW50LnJlc2V0KCk=",highlighted:`<span class="hljs-keyword">from</span> echo_env <span class="hljs-keyword">import</span> EchoEnv

<span class="hljs-keyword">with</span> EchoEnv(base_url=<span class="hljs-string">&quot;http://localhost:8000&quot;</span>).sync() <span class="hljs-keyword">as</span> client:
    result = client.reset()`,lang:"python",wrap:!1});var b=n(v,4);e(b,{title:"Next Steps",local:"next-steps",headingTag:"h2"});var A=n(b,4);f(A,{source:"https://github.com/openenv/OpenEnv/blob/main/docs/source/getting-started.md"}),Q(2),g(I,l),N()}export{q as component};
