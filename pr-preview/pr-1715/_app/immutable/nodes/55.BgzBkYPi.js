import"../chunks/DsnmJJEf.js";import{i as m,h as I,C as v,H as n,a,E as g,s as b}from"../chunks/Dz7d4_v9.js";import{p as C,o as W,s,f as B,a as j,b as A,c as u,n as k}from"../chunks/BSuxAqoA.js";const V='{"title":"Async vs Sync","local":"async-vs-sync","sections":[{"title":"When to Use Each","local":"when-to-use-each","sections":[],"depth":2},{"title":"Sync Usage","local":"sync-usage","sections":[],"depth":2},{"title":"Async Usage (Recommended)","local":"async-usage-recommended","sections":[],"depth":2},{"title":"Parallel Environments","local":"parallel-environments","sections":[],"depth":2},{"title":"Next Steps","local":"next-steps","sections":[],"depth":2}],"depth":1}';var Q=u('<meta name="hf:doc:metadata"/>'),S=u('<p></p> <!> <!> <p>OpenEnv supports both asynchronous and synchronous usage patterns.</p> <!> <table><thead><tr><th>Pattern</th><th>Best For</th><th>Performance</th></tr></thead><tbody><tr><td><strong>Async</strong></td><td>Production, multiple environments, high throughput</td><td>⚡ Best</td></tr><tr><td><strong>Sync</strong></td><td>Scripts, notebooks, quick experiments</td><td>🐢 Good enough</td></tr></tbody></table> <!> <p>For simple scripts and notebooks, use the <code>.sync()</code> wrapper:</p> <!> <!> <p>For production and parallel environments, use async directly:</p> <!> <!> <p>Run multiple environments concurrently:</p> <!> <!> <ul><li><a href="training">Training with OpenEnv</a> - Using async with RL training</li> <li><a href="auto-discovery">Auto-Discovery</a> - Discover and load environments</li></ul> <!> <p></p>',1);function X(J,U){C(U,!1),W(()=>{new URLSearchParams(window.location.search).get("fw")}),m();var l=S();I("1j1lfq9",M=>{var h=Q();b(h,"content",V),j(M,h)});var e=s(B(l),2);v(e,{containerStyle:"float: inline-end; margin-inline-start: 10px; display: inline-flex; position: relative; z-index: 10;"});var t=s(e,2);n(t,{title:"Async vs Sync",local:"async-vs-sync",headingTag:"h1"});var o=s(t,4);n(o,{title:"When to Use Each",local:"when-to-use-each",headingTag:"h2"});var c=s(o,4);n(c,{title:"Sync Usage",local:"sync-usage",headingTag:"h2"});var p=s(c,4);a(p,{code:"ZnJvbSUyMG9wZW5lbnYlMjBpbXBvcnQlMjBBdXRvRW52JTBBJTBBZW52JTIwJTNEJTIwQXV0b0Vudi5mcm9tX2VudiglMjJlY2hvJTIyKSUwQSUwQXdpdGglMjBlbnYuc3luYygpJTIwYXMlMjBjbGllbnQlM0ElMEElMjAlMjAlMjAlMjByZXN1bHQlMjAlM0QlMjBjbGllbnQucmVzZXQoKSUwQSUyMCUyMCUyMCUyMHJlc3VsdCUyMCUzRCUyMGNsaWVudC5zdGVwKGFjdGlvbik=",highlighted:`<span class="hljs-keyword">from</span> openenv <span class="hljs-keyword">import</span> AutoEnv

env = AutoEnv.from_env(<span class="hljs-string">&quot;echo&quot;</span>)

<span class="hljs-keyword">with</span> env.sync() <span class="hljs-keyword">as</span> client:
    result = client.reset()
    result = client.step(action)`,lang:"python",wrap:!1});var r=s(p,2);n(r,{title:"Async Usage (Recommended)",local:"async-usage-recommended",headingTag:"h2"});var i=s(r,4);a(i,{code:"aW1wb3J0JTIwYXN5bmNpbyUwQWZyb20lMjBvcGVuZW52JTIwaW1wb3J0JTIwQXV0b0VudiUwQSUwQWFzeW5jJTIwZGVmJTIwbWFpbigpJTNBJTBBJTIwJTIwJTIwJTIwZW52JTIwJTNEJTIwQXV0b0Vudi5mcm9tX2VudiglMjJlY2hvJTIyKSUwQSUwQSUyMCUyMCUyMCUyMGFzeW5jJTIwd2l0aCUyMGVudiUyMGFzJTIwY2xpZW50JTNBJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwcmVzdWx0JTIwJTNEJTIwYXdhaXQlMjBjbGllbnQucmVzZXQoKSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMHJlc3VsdCUyMCUzRCUyMGF3YWl0JTIwY2xpZW50LnN0ZXAoYWN0aW9uKSUwQSUwQWFzeW5jaW8ucnVuKG1haW4oKSk=",highlighted:`<span class="hljs-keyword">import</span> asyncio
<span class="hljs-keyword">from</span> openenv <span class="hljs-keyword">import</span> AutoEnv

<span class="hljs-keyword">async</span> <span class="hljs-keyword">def</span> <span class="hljs-title function_">main</span>():
    env = AutoEnv.from_env(<span class="hljs-string">&quot;echo&quot;</span>)

    <span class="hljs-keyword">async</span> <span class="hljs-keyword">with</span> env <span class="hljs-keyword">as</span> client:
        result = <span class="hljs-keyword">await</span> client.reset()
        result = <span class="hljs-keyword">await</span> client.step(action)

asyncio.run(main())`,lang:"python",wrap:!1});var y=s(i,2);n(y,{title:"Parallel Environments",local:"parallel-environments",headingTag:"h2"});var d=s(y,4);a(d,{code:"aW1wb3J0JTIwYXN5bmNpbyUwQWZyb20lMjBvcGVuZW52JTIwaW1wb3J0JTIwQXV0b0VudiUwQSUwQWFzeW5jJTIwZGVmJTIwcnVuX2VwaXNvZGUoZW52X25hbWUlM0ElMjBzdHIpJTNBJTBBJTIwJTIwJTIwJTIwZW52JTIwJTNEJTIwQXV0b0Vudi5mcm9tX2VudihlbnZfbmFtZSklMEElMjAlMjAlMjAlMjBhc3luYyUyMHdpdGglMjBlbnYlMjBhcyUyMGNsaWVudCUzQSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMHJlc3VsdCUyMCUzRCUyMGF3YWl0JTIwY2xpZW50LnJlc2V0KCklMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjMlMjAuLi4lMjBydW4lMjBlcGlzb2RlJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwcmV0dXJuJTIwcmVzdWx0JTBBJTBBYXN5bmMlMjBkZWYlMjBtYWluKCklM0ElMEElMjAlMjAlMjAlMjAlMjMlMjBSdW4lMjA0JTIwZW52aXJvbm1lbnRzJTIwaW4lMjBwYXJhbGxlbCUwQSUyMCUyMCUyMCUyMHJlc3VsdHMlMjAlM0QlMjBhd2FpdCUyMGFzeW5jaW8uZ2F0aGVyKCUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMHJ1bl9lcGlzb2RlKCUyMmVjaG8lMjIpJTJDJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwcnVuX2VwaXNvZGUoJTIyZWNobyUyMiklMkMlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjBydW5fZXBpc29kZSglMjJlY2hvJTIyKSUyQyUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMHJ1bl9lcGlzb2RlKCUyMmVjaG8lMjIpJTJDJTBBJTIwJTIwJTIwJTIwKSUwQSUwQWFzeW5jaW8ucnVuKG1haW4oKSk=",highlighted:`<span class="hljs-keyword">import</span> asyncio
<span class="hljs-keyword">from</span> openenv <span class="hljs-keyword">import</span> AutoEnv

<span class="hljs-keyword">async</span> <span class="hljs-keyword">def</span> <span class="hljs-title function_">run_episode</span>(<span class="hljs-params">env_name: <span class="hljs-built_in">str</span></span>):
    env = AutoEnv.from_env(env_name)
    <span class="hljs-keyword">async</span> <span class="hljs-keyword">with</span> env <span class="hljs-keyword">as</span> client:
        result = <span class="hljs-keyword">await</span> client.reset()
        <span class="hljs-comment"># ... run episode</span>
        <span class="hljs-keyword">return</span> result

<span class="hljs-keyword">async</span> <span class="hljs-keyword">def</span> <span class="hljs-title function_">main</span>():
    <span class="hljs-comment"># Run 4 environments in parallel</span>
    results = <span class="hljs-keyword">await</span> asyncio.gather(
        run_episode(<span class="hljs-string">&quot;echo&quot;</span>),
        run_episode(<span class="hljs-string">&quot;echo&quot;</span>),
        run_episode(<span class="hljs-string">&quot;echo&quot;</span>),
        run_episode(<span class="hljs-string">&quot;echo&quot;</span>),
    )

asyncio.run(main())`,lang:"python",wrap:!1});var w=s(d,2);n(w,{title:"Next Steps",local:"next-steps",headingTag:"h2"});var T=s(w,4);g(T,{source:"https://github.com/openenv/OpenEnv/blob/main/docs/source/guides/async-sync.md"}),k(2),j(J,l),A()}export{X as component};
