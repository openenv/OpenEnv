import"../chunks/DsnmJJEf.js";import{i as N,h as Q,C as V,H as e,a as s,E as R,s as k}from"../chunks/Dz7d4_v9.js";import{p as Y,o as x,s as l,f as z,a as W,b as F,c as S,n as q}from"../chunks/BSuxAqoA.js";const X='{"title":"Julia Environment","local":"julia-environment","sections":[{"title":"Quick Start","local":"quick-start","sections":[],"depth":2},{"title":"Building the Docker Image","local":"building-the-docker-image","sections":[],"depth":2},{"title":"Environment Details","local":"environment-details","sections":[{"title":"Action","local":"action","sections":[],"depth":3},{"title":"Observation","local":"observation","sections":[],"depth":3},{"title":"State","local":"state","sections":[],"depth":3},{"title":"Reward Calculation","local":"reward-calculation","sections":[],"depth":3}],"depth":2},{"title":"Features","local":"features","sections":[],"depth":2},{"title":"Advanced Usage","local":"advanced-usage","sections":[{"title":"Connecting to an Existing Server","local":"connecting-to-an-existing-server","sections":[],"depth":3},{"title":"Custom Timeout","local":"custom-timeout","sections":[],"depth":3},{"title":"Running with Docker Directly","local":"running-with-docker-directly","sections":[],"depth":3}],"depth":2},{"title":"Example Code","local":"example-code","sections":[],"depth":2},{"title":"Compatibility","local":"compatibility","sections":[],"depth":2},{"title":"Development","local":"development","sections":[],"depth":2},{"title":"License","local":"license","sections":[],"depth":2}],"depth":1}';var H=S('<meta name="hf:doc:metadata"/>'),D=S("<p></p> <!> <!> <p>A Julia code execution environment that runs Julia code with test result tracking and reward calculation. Perfect for reinforcement learning training with Julia programming tasks.</p> <!> <p>The simplest way to use the Julia environment is through the <code>JuliaEnv</code> class:</p> <!> <p>That’s it! The <code>JuliaEnv.from_docker_image()</code> method handles:</p> <ul><li>Starting the Docker container</li> <li>Waiting for the server to be ready</li> <li>Connecting to the environment</li> <li>Container cleanup when you call <code>close()</code></li></ul> <!> <p>Before using the environment, you need to build the Docker image:</p> <!> <!> <!> <p><strong>JuliaAction</strong>: Contains two fields for Julia code execution</p> <ul><li><code>core_code</code> (str) - The main Julia code to execute (e.g., function definitions)</li> <li><code>test_code</code> (str) - Test code using Julia’s <code>Test</code> module (e.g., <code>@test</code> statements)</li></ul> <!> <p><strong>JuliaObservation</strong>: Contains the execution results and test outcomes</p> <ul><li><code>stdout</code> (str) - Standard output from Julia execution</li> <li><code>stderr</code> (str) - Standard error from Julia execution</li> <li><code>exit_code</code> (int) - Exit code (0 for success, non-zero for errors)</li> <li><code>tests_passed</code> (int) - Number of tests that passed</li> <li><code>tests_failed</code> (int) - Number of tests that failed</li> <li><code>code_compiles</code> (bool) - Whether the core code compiled/executed successfully</li></ul> <!> <p><strong>JuliaState</strong>: Tracks episode execution state</p> <ul><li><code>episode_id</code> (str) - Unique identifier for the episode</li> <li><code>step_count</code> (int) - Number of steps taken in the episode</li> <li><code>last_exit_code</code> (int) - Exit code from the last execution</li> <li><code>last_code_compiles</code> (bool) - Whether the last code compiled successfully</li> <li><code>total_tests_passed</code> (int) - Cumulative number of tests passed in the episode</li> <li><code>total_tests_failed</code> (int) - Cumulative number of tests failed in the episode</li></ul> <!> <p>The environment calculates rewards based on execution success and test results:</p> <ul><li>Code compiles successfully: Base reward</li> <li>Tests pass: Additional reward per test</li> <li>Tests fail or code doesn’t compile: Negative reward</li></ul> <p>See <code>server/julia_transforms.py</code> for detailed reward logic.</p> <!> <ul><li>✅ Execute Julia code in isolated subprocess</li> <li>✅ No network and no server secrets for Julia code: environment allowlist, <code>nobody</code> when the server is root, a non-dumpable server, and a Linux seccomp filter that refuses sockets</li> <li>✅ Parse Julia <code>Test</code> module output (tests passed/failed)</li> <li>✅ Calculate rewards based on execution results and test outcomes</li> <li>✅ Safety transforms for output truncation (prevents excessive output)</li> <li>✅ Docker support for reproducible execution</li> <li>✅ Compatible with GRPO and other RL training frameworks</li></ul> <!> <!> <p>If you already have a Julia environment server running, you can connect directly:</p> <!> <p>Note: When connecting to an existing server, <code>julia_env.close()</code> will NOT stop the server.</p> <!> <p>The Julia environment uses a longer timeout (180s) by default to accommodate Julia compilation and execution:</p> <!> <!> <!> <!> <p>Here’s a more complex example demonstrating test-driven development:</p> <!> <!> <p>This environment is compatible with:</p> <ul><li><strong>torchforge</strong>: For GRPO training with Julia tasks</li> <li><strong>TRL</strong>: Hugging Face’s RL library</li> <li><strong>Any OpenEnv-compatible RL framework</strong></li></ul> <!> <p>For local development without Docker:</p> <!> <!> <p>This source code is licensed under the BSD-style license found in the LICENSE file in the root directory of this source tree.</p> <!> <p></p>",1);function P(E,A){Y(A,!1),x(()=>{new URLSearchParams(window.location.search).get("fw")}),N();var t=D();Q("1ny8w75",G=>{var Z=H();k(Z,"content",X),W(G,Z)});var a=l(z(t),2);V(a,{containerStyle:"float: inline-end; margin-inline-start: 10px; display: inline-flex; position: relative; z-index: 10;"});var n=l(a,2);e(n,{title:"Julia Environment",local:"julia-environment",headingTag:"h1"});var i=l(n,4);e(i,{title:"Quick Start",local:"quick-start",headingTag:"h2"});var o=l(i,4);s(o,{code:"ZnJvbSUyMGVudnMuanVsaWFfZW52JTIwaW1wb3J0JTIwSnVsaWFBY3Rpb24lMkMlMjBKdWxpYUVudiUwQSUwQXRyeSUzQSUwQSUyMCUyMCUyMCUyMCUyMyUyMENyZWF0ZSUyMGVudmlyb25tZW50JTIwZnJvbSUyMERvY2tlciUyMGltYWdlJTBBJTIwJTIwJTIwJTIwanVsaWFfZW52JTIwJTNEJTIwSnVsaWFFbnYuZnJvbV9kb2NrZXJfaW1hZ2UoJTIyanVsaWEtZW52JTNBbGF0ZXN0JTIyKSUwQSUwQSUyMCUyMCUyMCUyMCUyMyUyMFJlc2V0JTBBJTIwJTIwJTIwJTIwcmVzdWx0JTIwJTNEJTIwanVsaWFfZW52LnJlc2V0KCklMEElMjAlMjAlMjAlMjBwcmludChmJTIyUmVzZXQlMjBjb21wbGV0ZSUzQSUyMGV4aXRfY29kZSUzRCU3QnJlc3VsdC5vYnNlcnZhdGlvbi5leGl0X2NvZGUlN0QlMjIpJTBBJTBBJTIwJTIwJTIwJTIwJTIzJTIwRXhlY3V0ZSUyMEp1bGlhJTIwY29kZSUyMHdpdGglMjB0ZXN0cyUwQSUyMCUyMCUyMCUyMGFjdGlvbiUyMCUzRCUyMEp1bGlhQWN0aW9uKCUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMGNvcmVfY29kZSUzRCUyMiUyMiUyMiUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMGZ1bmN0aW9uJTIwbXVsdGlwbHkoYSUyQyUyMGIpJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwcmV0dXJuJTIwYSUyMColMjBiJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwZW5kJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIyJTIyJTIyJTJDJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwdGVzdF9jb2RlJTNEJTIyJTIyJTIyJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwdXNpbmclMjBUZXN0JTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTQwdGVzdCUyMG11bHRpcGx5KDMlMkMlMjA0KSUyMCUzRCUzRCUyMDEyJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTQwdGVzdCUyMG11bHRpcGx5KDUlMkMlMjA2KSUyMCUzRCUzRCUyMDMwJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIyJTIyJTIyJTBBJTIwJTIwJTIwJTIwKSUwQSUwQSUyMCUyMCUyMCUyMHJlc3VsdCUyMCUzRCUyMGp1bGlhX2Vudi5zdGVwKGFjdGlvbiklMEElMjAlMjAlMjAlMjBwcmludChmJTIyVGVzdHMlMjBwYXNzZWQlM0ElMjAlN0JyZXN1bHQub2JzZXJ2YXRpb24udGVzdHNfcGFzc2VkJTdEJTIyKSUwQSUyMCUyMCUyMCUyMHByaW50KGYlMjJUZXN0cyUyMGZhaWxlZCUzQSUyMCU3QnJlc3VsdC5vYnNlcnZhdGlvbi50ZXN0c19mYWlsZWQlN0QlMjIpJTBBJTIwJTIwJTIwJTIwcHJpbnQoZiUyMkNvZGUlMjBjb21waWxlcyUzQSUyMCU3QnJlc3VsdC5vYnNlcnZhdGlvbi5jb2RlX2NvbXBpbGVzJTdEJTIyKSUwQSUyMCUyMCUyMCUyMHByaW50KGYlMjJSZXdhcmQlM0ElMjAlN0JyZXN1bHQucmV3YXJkJTdEJTIyKSUwQSUwQWZpbmFsbHklM0ElMEElMjAlMjAlMjAlMjAlMjMlMjBBbHdheXMlMjBjbGVhbiUyMHVwJTBBJTIwJTIwJTIwJTIwanVsaWFfZW52LmNsb3NlKCk=",highlighted:`<span class="hljs-keyword">from</span> envs.julia_env <span class="hljs-keyword">import</span> JuliaAction, JuliaEnv

<span class="hljs-keyword">try</span>:
    <span class="hljs-comment"># Create environment from Docker image</span>
    julia_env = JuliaEnv.from_docker_image(<span class="hljs-string">&quot;julia-env:latest&quot;</span>)

    <span class="hljs-comment"># Reset</span>
    result = julia_env.reset()
    <span class="hljs-built_in">print</span>(<span class="hljs-string">f&quot;Reset complete: exit_code=<span class="hljs-subst">{result.observation.exit_code}</span>&quot;</span>)

    <span class="hljs-comment"># Execute Julia code with tests</span>
    action = JuliaAction(
        core_code=<span class="hljs-string">&quot;&quot;&quot;
        function multiply(a, b)
            return a * b
        end
        &quot;&quot;&quot;</span>,
        test_code=<span class="hljs-string">&quot;&quot;&quot;
        using Test
        @test multiply(3, 4) == 12
        @test multiply(5, 6) == 30
        &quot;&quot;&quot;</span>
    )

    result = julia_env.step(action)
    <span class="hljs-built_in">print</span>(<span class="hljs-string">f&quot;Tests passed: <span class="hljs-subst">{result.observation.tests_passed}</span>&quot;</span>)
    <span class="hljs-built_in">print</span>(<span class="hljs-string">f&quot;Tests failed: <span class="hljs-subst">{result.observation.tests_failed}</span>&quot;</span>)
    <span class="hljs-built_in">print</span>(<span class="hljs-string">f&quot;Code compiles: <span class="hljs-subst">{result.observation.code_compiles}</span>&quot;</span>)
    <span class="hljs-built_in">print</span>(<span class="hljs-string">f&quot;Reward: <span class="hljs-subst">{result.reward}</span>&quot;</span>)

<span class="hljs-keyword">finally</span>:
    <span class="hljs-comment"># Always clean up</span>
    julia_env.close()`,lang:"python",wrap:!1});var c=l(o,6);e(c,{title:"Building the Docker Image",local:"building-the-docker-image",headingTag:"h2"});var J=l(c,4);s(J,{code:"JTIzJTIwRnJvbSUyMHRoZSUyMGp1bGlhX2VudiUyMGRpcmVjdG9yeSUwQWNkJTIwZW52cyUyRmp1bGlhX2VudiUwQWRvY2tlciUyMGJ1aWxkJTIwLXQlMjBqdWxpYS1lbnYlM0FsYXRlc3QlMjAtZiUyMHNlcnZlciUyRkRvY2tlcmZpbGUlMjAu",highlighted:`<span class="hljs-comment"># From the julia_env directory</span>
<span class="hljs-built_in">cd</span> envs/julia_env
docker build -t julia-env:latest -f server/Dockerfile .`,lang:"bash",wrap:!1});var M=l(J,2);e(M,{title:"Environment Details",local:"environment-details",headingTag:"h2"});var d=l(M,2);e(d,{title:"Action",local:"action",headingTag:"h3"});var r=l(d,6);e(r,{title:"Observation",local:"observation",headingTag:"h3"});var T=l(r,6);e(T,{title:"State",local:"state",headingTag:"h3"});var p=l(T,6);e(p,{title:"Reward Calculation",local:"reward-calculation",headingTag:"h3"});var w=l(p,8);e(w,{title:"Features",local:"features",headingTag:"h2"});var u=l(w,4);e(u,{title:"Advanced Usage",local:"advanced-usage",headingTag:"h2"});var y=l(u,2);e(y,{title:"Connecting to an Existing Server",local:"connecting-to-an-existing-server",headingTag:"h3"});var U=l(y,4);s(U,{code:"ZnJvbSUyMGVudnMuanVsaWFfZW52JTIwaW1wb3J0JTIwSnVsaWFFbnYlMkMlMjBKdWxpYUFjdGlvbiUwQSUwQSUyMyUyMENvbm5lY3QlMjB0byUyMGV4aXN0aW5nJTIwc2VydmVyJTBBanVsaWFfZW52JTIwJTNEJTIwSnVsaWFFbnYoYmFzZV91cmwlM0QlMjJodHRwJTNBJTJGJTJGbG9jYWxob3N0JTNBODAwMCUyMiklMEElMEElMjMlMjBVc2UlMjBhcyUyMG5vcm1hbCUwQXJlc3VsdCUyMCUzRCUyMGp1bGlhX2Vudi5yZXNldCgpJTBBcmVzdWx0JTIwJTNEJTIwanVsaWFfZW52LnN0ZXAoSnVsaWFBY3Rpb24oJTBBJTIwJTIwJTIwJTIwY29yZV9jb2RlJTNEJTIycHJpbnRsbigyJTIwJTJCJTIwMiklMjIlMkMlMEElMjAlMjAlMjAlMjB0ZXN0X2NvZGUlM0QlMjIlMjIlMEEpKQ==",highlighted:`<span class="hljs-keyword">from</span> envs.julia_env <span class="hljs-keyword">import</span> JuliaEnv, JuliaAction

<span class="hljs-comment"># Connect to existing server</span>
julia_env = JuliaEnv(base_url=<span class="hljs-string">&quot;http://localhost:8000&quot;</span>)

<span class="hljs-comment"># Use as normal</span>
result = julia_env.reset()
result = julia_env.step(JuliaAction(
    core_code=<span class="hljs-string">&quot;println(2 + 2)&quot;</span>,
    test_code=<span class="hljs-string">&quot;&quot;</span>
))`,lang:"python",wrap:!1});var I=l(U,4);e(I,{title:"Custom Timeout",local:"custom-timeout",headingTag:"h3"});var h=l(I,4);s(h,{code:"JTIzJTIwQ3VzdG9tJTIwdGltZW91dCUyMChpbiUyMHNlY29uZHMpJTBBanVsaWFfZW52JTIwJTNEJTIwSnVsaWFFbnYoYmFzZV91cmwlM0QlMjJodHRwJTNBJTJGJTJGbG9jYWxob3N0JTNBODAwMCUyMiUyQyUyMG1lc3NhZ2VfdGltZW91dF9zJTNEMzAwLjAp",highlighted:`<span class="hljs-comment"># Custom timeout (in seconds)</span>
julia_env = JuliaEnv(base_url=<span class="hljs-string">&quot;http://localhost:8000&quot;</span>, message_timeout_s=<span class="hljs-number">300.0</span>)`,lang:"python",wrap:!1});var j=l(h,2);e(j,{title:"Running with Docker Directly",local:"running-with-docker-directly",headingTag:"h3"});var m=l(j,2);s(m,{code:"JTIzJTIwUnVuJTIwd2l0aCUyMGRlZmF1bHQlMjBzZXR0aW5ncyUyMChwb3J0JTIwODAwMCklMEFkb2NrZXIlMjBydW4lMjAtZCUyMC1wJTIwODAwMCUzQTgwMDAlMjAtLW5hbWUlMjBqdWxpYS1lbnYlMjBqdWxpYS1lbnYlM0FsYXRlc3QlMEElMEElMjMlMjBDaGVjayUyMGhlYWx0aCUwQWN1cmwlMjBodHRwJTNBJTJGJTJGbG9jYWxob3N0JTNBODAwMCUyRmhlYWx0aCUwQSUwQSUyMyUyMFZpZXclMjBsb2dzJTBBZG9ja2VyJTIwbG9ncyUyMC1mJTIwanVsaWEtZW52",highlighted:`<span class="hljs-comment"># Run with default settings (port 8000)</span>
docker run -d -p 8000:8000 --name julia-env julia-env:latest

<span class="hljs-comment"># Check health</span>
curl http://localhost:8000/health

<span class="hljs-comment"># View logs</span>
docker logs -f julia-env`,lang:"bash",wrap:!1});var b=l(m,2);e(b,{title:"Example Code",local:"example-code",headingTag:"h2"});var C=l(b,4);s(C,{code:"ZnJvbSUyMGVudnMuanVsaWFfZW52JTIwaW1wb3J0JTIwSnVsaWFBY3Rpb24lMkMlMjBKdWxpYUVudiUwQSUwQWp1bGlhX2VudiUyMCUzRCUyMEp1bGlhRW52LmZyb21fZG9ja2VyX2ltYWdlKCUyMmp1bGlhLWVudiUzQWxhdGVzdCUyMiklMEElMEF0cnklM0ElMEElMjAlMjAlMjAlMjAlMjMlMjBSZXNldCUyMHRoZSUyMGVudmlyb25tZW50JTBBJTIwJTIwJTIwJTIwanVsaWFfZW52LnJlc2V0KCklMEElMEElMjAlMjAlMjAlMjAlMjMlMjBTdGVwJTIwMSUzQSUyMERlZmluZSUyMGElMjBmdW5jdGlvbiUyMHdpdGglMjB0ZXN0cyUwQSUyMCUyMCUyMCUyMGFjdGlvbiUyMCUzRCUyMEp1bGlhQWN0aW9uKCUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMGNvcmVfY29kZSUzRCUyMiUyMiUyMiUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMGZ1bmN0aW9uJTIwZmlib25hY2NpKG4pJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwaWYlMjBuJTIwJTNDJTNEJTIwMSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMHJldHVybiUyMG4lMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjBlbmQlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjByZXR1cm4lMjBmaWJvbmFjY2kobi0xKSUyMCUyQiUyMGZpYm9uYWNjaShuLTIpJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwZW5kJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIyJTIyJTIyJTJDJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwdGVzdF9jb2RlJTNEJTIyJTIyJTIyJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwdXNpbmclMjBUZXN0JTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTQwdGVzdCUyMGZpYm9uYWNjaSgwKSUyMCUzRCUzRCUyMDAlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlNDB0ZXN0JTIwZmlib25hY2NpKDEpJTIwJTNEJTNEJTIwMSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCU0MHRlc3QlMjBmaWJvbmFjY2koNSklMjAlM0QlM0QlMjA1JTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTQwdGVzdCUyMGZpYm9uYWNjaSgxMCklMjAlM0QlM0QlMjA1NSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMiUyMiUyMiUwQSUyMCUyMCUyMCUyMCklMEElMEElMjAlMjAlMjAlMjByZXN1bHQlMjAlM0QlMjBqdWxpYV9lbnYuc3RlcChhY3Rpb24pJTBBJTIwJTIwJTIwJTIwcHJpbnQoZiUyMlN0ZXAlMjAxJTIwLSUyMFRlc3RzJTIwcGFzc2VkJTNBJTIwJTdCcmVzdWx0Lm9ic2VydmF0aW9uLnRlc3RzX3Bhc3NlZCU3RCUyRjQlMjIpJTBBJTIwJTIwJTIwJTIwcHJpbnQoZiUyMlN0ZXAlMjAxJTIwLSUyMFJld2FyZCUzQSUyMCU3QnJlc3VsdC5yZXdhcmQlN0QlMjIpJTBBJTBBJTIwJTIwJTIwJTIwJTIzJTIwR2V0JTIwY3VycmVudCUyMHN0YXRlJTBBJTIwJTIwJTIwJTIwc3RhdGUlMjAlM0QlMjBqdWxpYV9lbnYuc3RhdGUoKSUwQSUyMCUyMCUyMCUyMHByaW50KGYlMjJUb3RhbCUyMHRlc3RzJTIwcGFzc2VkJTIwc28lMjBmYXIlM0ElMjAlN0JzdGF0ZS50b3RhbF90ZXN0c19wYXNzZWQlN0QlMjIpJTBBJTIwJTIwJTIwJTIwcHJpbnQoZiUyMlRvdGFsJTIwdGVzdHMlMjBmYWlsZWQlMjBzbyUyMGZhciUzQSUyMCU3QnN0YXRlLnRvdGFsX3Rlc3RzX2ZhaWxlZCU3RCUyMiklMEElMEFmaW5hbGx5JTNBJTBBJTIwJTIwJTIwJTIwanVsaWFfZW52LmNsb3NlKCk=",highlighted:`<span class="hljs-keyword">from</span> envs.julia_env <span class="hljs-keyword">import</span> JuliaAction, JuliaEnv

julia_env = JuliaEnv.from_docker_image(<span class="hljs-string">&quot;julia-env:latest&quot;</span>)

<span class="hljs-keyword">try</span>:
    <span class="hljs-comment"># Reset the environment</span>
    julia_env.reset()

    <span class="hljs-comment"># Step 1: Define a function with tests</span>
    action = JuliaAction(
        core_code=<span class="hljs-string">&quot;&quot;&quot;
        function fibonacci(n)
            if n &lt;= 1
                return n
            end
            return fibonacci(n-1) + fibonacci(n-2)
        end
        &quot;&quot;&quot;</span>,
        test_code=<span class="hljs-string">&quot;&quot;&quot;
        using Test
        @test fibonacci(0) == 0
        @test fibonacci(1) == 1
        @test fibonacci(5) == 5
        @test fibonacci(10) == 55
        &quot;&quot;&quot;</span>
    )

    result = julia_env.step(action)
    <span class="hljs-built_in">print</span>(<span class="hljs-string">f&quot;Step 1 - Tests passed: <span class="hljs-subst">{result.observation.tests_passed}</span>/4&quot;</span>)
    <span class="hljs-built_in">print</span>(<span class="hljs-string">f&quot;Step 1 - Reward: <span class="hljs-subst">{result.reward}</span>&quot;</span>)

    <span class="hljs-comment"># Get current state</span>
    state = julia_env.state()
    <span class="hljs-built_in">print</span>(<span class="hljs-string">f&quot;Total tests passed so far: <span class="hljs-subst">{state.total_tests_passed}</span>&quot;</span>)
    <span class="hljs-built_in">print</span>(<span class="hljs-string">f&quot;Total tests failed so far: <span class="hljs-subst">{state.total_tests_failed}</span>&quot;</span>)

<span class="hljs-keyword">finally</span>:
    julia_env.close()`,lang:"python",wrap:!1});var v=l(C,2);e(v,{title:"Compatibility",local:"compatibility",headingTag:"h2"});var g=l(v,6);e(g,{title:"Development",local:"development",headingTag:"h2"});var B=l(g,4);s(B,{code:"JTIzJTIwSW5zdGFsbCUyMEp1bGlhJTIwMS4xMCUyQiUyMGZyb20lMjBodHRwcyUzQSUyRiUyRmp1bGlhbGFuZy5vcmclMkZkb3dubG9hZHMlMkYlMEElMEElMjMlMjBJbnN0YWxsJTIwUHl0aG9uJTIwZGVwZW5kZW5jaWVzJTBBY2QlMjBlbnZzJTJGanVsaWFfZW52JTBBcGlwJTIwaW5zdGFsbCUyMC1lJTIwLiUwQSUwQSUyMyUyMFJ1biUyMHRoZSUyMHNlcnZlciUyMGxvY2FsbHklMEF1dmljb3JuJTIwanVsaWFfZW52LnNlcnZlci5hcHAlM0FhcHAlMjAtLWhvc3QlMjAwLjAuMC4wJTIwLS1wb3J0JTIwODAwMA==",highlighted:`<span class="hljs-comment"># Install Julia 1.10+ from https://julialang.org/downloads/</span>

<span class="hljs-comment"># Install Python dependencies</span>
<span class="hljs-built_in">cd</span> envs/julia_env
pip install -e .

<span class="hljs-comment"># Run the server locally</span>
uvicorn julia_env.server.app:app --host 0.0.0.0 --port 8000`,lang:"bash",wrap:!1});var f=l(B,2);e(f,{title:"License",local:"license",headingTag:"h2"});var _=l(f,4);R(_,{source:"https://github.com/openenv/OpenEnv/blob/main/docs/source/environments/julia.md"}),q(2),W(E,t),F()}export{P as component};
