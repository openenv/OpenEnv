<!-- openenv-source: helium_browser_env -->
# Helium Browser Environment

This example shows the main pieces of an OpenEnv environment by wrapping a
Chromium browser. A client sends one typed action at a time. The environment
performs it with Helium and Selenium, then returns a typed observation with a
fresh screenshot and the current URL.

The browser is useful here because the boundary is easy to see: OpenEnv owns
the `reset`/`step` protocol, while the environment owns all browser-specific
behavior. The example does not include an agent, so the OpenEnv parts stay in
focus.

![illustrated](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/helium_demo.png)

## How to run the browser agent

The actual browser loop is [browser.py](https://github.com/huggingface/OpenEnv/blob/main/envs/helium_browser_env/browser.py). You can run it with following steps. First publish the environment. Make sure to authenticate yourself with a write token.

```bash
hf auth login
openenv push envs/helium_browser_env \
  --repo-id YOUR_USERNAME/helium-browser-env
```

The agent uses the token saved by `hf auth login`, or `HF_TOKEN` if it is set.
Make sure that token can access your inference endpoint. Then deploy the model
and provide your endpoint.

```bash
export HELIUM_BROWSER_IMAGE='hf.co/spaces/YOUR_USERNAME/helium-browser-env'
export BASE_URL='https://YOUR_ENDPOINT.endpoints.huggingface.cloud/v1'
export MODEL_ID="HF-MODEL-ID" # replace with the model you run endpoint with
```

Run browser.py loop inside Hugging Face Sandboxes.

```python
import os
import runpy
from openenv.core.containers.runtime.hf_sandbox_provider import HFSandboxProvider

with HFSandboxProvider(
    image=os.environ["HELIUM_BROWSER_IMAGE"],
    flavor="cpu-basic",
    env_vars={"BROWSER_SANDBOX": "hf"},
) as provider:
    url = provider.start_container()
    provider.wait_for_ready(url, timeout_s=300)
    os.environ["BROWSER_BASE_URL"] = url
    runpy.run_path("envs/helium_browser_env/browser.py", run_name="__main__")
```

Rest of this tutorial goes through OpenEnv basics and how to build a multimodal browser agent.

## How browser agents work

A natively multimodal browser agent operates in a loop:

1. **Observe:** receive the current page, as a screenshot.
2. **Decide:** choose one small action that moves the task forward.
3. **Act:** click, type, press a key, scroll, or navigate.
4. **Observe again:** inspect the page after the action instead of assuming it
   worked.
5. **Stop:** finish when the task is complete or the episode reaches a limit.

This example uses screenshots as the main observation. The agent sees the same
800×600 browser image that the environment uses for coordinates, then returns
one structured action. A click such as `x=320, y=180` means a click position in that
image; it is not a CSS selector or a reference to the page's HTML.

One action per step matters because web pages change. A click may open a menu,
show a cookie banner, navigate to another page, or fail. Returning a fresh
screenshot and URL after every action lets the caller react to what actually
happened. It also produces a clear trajectory: observation, action, next
observation.

That loop explains the design choices used later in this example:

- **Typed actions** give OpenEnv a small, validated set of actions for controlling
  the browser.
- **A fixed viewport** keeps screenshot pixels and click coordinates aligned.
- **Real pointer events** let clicks reach visual controls such as overlays and
  canvases without exposing DOM selectors to the agent.
- **An isolated browser** keeps temporary profiles and untrusted pages away
  from the user's personal browser session.
- **Step limits and `done`** give every browser episode a definite boundary.

The agent or policy that chooses actions is intentionally outside this
environment. OpenEnv provides the repeatable interaction contract; the caller
decides how to turn an observation into the next `BrowserAction`.

## Tasks

Websites always keep on changing so using a regular dataset doesn't work. You can try the samples [in this dataset](https://huggingface.co/datasets/merve/browser-tasks) as they are created late 2026. They are divided into three:

- **Information:** getting an information from a website by browsing it.
- **Navigation:** navigating a specific part of the website.
- **Interaction:** clicking, filtering etc.

## The OpenEnv pieces

An OpenEnv environment has four small parts in this example:

| OpenEnv concept | Implementation | Responsibility |
| --- | --- | --- |
| Data contract | `BrowserAction`, `BrowserObservation`, `BrowserState` | Defines the validated objects that cross the client/server boundary. |
| Environment | `BrowserEnvironment` | Implements the episode lifecycle: `reset`, `step`, `state`, and `close`. |
| Server | `create_app(...)` in `server/app.py` | Turns the environment class and data models into an HTTP/WebSocket service. |
| Client | `BrowserClient` | Serializes actions and reconstructs typed observations and step results. |

These parts separate domain logic from transport. `BrowserEnvironment` knows
how to drive a browser, but it does not define HTTP routes. `BrowserClient`
knows how to exchange OpenEnv messages, but it does not import Helium or
Selenium.

### 1. Define the data contract

`models.py` subclasses OpenEnv's three base models with the actions an agent can take: "click", "type", "key", "scroll", "back", "wait", "finish". x and y are clicking coordinates, dy is change in vertical direction (scroll dy pixels), text is text to be typed on e.g. search bars.

```python
class BrowserAction(Action):
    op: Literal["click", "type", "key", "scroll", "back", "wait", "finish"]
    x: int | None = Field(default=None, ge=0, lt=800)
    y: int | None = Field(default=None, ge=0, lt=600)
    text: str = Field(default="", max_length=2000)
    dy: int = Field(default=0, ge=-1200, le=1200)


class BrowserObservation(Observation):
    screenshot: str
    url: str
    error: str = ""


class BrowserState(State):
    pass
```

Actions are taken by the agent and observations are the result of the actions in the environment. On the other hand, OpenEnv uses state to report episode
metadata such as the episode ID and step count, without adding those fields to
every browser observation.

| Operation | Fields | What it does |
| --- | --- | --- |
| `click` | `x`, `y` | Clicks one point in the 800×600 browser viewport. |
| `type` | `text` | Types into the currently focused element. |
| `key` | `text` | Presses a supported key such as `ENTER`, `TAB`, or `CTRL+A`. |
| `scroll` | `dy` | Scrolls down for positive values and up for negative values. |
| `back` | none | Goes back in browser history. |
| `wait` | none | Waits for the page without performing another action. |
| `finish` | optional `text` | Ends the episode. |

Click coordinates are browser pixels: `x` is from 0 to 799 and `y` is from 0
to 599. Scroll values are limited to the range -1200 to 1200.

Supported keys are `ENTER`, `TAB`, `ESCAPE`, `BACKSPACE`, `CTRL+A`, `UP`,
`DOWN`, `LEFT`, and `RIGHT`.

Every observation contains:

- `screenshot`: the current browser window as a base64-encoded PNG;
- `url`: the current page URL;
- `error`: an empty string or a short error code;
- `done`: whether the episode has ended.

The OpenEnv step result carries a reward of `0` because this example does not
score browser actions.

The episode ends when it receives `finish`, reaches the configured step limit,
encounters a renderer timeout, or detects a supported bot-verification page.
An action error is reported by exception class name so the caller can inspect
the next screenshot and decide what to do.

### 2. Implement the environment lifecycle

`BrowserEnvironment` subclasses OpenEnv's `Environment`. Its methods have the
same roles they would have in a game, simulator, or other environment:

- `reset(...)` starts a new episode and returns its first observation. Here it
  creates a temporary browser profile, opens `start_url`, and takes a
  screenshot. Without `start_url` it opens `BROWSER_START_URL`
  (default `https://huggingface.co`), so the `/web` playground's Reset works.
- `step(action)` applies exactly one validated action and returns the next
  observation.
- `state` exposes the current `BrowserState`, including OpenEnv's episode ID
  and step count.
- `close()` releases the resources owned by the episode.

This is the Gym-like core of OpenEnv. Browser setup and Helium calls are
ordinary implementation details behind that interface.

### 3. Turn it into a server

`server/app.py` passes the environment class and its wire types to `create_app`:

```python
app = create_app(
    BrowserEnvironment,
    BrowserAction,
    BrowserObservation,
    env_name="helium_browser_env",
    max_concurrent_envs=1,
)
```

`create_app` supplies the FastAPI application and the standard OpenEnv
endpoints, including health, reset, step, state, and WebSocket access. The
environment code only implements the lifecycle; it does not duplicate those
routes.

The environment class is passed instead of a pre-built instance so OpenEnv can
own its lifecycle. Concurrency is limited to one because Helium stores the
active Selenium driver globally and Selenium drivers are not thread-safe.

### 4. Add the typed client

`BrowserClient` subclasses OpenEnv's generic `EnvClient` with the action,
observation, and state types:

```python
class BrowserClient(EnvClient[BrowserAction, BrowserObservation, BrowserState]):
    def _step_payload(self, action):
        return action.model_dump()

    def _parse_result(self, payload): ...

    def _parse_state(self, payload):
        return BrowserState.model_validate(payload)
```

The base client handles the connection and the common `reset`, `step`, and
state operations. This subclass only teaches it how this environment's models
map to JSON. Callers therefore work with `BrowserAction` and
`BrowserObservation`, not untyped dictionaries.

## How it fits together

```text
BrowserClient -> OpenEnv server -> BrowserEnvironment -> Helium/Selenium -> Chromium
BrowserClient <- screenshot, URL, error, done <- BrowserEnvironment
```

For a step, `BrowserClient` serializes a `BrowserAction`. OpenEnv validates it,
calls `BrowserEnvironment.step()`, and serializes the returned
`BrowserObservation`. The client then rebuilds a typed `StepResult` containing
the observation, reward, and `done` flag.

## The OpenEnv loop

Here's how OpenEnv loop works: once the model server is running, reset the browser to a URL and send one action at a time.

```python
from helium_browser_env.client import BrowserClient
from helium_browser_env.models import BrowserAction

with BrowserClient(base_url="http://localhost:8000").sync() as browser:
    result = browser.reset(start_url="https://example.com")
    observation = result.observation

    result = browser.step(BrowserAction(op="scroll", dy=500))
    observation = result.observation

    print(observation.url)
    print(observation.error)
    print(result.done, result.reward)
    print(observation.screenshot)  # Base64-encoded PNG
```

It starts a fresh browser profile, opens the page, and returns the first
observation. `step` performs one action and returns an OpenEnv `StepResult`.
The result groups the next observation with the reward and episode termination
flag.

## Notes

The environment is designed to run inside its browser sandbox, not against a
personal desktop browser. Chromium uses a temporary profile for each episode.
Downloads and password storage are disabled, and a reset rejects URLs that
contain a username or password.

Web pages are untrusted input. Do not enter credentials or sensitive data, and
expect some sites to show automation challenges. This example provides the
browser interaction loop; it does not provide tasks, scoring, or a reward
function.
