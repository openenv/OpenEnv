# Customizing the Web UI

The web UI is off by default. When `ENABLE_WEB_INTERFACE=true` (which `openenv push` sets for Spaces), the server serves a default Gradio app at `/web` that follows the loop an agent runs: reset, take an action (for MCP environments, pick a tool and fill its arguments), read the result, the episode so far, and the same call in Python. Environment authors can draw the environment's state and offer one-click actions with two optional methods, or **add** a whole custom tab with a Gradio builder.

## Draw the state and offer one-click actions

Override these `Environment` methods to make the default playground visual. Both receive the serialized observation, as `reset()` and `step()` return it, and both are optional:

- `render_web(observation)` returns HTML that draws it (a board, a page, a plot), shown next to the action controls. The default returns `None` and the playground lists the observation's fields.
- `web_actions(observation)` returns `(label, action)` pairs shown as buttons, where each action is the dict `step()` receives. Clicking one runs that step. The default returns `[]`.

For example, `openspiel_env` draws the Catch board and offers the legal moves:

```python
class OpenSpielEnvironment(Environment):
    def web_actions(self, observation):
        names = {0: "left", 1: "stay", 2: "right"}
        return [(f"{a} · {names[a]}", {"action_id": a}) for a in observation["legal_actions"]]

    def render_web(self, observation):
        ...  # a 10 x 5 grid built from observation["info_state"]
```

## Extension point: `gradio_builder`

`create_app()` accepts an optional **`gradio_builder`** callable. When set, the UI at `/web` is built with [Gradio’s TabbedInterface](https://www.gradio.app/docs/gradio/tabbedinterface): by default the **first tab (“Playground”)** is the default OpenEnv UI, and the **second tab (“Custom”)** is the `gr.Blocks` returned by your builder (see [Naming and ordering the tabs](#naming-and-ordering-the-tabs) to change this). Users can switch between the default Playground and your custom interface without losing either. The same `/web/reset`, `/web/step`, `/web/state`, and `/web/metadata` API routes remain available; your custom tab can use the provided `web_manager` in-process or call those endpoints.

### Builder signature

```python
def my_gradio_builder(
    web_manager,      # WebInterfaceManager: .reset_environment(), .step_environment(), .get_state()
    action_fields,    # list[dict]: from action schema for form generation
    metadata,        # EnvironmentMetadata | None: name, readme_content, etc.
    is_chat_env,     # bool: True if single message input
    title,           # str: app title (e.g. metadata.name)
    quick_start_md,  # str: Quick Start markdown (class names already replaced)
) -> gr.Blocks:
    ...
```

Return a `gr.Blocks` instance. By default it is shown in the **“Custom”** tab, next to the **“Playground”** tab with the default OpenEnv UI. Core applies the same theme/css when mounting.

### Naming and ordering the tabs

`create_app()` takes a few options for the custom UI:

| Option | Default | Effect |
|---|---|---|
| `custom_tab_name` | `"Custom"` | Label of your tab |
| `custom_tab_primary` | `False` | Show your tab first, before Playground |
| `show_default_tab` | `True` | When `False`, mount only your builder's UI, with no Playground and no tabs |
| `title_override` | `None` | App and browser-tab title, instead of `"OpenEnv Agentic Environment: {name}"` |

---

## Option 1: Add a custom tab

Provide a builder that returns your own `gr.Blocks`; it appears as the second tab (“Custom”) next to the default “Playground” tab:

```python
# server/app.py
from openenv.core.env_server.http_server import create_app
from .my_environment import MyEnvironment
from ..models import MyAction, MyObservation
from .gradio_ui import build_my_gradio_app  # your module

app = create_app(
    MyEnvironment,
    MyAction,
    MyObservation,
    env_name="my_env",
    gradio_builder=build_my_gradio_app,
)
```

In `server/gradio_ui.py` implement `build_my_gradio_app(web_manager, action_fields, metadata, is_chat_env, title, quick_start_md)` returning a `gr.Blocks` (e.g. env-specific visualizations, extra controls). Use `web_manager.reset_environment()`, `web_manager.step_environment(action_data)`, and `web_manager.get_state()` in your Gradio event handlers. The default Playground tab remains available in the first tab.

---

## Option 2: Custom tab that wraps or reuses the default

Your builder can call the core `build_gradio_app` to get a Blocks instance and embed it inside your custom tab (e.g. in a `gr.Tabs` or as one section). That way your “Custom” tab can show both the default layout and additional content in one place.

---

## Option 3: Custom Quick Start or README only

You don’t need a custom builder only to change text. The default UI uses:

- **Quick Start**: generated from `get_quick_start_markdown(metadata, action_cls, observation_cls)` (init-style class names).
- **README**: `metadata.readme_content` (loaded from the env’s README).

So you can influence the default UI by ensuring `metadata` and README are correct. To change the Quick Start template itself (e.g. different wording or placeholders), you would use a custom `gradio_builder` that calls `build_gradio_app` with a custom `quick_start_md` string you build yourself (or by copying and adapting the default template from the core).

---

## Migration from custom HTML override (e.g. wildfire)

Environments that currently override `/web` with custom HTML (e.g. by removing the default route and adding a GET `/web` that returns HTML) should migrate to a **gradio_builder** that returns a `gr.Blocks` app. The custom UI then appears in the **“Custom”** tab alongside the default **“Playground”** tab. Benefits:

- Single, supported extension point using [TabbedInterface](https://www.gradio.app/docs/gradio/tabbedinterface).
- No need to remove or override routes; the default UI stays in the first tab.
- Same `/web` path; both tabs can use `web_manager` or `/web/reset`, `/web/step`, `/web/state`.

If you need a non-Gradio custom UI (e.g. static HTML/JS), you can still register your own route after `create_app` (e.g. at `/web/custom` or another path), but the main `/web` slot is the Gradio tabbed app when `ENABLE_WEB_INTERFACE=true`.

---

## Sign in with Hugging Face on a Space

A custom tab can sign visitors in with their Hugging Face account, for example so each visitor's runs use their own [Inference Providers](https://huggingface.co/docs/inference-providers) credits instead of a token stored on the Space. [`tau2_env`](../environments/tau2) does this. A Docker Space needs four things:

1. **The Space README** turns OAuth on, with the scopes you need:

   ```yaml
   hf_oauth: true
   hf_oauth_scopes:
     - inference-api
   ```

2. **The image** installs Gradio's OAuth dependencies, `authlib` and `itsdangerous` (the `gradio[oauth]` extra).
3. **`SYSTEM=spaces`** is set before the app is created. Gradio only uses the Space's real OAuth when it is set, and Docker Spaces don't set it, so it mocks the login instead:

   ```python
   if os.environ.get("SPACE_ID"):
       os.environ.setdefault("SYSTEM", "spaces")
   ```

4. **The OAuth routes are forwarded to `/web`.** The UI, and so Gradio's `/login/huggingface`, `/login/callback` and `/logout`, are mounted under `/web`, while the sign-in button and Hugging Face's callback use them at the root:

   ```python
   @app.get("/login/huggingface", include_in_schema=False)
   @app.get("/login/callback", include_in_schema=False)
   @app.get("/logout", include_in_schema=False)
   def oauth_under_web(request: Request) -> RedirectResponse:
       query = f"?{request.url.query}" if request.url.query else ""
       return RedirectResponse(f"/web{request.url.path}{query}")
   ```

In the builder, add a `gr.LoginButton()` and take a `gr.OAuthToken | None` argument in the event handlers that need the visitor's token. Off a Space the login is mocked with your local Hugging Face login, so show the button only when `SPACE_ID` is set.

---

## Summary

| Goal                         | Approach                                                                 |
|-----------------------------|---------------------------------------------------------------------------|
| Use default UI only         | Do not pass `gradio_builder`.                                            |
| Draw the state in the playground | Override `render_web(observation)` to return HTML.                    |
| One-click actions           | Override `web_actions(observation)` to return `(label, action)` pairs.   |
| Add a custom tab            | Pass `gradio_builder=my_builder`; return your own `gr.Blocks` (shown in “Custom” tab). |
| Custom tab + default inside | In your builder, call `build_gradio_app(...)` and embed or wrap it in your Blocks. |
| Change Quick Start / README | Rely on metadata/README, or custom builder that builds custom markdown.  |
| Sign visitors in on a Space | `hf_oauth` in the README, `SYSTEM=spaces`, OAuth routes forwarded to `/web` ([details](#sign-in-with-hugging-face-on-a-space)). |

The default Playground tab is built with `openenv.core.env_server.gradio_ui.build_gradio_app`; you can import and call it with the same arguments if your custom tab needs to embed or extend it.
