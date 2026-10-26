"""BrowserGym Environment implementation for OpenEnv.

This module wraps the BrowserGym framework to provide a compatible interface
with OpenEnv's Environment ABC. BrowserGym includes multiple benchmarks:
- MiniWoB++: Training environment with 100+ simple web tasks
- WebArena: Realistic evaluation with 812 complex tasks
- VisualWebArena: Visual web navigation tasks
- WorkArena: Enterprise task automation
"""

import base64
import html
import importlib
import io
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional
from uuid import uuid4

import gymnasium as gym
import numpy as np
from browsergym_env.models import (
    BrowserGymAction,
    BrowserGymObservation,
    BrowserGymState,
)
from openenv.core.env_server.interfaces import Environment
from PIL import Image

logger = logging.getLogger(__name__)

# browsergym.core keeps one process-wide Playwright instance (sync API), which only
# works on the thread that started it. Every BrowserGym call runs on this single thread
# so the HTTP API, WebSocket sessions and the /web playground can share one process.
_PLAYWRIGHT_THREAD = ThreadPoolExecutor(max_workers=1, thread_name_prefix="browsergym")


def _on_playwright_thread(fn, *args, **kwargs):
    """Run `fn` on the thread that owns the BrowserGym Playwright instance."""
    return _PLAYWRIGHT_THREAD.submit(fn, *args, **kwargs).result()


def _to_jsonable(value: Any) -> Any:
    """Convert BrowserGym payload values into JSON-safe Python objects."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        try:
            return _to_jsonable(value.tolist())
        except Exception:  # noqa: BLE001 - third-party array-likes
            pass
    if hasattr(value, "item"):
        try:
            return _to_jsonable(value.item())
        except Exception:  # noqa: BLE001 - third-party scalar-likes
            pass
    return str(value)


def _get_axtree_txt(obs: Dict[str, Any]) -> str:
    """Extract accessibility tree text from BrowserGym observation.

    BrowserGym returns raw `axtree_object` which needs to be converted to text
    using the `flatten_axtree_to_str` utility function.
    """
    # If already processed as text, return directly
    if "axtree_txt" in obs and obs["axtree_txt"]:
        return obs["axtree_txt"]

    # Try to convert from raw axtree_object
    if "axtree_object" in obs and obs["axtree_object"]:
        try:
            from browsergym.utils.obs import flatten_axtree_to_str

            return flatten_axtree_to_str(obs["axtree_object"])
        except ImportError:
            logger.warning(
                "browsergym.utils.obs not available, cannot convert axtree_object to text"
            )
        except Exception as e:
            logger.warning(f"Failed to convert axtree_object to text: {e}")

    return ""


def _get_pruned_html(obs: Dict[str, Any]) -> str:
    """Extract pruned HTML from BrowserGym observation.

    BrowserGym returns raw `dom_object` which needs to be converted to text
    and then pruned using the `flatten_dom_to_str` and `prune_html` utilities.
    """
    # If already processed as pruned_html, return directly
    if "pruned_html" in obs and obs["pruned_html"]:
        return obs["pruned_html"]

    # Try to convert from raw dom_object
    if "dom_object" in obs and obs["dom_object"]:
        try:
            from browsergym.utils.obs import flatten_dom_to_str, prune_html

            dom_str = flatten_dom_to_str(obs["dom_object"])
            return prune_html(dom_str)
        except ImportError:
            logger.warning(
                "browsergym.utils.obs not available, cannot convert dom_object to pruned_html"
            )
        except Exception as e:
            logger.warning(f"Failed to convert dom_object to pruned_html: {e}")

    return ""


_MINIWOB_LOAD_HELP = (
    "MiniWoB tasks require the MiniWoB HTML bundle to be served over HTTP. "
    "The official BrowserGym Docker image handles this automatically by "
    "serving the bundle on port 8888. For custom or non-Docker deployments, "
    "clone the MiniWoB++ repository, start a static server inside "
    "`miniwob-plusplus/miniwob/html` (e.g. `python -m http.server 8888`), and "
    "set the MINIWOB_URL environment variable to the served base URL such as "
    "`http://localhost:8888/miniwob/`."
)


class BrowserGymEnvironment(Environment):
    """BrowserGym environment wrapper for OpenEnv.

    This environment wraps BrowserGym's Gymnasium-compatible environments to
    provide unified access to multiple web navigation benchmarks.
    """

    SUPPORTS_CONCURRENT_SESSIONS = True
    REQUIRES_SINGLE_THREAD_EXECUTOR = True

    def __init__(
        self,
        benchmark: str = "miniwob",
        task_name: Optional[str] = None,
        headless: bool = True,
        viewport_width: int = 1280,
        viewport_height: int = 720,
        timeout: float = 10000.0,
        include_screenshot: bool = False,
        **gym_kwargs: Any,
    ):
        """Initialize the BrowserGym environment.

        Args:
            benchmark: Benchmark to use ('miniwob', 'webarena', 'visualwebarena', etc.)
            task_name: Specific task within the benchmark (e.g., 'click-test', 'click-button')
                      If None, will use first available task
            headless: Whether to run browser in headless mode
            viewport_width: Browser viewport width
            viewport_height: Browser viewport height
            timeout: Action timeout in milliseconds
            **gym_kwargs: Additional arguments passed to gym.make()
        """
        super().__init__()

        self.benchmark = benchmark
        self.task_name = task_name
        self.headless = headless
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height
        self.timeout = timeout
        self.include_screenshot = include_screenshot
        self.gym_kwargs = dict(gym_kwargs)

        # force import the benchmark module
        benchmark_modules = {
            "miniwob": "browsergym.miniwob",
            "webarena": "browsergym.webarena",
            "visualwebarena": "browsergym.visualwebarena",
            "workarena": "browsergym.workarena",
        }
        module_path = benchmark_modules.get(benchmark)
        try:
            if module_path:
                importlib.import_module(module_path)
            else:
                importlib.import_module("browsergym")
        except ModuleNotFoundError as import_error:
            message = (
                "Failed to import BrowserGym benchmark "
                f"'{benchmark}': {import_error}\n"
                "Install the matching browsergym package "
                f"(e.g., browsergym-{benchmark})."
            )
            raise ValueError(message) from import_error

        self.env_id = self._build_env_id(task_name)
        self.gym_env = self._create_gym_env(self.env_id)

        # State tracking
        self._state = BrowserGymState(
            episode_id=str(uuid4()),
            step_count=0,
            benchmark=benchmark,
            task_name=task_name or "",
        )

        self._last_obs: Optional[Dict[str, Any]] = None
        self._last_info: Optional[Dict[str, Any]] = None

    def _build_env_id(self, task_name: Optional[str]) -> str:
        if task_name:
            return f"browsergym/{self.benchmark}.{task_name}"
        return f"browsergym/{self.benchmark}"

    def _create_gym_env(self, env_id: str):
        try:
            return gym.make(
                env_id,
                headless=self.headless,
                viewport={
                    "width": self.viewport_width,
                    "height": self.viewport_height,
                },
                timeout=self.timeout,
                **self.gym_kwargs,
            )
        except Exception as e:  # noqa: BLE001 - gym.make
            message = (
                "Failed to create BrowserGym environment "
                f"'{env_id}': {e}\n"
                "Make sure the benchmark package is installed "
                f"(e.g., pip install browsergym-{self.benchmark})."
            )
            raise ValueError(message) from e

    def _switch_task_if_needed(self, requested_task: str) -> None:
        if requested_task == (self.task_name or ""):
            return

        next_task_name = requested_task or None
        next_env_id = self._build_env_id(next_task_name)
        next_gym_env = self._create_gym_env(next_env_id)

        previous_gym_env = getattr(self, "gym_env", None)
        self.task_name = next_task_name
        self.env_id = next_env_id
        self.gym_env = next_gym_env

        if previous_gym_env is not None:
            try:
                _on_playwright_thread(previous_gym_env.close)
            except Exception as exc:  # noqa: BLE001 - browsergym/playwright cleanup
                logger.warning("BrowserGym cleanup failed during task switch: %s", exc)

    def reset(
        self,
        seed: Optional[int] = None,
        task_name: Optional[str] = None,
        episode_id: Optional[str] = None,
    ) -> BrowserGymObservation:
        """Reset the environment with a specific task.

        Args:
            seed: Random seed for reproducibility
            episode_id: Optional episode ID to use
            task_name: Override task name for this episode

        Returns:
            Initial observation for the task
        """
        # Generate new episode ID
        resolved_task_name = task_name or self.task_name or ""
        self._switch_task_if_needed(resolved_task_name)

        self._state = BrowserGymState(
            episode_id=episode_id if episode_id is not None else str(uuid4()),
            step_count=0,
            benchmark=self.benchmark,
            task_name=resolved_task_name,
        )

        # Reset options
        reset_options = {}
        if seed is not None:
            reset_options["seed"] = seed

        # Reset the gym environment
        try:
            obs, info = _on_playwright_thread(self.gym_env.reset, **reset_options)
        except AttributeError as err:
            if "context" in str(err) and hasattr(self.gym_env, "close"):
                # BrowserGym can leave partially initialized state after a
                # failed reset. Close the hanging resources and try once more.
                _on_playwright_thread(self.gym_env.close)
                obs, info = _on_playwright_thread(self.gym_env.reset, **reset_options)
            else:
                raise
        except Exception as err:  # noqa: BLE001 - browsergym
            message = str(err)
            if self.benchmark == "miniwob" and "core is not defined" in message:
                raise ValueError(_MINIWOB_LOAD_HELP) from err
            raise

        self._last_obs = obs
        self._last_info = info

        # Extract observation details
        return self._create_observation(obs, info, done=False, reward=0.0)

    def step(self, action: BrowserGymAction) -> BrowserGymObservation:
        """Execute an action in the environment.

        Args:
            action: The action to execute

        Returns:
            Observation after executing the action
        """
        self._state.step_count += 1

        # Execute action in gym environment
        try:
            obs, reward, terminated, truncated, info = _on_playwright_thread(
                self.gym_env.step, action.action_str
            )

            self._last_obs = obs
            self._last_info = info

            # Update state
            done = terminated or truncated
            self._state.cum_reward += float(reward)

            # Extract goal from info if available
            if "goal" in info:
                self._state.goal = str(info["goal"])

            return self._create_observation(obs, info, done=done, reward=float(reward))

        except Exception as e:
            # Handle action execution errors
            error_msg = str(e)
            return BrowserGymObservation(
                text=self._last_obs.get("text", "") if self._last_obs else "",
                url=self._last_obs.get("url", "") if self._last_obs else "",
                goal=self._state.goal,
                error=error_msg,
                last_action_error=True,
                done=False,
                reward=0.0,
            )

    def _create_observation(
        self,
        obs: Dict[str, Any],
        info: Dict[str, Any],
        done: bool,
        reward: float,
    ) -> BrowserGymObservation:
        """Convert BrowserGym observation to OpenEnv format.

        Args:
            obs: BrowserGym observation dict
            info: BrowserGym info dict
            done: Whether episode is done
            reward: Reward for the step

        Returns:
            BrowserGymObservation
        """
        # Generate text representations from raw BrowserGym objects
        # BrowserGym returns axtree_object and dom_object which need conversion
        axtree_txt = _get_axtree_txt(obs) if isinstance(obs, dict) else ""
        pruned_html = _get_pruned_html(obs) if isinstance(obs, dict) else ""

        # Extract text observation - prefer axtree_txt, fallback to pruned_html
        text = axtree_txt or pruned_html
        if not text and isinstance(obs, str):
            text = obs

        # Extract URL from obs (BrowserGym stores it there)
        url = ""
        if isinstance(obs, dict):
            url = obs.get("url", "")

        # Extract goal/instruction from goal_object or legacy goal field
        goal = ""
        if isinstance(obs, dict):
            # New format: goal_object is a list of messages
            goal_object = obs.get("goal_object", [])
            if goal_object:
                # Extract text content from goal messages
                goal_texts = []
                for msg in goal_object:
                    if isinstance(msg, dict):
                        content = msg.get("content", "")
                        if isinstance(content, str):
                            goal_texts.append(content)
                        elif isinstance(content, list):
                            for item in content:
                                if (
                                    isinstance(item, dict)
                                    and item.get("type") == "text"
                                ):
                                    goal_texts.append(item.get("text", ""))
                goal = " ".join(goal_texts)
            # Fallback to legacy goal field
            if not goal:
                goal = obs.get("goal", "")

        # Update state
        self._state.current_url = url
        self._state.goal = goal

        # Extract additional observation modalities
        screenshot = None
        if self.include_screenshot and isinstance(obs, dict):
            screenshot = _to_jsonable(obs.get("screenshot"))

        # Extract last_action_error from obs (BrowserGym includes this)
        last_action_error = False
        if isinstance(obs, dict):
            last_action_error = bool(obs.get("last_action_error"))

        # Store full BrowserGym observation and info in metadata
        # This preserves timestamps, additional fields, and any future extensions
        # Note: We exclude large objects (dom_object, axtree_object) to reduce payload size
        browsergym_metadata = {}
        if isinstance(obs, dict):
            # Include useful fields but exclude large raw objects
            browsergym_metadata["browsergym_obs"] = {
                k: _to_jsonable(v)
                for k, v in obs.items()
                if k not in ("dom_object", "axtree_object", "screenshot")
            }
        browsergym_metadata["browsergym_info"] = _to_jsonable(info)

        return BrowserGymObservation(
            text=text,
            url=url,
            screenshot=screenshot,
            goal=goal,
            axtree_txt=axtree_txt,
            pruned_html=pruned_html,
            error="",
            last_action_error=last_action_error,
            done=done,
            reward=reward,
            metadata=browsergym_metadata,
        )

    def render_web(self, observation: Dict[str, Any]) -> Optional[str]:
        """Draw the page screenshot with the goal, the URL and the last action error."""
        if "screenshot" not in observation:
            return None
        screenshot = observation["screenshot"]
        goal = observation["goal"]
        if not screenshot and not goal:
            return None
        parts = []
        if goal:
            parts.append(f"<div><b>Goal:</b> {html.escape(goal)}</div>")
        if observation["url"]:
            parts.append(
                '<div style="font-size:12px;opacity:0.7;word-break:break-all">'
                f"{html.escape(observation['url'])}</div>"
            )
        if screenshot:
            image = Image.fromarray(np.asarray(screenshot, dtype=np.uint8))
            image.thumbnail((800, 800))
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=80)
            data = base64.b64encode(buffer.getvalue()).decode()
            parts.append(
                f'<img alt="BrowserGym page" src="data:image/jpeg;base64,{data}" '
                'style="max-width:100%;border:1px solid var(--border-color-primary);border-radius:6px">'
            )
        error = observation["error"] or observation["metadata"].get(
            "browsergym_obs", {}
        ).get("last_action_error")
        if error:
            parts.append(
                '<div style="color:var(--color-accent)"><b>Last action error:</b> '
                f"{html.escape(str(error))}</div>"
            )
        return (
            '<div style="display:flex;flex-direction:column;'
            "gap:8px;max-width:500px;padding:10px;color:var(--body-text-color);"
            "border:1px solid var(--border-color-primary);border-radius:10px;"
            'background:var(--background-fill-secondary)">' + "".join(parts) + "</div>"
        )

    @property
    def state(self) -> BrowserGymState:
        """Get the current environment state."""
        return self._state

    def close(self) -> None:
        """Clean up environment resources."""
        if hasattr(self, "gym_env"):
            try:
                _on_playwright_thread(self.gym_env.close)
            except Exception as exc:  # noqa: BLE001 - browsergym/playwright cleanup
                logger.warning("BrowserGym cleanup failed: %s", exc)
