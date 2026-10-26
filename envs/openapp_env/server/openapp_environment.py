# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
OpenApp Environment Implementation.

A web application simulation environment that wraps OpenApps and BrowserGym.
This environment provides agent interaction with simulated web apps including
calendar, todo, messenger, and maps applications.
"""

import logging
import os
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

# Support both in-repo and standalone imports
try:
    from ..models import OpenAppAction, OpenAppObservation
except ImportError:
    from openapp_env.models import OpenAppAction, OpenAppObservation

logger = logging.getLogger(__name__)


class GenericOpenAppsTask:
    """
    A generic task for OpenApps interaction without specific goals.

    This is a simple wrapper that allows BrowserGym to interact with OpenApps
    without requiring a specific task. For task-based interaction, use the
    OpenAppsTask from open_apps.tasks.add_tasks_to_browsergym.
    """

    def __init__(
        self,
        base_url: str,
        seed: int = 1,
        **kwargs,
    ) -> None:
        """
        Initialize generic OpenApps task.

        Args:
            base_url: Base URL of the OpenApps server
            seed: Random seed (required by BrowserGym)
            **kwargs: Additional arguments (ignored)
        """
        try:
            import playwright.sync_api
            from browsergym.core.task import AbstractBrowserTask
        except ImportError:
            raise ImportError(
                "BrowserGym is required. Install with: pip install browsergym"
            )

        # Store as instance attributes
        self.base_url = base_url
        self.seed = seed

        # BrowserGym task properties
        self.viewport = {"width": 1024, "height": 768}
        self.slow_mo = 100
        self.timeout = 5000

        # Additional properties that BrowserGym might expect
        self.locale = None
        self.timezone_id = None
        self.geolocation = None

    def setup(self, page: "playwright.sync_api.Page") -> Tuple[str, Dict[str, Any]]:
        """
        Set up the task by navigating to the base URL.

        Args:
            page: Playwright page object

        Returns:
            Tuple of (goal_string, info_dict)
        """
        page.goto(self.base_url)
        return "Explore OpenApps", {}

    def teardown(self) -> None:
        """Clean up after task completion."""
        pass

    def validate(
        self, page: "playwright.sync_api.Page", chat_messages: list[str]
    ) -> Tuple[float, bool, str, Dict[str, Any]]:
        """
        Validate task state and return reward.

        Args:
            page: Playwright page object
            chat_messages: List of chat messages

        Returns:
            Tuple of (reward, done, message, info)
        """
        # Generic task never completes automatically
        return 0.0, False, "", {}

    def cheat(self, page: "playwright.sync_api.Page", chat_messages: list[str]) -> None:
        """Cheat method (no-op for generic task)."""
        pass


class OpenAppEnvironment(Environment):
    """
    A web application environment that wraps OpenApps and BrowserGym.

    This environment launches OpenApps web server and provides a BrowserGym-like
    interface for agents to interact with simulated web applications.

    Args:
        openapps_path: Path to OpenApps directory (default: auto-detect)
        web_app_port: Port for OpenApps web server (default: 5001)
        headless: Run browser in headless mode (default: True)
        task_name: Optional task name to evaluate (e.g., "add_meeting_with_dennis")
        apps_config: Configuration for apps (default: all enabled)
        max_steps: Maximum steps per episode (default: 50)

    Example:
        >>> env = OpenAppEnvironment()
        >>> obs = env.reset()
        >>> print(obs.url)  # Starting page URL
        >>>
        >>> # Click on an element
        >>> action = OpenAppAction(action_type="click", bid="calendar-btn")
        >>> obs = env.step(action)
        >>> print(obs.html)
    """

    def __init__(
        self,
        openapps_url: Optional[str] = None,
        openapps_path: Optional[str] = None,
        web_app_port: int = 5001,
        headless: bool = True,
        task_name: Optional[str] = None,
        apps_config: Optional[Dict[str, Any]] = None,
        max_steps: int = 50,
    ):
        """Initialize the OpenApp environment."""
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._max_steps = max_steps

        # OpenApps configuration
        # Priority: 1. openapps_url, 2. OPENAPPS_URL env var, 3. Try to find/launch
        self.openapps_url = openapps_url or os.environ.get("OPENAPPS_URL")
        if not self.openapps_url:
            self.web_app_port = web_app_port
            self.openapps_url = f"http://localhost:{web_app_port}"

        self.openapps_path = openapps_path
        self.headless = headless
        self.task_name = task_name
        self.apps_config = apps_config or {}

        # Runtime state
        self._apps_process: Optional[subprocess.Popen] = None
        self._browser_env = None
        self._current_html = ""
        self._current_url = ""
        self._current_axtree = ""
        self._app_state = {}
        self._last_action_error = None
        self._episode_reward = 0.0

    def _detect_openapps_path(self) -> str:
        """
        Auto-detect OpenApps path.

        Since OpenApps is installed as a Python package, we use the installed
        package location instead of requiring a separate directory.
        """
        # Check if user provided a custom path via environment variable
        env_path = os.environ.get("OPENAPPS_PATH")
        if env_path and Path(env_path).exists():
            return env_path

        # Try to find OpenApps as an installed package
        try:
            import open_apps

            openapps_pkg_path = Path(open_apps.__file__).parent.parent
            if openapps_pkg_path.exists():
                return str(openapps_pkg_path)
        except ImportError:
            pass

        raise ValueError(
            "OpenApps not found. Please install it with: "
            "pip install git+https://github.com/facebookresearch/OpenApps.git "
            "or set OPENAPPS_PATH environment variable."
        )

    def _launch_openapps_server(self) -> Optional[subprocess.Popen]:
        """
        Launch OpenApps web server in background.

        Returns None if server is expected to be already running (OPENAPPS_URL set).
        """
        # If OPENAPPS_URL is set, assume server is already running
        if os.environ.get("OPENAPPS_URL"):
            logger.info(f"Using existing OpenApps server at {self.openapps_url}")
            # Wait for server to be available
            self._wait_for_server(max_wait=5)
            return None

        # Otherwise, provide helpful error message
        raise NotImplementedError(
            "Automatic OpenApps server launch is not yet implemented.\n"
            "\n"
            "Please start OpenApps manually in a separate terminal:\n"
            "  1. Clone OpenApps: git clone https://github.com/facebookresearch/OpenApps.git\n"
            "  2. Install: cd OpenApps && uv sync\n"
            "  3. Run: uv run launch.py\n"
            "\n"
            "Then set the OPENAPPS_URL environment variable:\n"
            "  export OPENAPPS_URL=http://localhost:5001\n"
            "\n"
            "Or use Docker mode which handles this automatically:\n"
            "  python examples/openapp_example.py --mode docker\n"
        )

    def _wait_for_server(self, max_wait: int = 30):
        """Wait for OpenApps server to become available."""
        for i in range(max_wait):
            try:
                response = urllib.request.urlopen(self.openapps_url, timeout=2)
                if response.status == 200:
                    return
            except Exception:
                pass
            time.sleep(1)

        raise TimeoutError(f"OpenApps server did not start within {max_wait} seconds")

    def _initialize_browser(self):
        """Initialize BrowserGym environment for interaction."""
        try:
            from browsergym.core.env import BrowserEnv
        except ImportError:
            raise ImportError(
                "BrowserGym is required for OpenApp environment. "
                "Install it with: pip install browsergym"
            )

        # Create BrowserGym environment with generic OpenApps task
        self._browser_env = BrowserEnv(
            task_entrypoint=GenericOpenAppsTask,
            task_kwargs={"base_url": self.openapps_url},
            headless=self.headless,
            slow_mo=200,  # Slow down actions so they're visible (200ms delay)
        )

    def _get_current_observation(self) -> Dict[str, Any]:
        """Extract current observation from browser state."""
        if self._browser_env is None:
            return {
                "html": "",
                "url": self.openapps_url,
                "open_pages_urls": [self.openapps_url],
                "active_page_index": 0,
                "axtree_txt": "",
                "app_state": {},
            }

        # Get browser state (implementation depends on BrowserGym API)
        # This is a simplified version - actual implementation would use BrowserGym's observation
        return {
            "html": self._current_html,
            "url": self._current_url,
            "open_pages_urls": [self._current_url],
            "active_page_index": 0,
            "axtree_txt": self._current_axtree,
            "app_state": self._app_state,
        }

    def reset(self) -> OpenAppObservation:
        """
        Reset the environment.

        Returns:
            OpenAppObservation with initial state
        """
        # Reset state
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._episode_reward = 0.0
        self._last_action_error = None

        # Check if OpenApps server is running, start if needed
        if self._apps_process is None and not os.environ.get("OPENAPPS_URL"):
            self._apps_process = self._launch_openapps_server()

        # Initialize browser
        if self._browser_env is None:
            self._initialize_browser()

        # Reset the BrowserGym environment
        try:
            obs, info = self._browser_env.reset()
            # Extract observation data from BrowserGym
            self._current_url = obs.get("url", self.openapps_url)
            self._current_html = obs.get("dom_txt", "")
            self._current_axtree = obs.get("axtree_txt", "")
            self._app_state = {}
        except Exception as e:
            logger.warning(f"Failed to reset browser environment: {e}")
            # Fallback to placeholder values
            self._current_url = self.openapps_url
            self._current_html = "<html><body>OpenApps Ready</body></html>"
            self._current_axtree = ""
            self._app_state = {}

        obs_data = self._get_current_observation()

        return OpenAppObservation(
            html=obs_data["html"],
            url=obs_data["url"],
            open_pages_urls=obs_data["open_pages_urls"],
            active_page_index=obs_data["active_page_index"],
            axtree_txt=obs_data["axtree_txt"],
            app_state=obs_data["app_state"],
            task_info={"task_name": self.task_name} if self.task_name else None,
            last_action_error=None,
            done=False,
            reward=0.0,
        )

    def step(self, action: OpenAppAction) -> OpenAppObservation:  # type: ignore[override]
        """
        Execute a step in the environment.

        Args:
            action: OpenAppAction to execute

        Returns:
            OpenAppObservation with resulting state and reward
        """
        self._state.step_count += 1
        self._last_action_error = None
        reward = 0.0

        try:
            # Execute action based on type
            if action.action_type == "click":
                reward = self._execute_click(action.bid)
            elif action.action_type == "fill":
                reward = self._execute_fill(action.bid, action.text)
            elif action.action_type == "select_option":
                reward = self._execute_select(action.bid, action.value)
            elif action.action_type == "goto":
                reward = self._execute_goto(action.url)
            elif action.action_type == "scroll":
                reward = self._execute_scroll(action.direction)
            elif action.action_type == "send_keys":
                reward = self._execute_send_keys(action.text)
            elif action.action_type == "noop":
                reward = 0.0
            else:
                self._last_action_error = f"Unknown action type: {action.action_type}"
                reward = -0.1

        except Exception as e:
            self._last_action_error = str(e)
            reward = -0.1

        # Update cumulative reward
        self._episode_reward += reward

        # Check if episode is done
        done = self._state.step_count >= self._max_steps

        # Get current observation
        obs_data = self._get_current_observation()

        return OpenAppObservation(
            html=obs_data["html"],
            url=obs_data["url"],
            open_pages_urls=obs_data["open_pages_urls"],
            active_page_index=obs_data["active_page_index"],
            axtree_txt=obs_data["axtree_txt"],
            app_state=obs_data["app_state"],
            task_info={"task_name": self.task_name} if self.task_name else None,
            last_action_error=self._last_action_error,
            done=done,
            reward=reward,
            metadata={"cumulative_reward": self._episode_reward},
        )

    def _execute_click(self, bid: str) -> float:
        """Execute click action. Returns reward.

        Supports two modes:
        1. CSS selector mode: If bid starts with '#', '.', or '[', it's treated as a CSS selector
           and uses Playwright directly (e.g., bid="#msg-input")
        2. BrowserGym mode: Otherwise, uses BrowserGym's accessibility tree bid
        """
        if self._browser_env is None:
            return 0.0

        try:
            # Check if bid is a CSS selector (starts with # or other CSS selector chars)
            if bid.startswith("#") or bid.startswith(".") or bid.startswith("["):
                # Use Playwright directly for CSS selectors
                return self._execute_click_playwright(bid)

            # BrowserGym action format: click("bid")
            action = f'click("{bid}")'
            obs, reward, done, truncated, info = self._browser_env.step(action)

            # Update current state from observation
            self._current_url = obs.get("url", self._current_url)
            self._current_html = obs.get("dom_txt", self._current_html)
            self._current_axtree = obs.get("axtree_txt", self._current_axtree)

            return float(reward) if reward else 0.0
        except Exception as e:
            self._last_action_error = f"Click failed: {str(e)}"
            return -0.1

    def _execute_fill(self, bid: str, text: str) -> float:
        """Execute fill action. Returns reward.

        Supports two modes:
        1. CSS selector mode: If bid starts with '#', it's treated as an HTML ID selector
           and uses Playwright directly (e.g., bid="#msg-input")
        2. BrowserGym mode: Otherwise, uses BrowserGym's accessibility tree bid
        """
        if self._browser_env is None:
            return 0.0

        try:
            # Check if bid is a CSS selector (starts with # or other CSS selector chars)
            if bid.startswith("#") or bid.startswith(".") or bid.startswith("["):
                # Use Playwright directly for CSS selectors
                return self._execute_fill_playwright(bid, text)

            # BrowserGym action format: fill("bid", "text")
            action = f'fill("{bid}", "{text}")'
            obs, reward, done, truncated, info = self._browser_env.step(action)

            # Update current state from observation
            self._current_url = obs.get("url", self._current_url)
            self._current_html = obs.get("dom_txt", self._current_html)
            self._current_axtree = obs.get("axtree_txt", self._current_axtree)

            return float(reward) if reward else 0.0
        except Exception as e:
            self._last_action_error = f"Fill failed: {str(e)}"
            return -0.1

    def _execute_fill_playwright(self, selector: str, text: str) -> float:
        """Execute fill action using Playwright directly with CSS selector."""
        try:
            # Access the underlying Playwright page from BrowserGym
            page = self._browser_env.unwrapped.page

            # Wait for element and fill it
            page.wait_for_selector(selector, timeout=5000)
            page.fill(selector, text)

            # Small delay to let the page update
            page.wait_for_timeout(200)

            # Update observation after action
            self._update_observation_from_page(page)

            return 0.0
        except Exception as e:
            self._last_action_error = f"Fill (Playwright) failed: {str(e)}"
            return -0.1

    def _execute_click_playwright(self, selector: str) -> float:
        """Execute click action using Playwright directly with CSS selector."""
        try:
            # Access the underlying Playwright page from BrowserGym
            page = self._browser_env.unwrapped.page

            # Wait for element and click it
            page.wait_for_selector(selector, timeout=5000)
            page.click(selector)

            # Longer delay to let HTMX process the request
            page.wait_for_timeout(500)

            # Update observation after action
            self._update_observation_from_page(page)

            return 0.0
        except Exception as e:
            self._last_action_error = f"Click (Playwright) failed: {str(e)}"
            return -0.1

    def _execute_press_key_playwright(self, key: str) -> float:
        """Execute key press using Playwright directly."""
        try:
            # Access the underlying Playwright page from BrowserGym
            page = self._browser_env.unwrapped.page

            # Press the key
            page.keyboard.press(key)

            # Delay to let the page update
            page.wait_for_timeout(500)

            # Update observation after action
            self._update_observation_from_page(page)

            return 0.0
        except Exception as e:
            self._last_action_error = f"Press key (Playwright) failed: {str(e)}"
            return -0.1

    def _update_observation_from_page(self, page) -> None:
        """Update internal observation state from Playwright page."""
        try:
            self._current_url = page.url
            # Note: We can't easily get axtree from Playwright directly,
            # so we'll just update URL. The next BrowserGym action will sync the state.
        except Exception:
            pass

    def _execute_select(self, bid: str, value: str) -> float:
        """Execute select option action. Returns reward."""
        if self._browser_env is None:
            return 0.0

        try:
            # BrowserGym action format: select_option("bid", "value")
            action = f'select_option("{bid}", "{value}")'
            obs, reward, done, truncated, info = self._browser_env.step(action)

            # Update current state from observation
            self._current_url = obs.get("url", self._current_url)
            self._current_html = obs.get("dom_txt", self._current_html)
            self._current_axtree = obs.get("axtree_txt", self._current_axtree)

            return float(reward) if reward else 0.0
        except Exception as e:
            self._last_action_error = f"Select failed: {str(e)}"
            return -0.1

    def _execute_goto(self, url: str) -> float:
        """Execute navigation action. Returns reward."""
        if self._browser_env is None:
            self._current_url = url
            return 0.0

        try:
            # BrowserGym action format: goto("url")
            action = f'goto("{url}")'
            obs, reward, done, truncated, info = self._browser_env.step(action)

            # Update current state from observation
            self._current_url = obs.get("url", url)
            self._current_html = obs.get("dom_txt", self._current_html)
            self._current_axtree = obs.get("axtree_txt", self._current_axtree)

            return float(reward) if reward else 0.0
        except Exception as e:
            self._last_action_error = f"Goto failed: {str(e)}"
            self._current_url = url  # Update URL even if failed
            return -0.1

    def _execute_scroll(self, direction: str) -> float:
        """Execute scroll action. Returns reward."""
        if self._browser_env is None:
            return 0.0

        try:
            # BrowserGym action format: scroll("direction")
            action = f'scroll("{direction}")'
            obs, reward, done, truncated, info = self._browser_env.step(action)

            # Update current state from observation
            self._current_url = obs.get("url", self._current_url)
            self._current_html = obs.get("dom_txt", self._current_html)
            self._current_axtree = obs.get("axtree_txt", self._current_axtree)

            return float(reward) if reward else 0.0
        except Exception as e:
            self._last_action_error = f"Scroll failed: {str(e)}"
            return -0.1

    def _execute_send_keys(self, text: str) -> float:
        """Execute send keys action. Returns reward."""
        if self._browser_env is None:
            return 0.0

        try:
            # Special handling for Enter key - use Playwright directly for reliability
            if text == "\n" or text.lower() == "enter":
                return self._execute_press_key_playwright("Enter")

            # BrowserGym action format: send_keys("text")
            action = f'send_keys("{text}")'
            obs, reward, done, truncated, info = self._browser_env.step(action)

            # Update current state from observation
            self._current_url = obs.get("url", self._current_url)
            self._current_html = obs.get("dom_txt", self._current_html)
            self._current_axtree = obs.get("axtree_txt", self._current_axtree)

            return float(reward) if reward else 0.0
        except Exception as e:
            self._last_action_error = f"Send keys failed: {str(e)}"
            return -0.1

    @property
    def state(self) -> State:
        """
        Get the current environment state.

        Returns:
            Current State with episode_id and step_count
        """
        return self._state

    def close(self):
        """Clean up resources."""
        if hasattr(self, "_browser_env") and self._browser_env is not None:
            try:
                self._browser_env.close()
            except Exception:
                pass
            self._browser_env = None

        if hasattr(self, "_apps_process") and self._apps_process is not None:
            try:
                self._apps_process.terminate()
                self._apps_process.wait(timeout=5)
            except Exception:
                self._apps_process.kill()
            self._apps_process = None

    def __del__(self):
        """Cleanup on deletion."""
        self.close()
