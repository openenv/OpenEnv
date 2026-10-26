"""The Helium browser environment that runs only inside the sandbox."""

import html
import os
import tempfile
import time
import uuid
from urllib.parse import urlsplit

import helium as h
from openenv.core.env_server import Environment
from selenium import webdriver
from selenium.common.exceptions import NoAlertPresentException, TimeoutException
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.keys import Keys

from ..models import BrowserAction, BrowserObservation, BrowserState
from .desktop import VirtualDesktop


def action_delay(operation):
    if operation == "finish":
        return 0.0
    name, default = (
        ("BROWSER_EXPLICIT_WAIT_SECONDS", "3")
        if operation == "wait"
        else ("BROWSER_ACTION_WAIT_SECONDS", "1.5")
    )
    seconds = float(os.getenv(name, default))
    if not 0 <= seconds <= 10:
        raise ValueError(f"{name} must be between 0 and 10")
    return seconds


def verification_block(text):
    normalized = " ".join((text or "").lower().split())
    return (
        "cloudflare" in normalized
        and any(value in normalized for value in ("verifying you are human", "verify you are human"))
    ) or (
        "vercel security checkpoint" in normalized
        and any(value in normalized for value in ("verify your browser", "verifying your browser"))
    )


class BrowserEnvironment(Environment):
    REQUIRES_SINGLE_THREAD_EXECUTOR = True
    VIEWPORT_GEOMETRY = [800, 600, 1, 0, 0]

    def __init__(self):
        super().__init__()
        if os.getenv("BROWSER_SANDBOX") != "hf":
            raise RuntimeError("start this environment through the HF Sandbox launcher")
        self.driver = None
        self.desktop = None
        self.profile = None
        self._state = BrowserState()
        self.done = False
        self.max_steps = int(os.getenv("BROWSER_MAX_STEPS", "20"))
        if not 1 <= self.max_steps <= 200:
            raise ValueError("BROWSER_MAX_STEPS must be between 1 and 200")

    def wait_for_viewport(self, process_id):
        for _ in range(30):
            geometry = self.driver.execute_script(
                "return [innerWidth, innerHeight, devicePixelRatio, screenX, screenY]"
            )
            if geometry == self.VIEWPORT_GEOMETRY:
                return
            time.sleep(0.5)
            try:
                self.desktop.focus_browser(process_id)
            except Exception:
                pass
        raise RuntimeError("browser viewport did not align to 800x600")

    def reset(self, seed=None, episode_id=None, start_url=None, **kwargs):
        self.close()
        start_url = start_url or os.getenv("BROWSER_START_URL", "https://huggingface.co")
        parsed = urlsplit(start_url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError("expected an HTTP(S) URL without credentials")
        self.profile = tempfile.TemporaryDirectory(prefix="browser-")
        options = webdriver.ChromeOptions()
        options.binary_location = "/usr/bin/chromium"
        for flag in (
            "--kiosk",
            "--window-position=0,0",
            "--force-device-scale-factor=1",
            "--no-sandbox",
            "--disable-dev-shm-usage",
            "--disable-quic",
            "--disable-background-networking",
            "--disable-features=WebRtcHideLocalIpsWithMdns",
            "--force-webrtc-ip-handling-policy=disable_non_proxied_udp",
            "--window-size=800,600",
            "--user-data-dir=" + self.profile.name,
        ):
            options.add_argument(flag)
        options.add_experimental_option(
            "prefs",
            {
                "download_restrictions": 3,
                "credentials_enable_service": False,
                "profile.password_manager_enabled": False,
            },
        )
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        try:
            self.desktop = VirtualDesktop()
            self.driver = webdriver.Chrome(
                service=Service("/usr/bin/chromedriver", env=self.desktop.env), options=options
            )
            self.driver.set_page_load_timeout(30)
            self.driver.set_script_timeout(10)
            self.wait_for_viewport(self.driver.capabilities["goog:processID"])
            h.set_driver(self.driver)
            h.Config.implicit_wait_secs = 2
            self._state = BrowserState(episode_id=episode_id or str(uuid.uuid4()))
            self.done = False
            error = ""
            try:
                h.go_to(start_url)
            except TimeoutException:
                error = "page_load_timeout"
            time.sleep(action_delay("reset"))
            return self.observe(error)
        except Exception:
            self.close()
            raise

    def step(self, action: BrowserAction, **kwargs):
        if not self.driver or self.done:
            raise ValueError("reset before stepping")
        self._state.step_count += 1
        error = ""
        try:
            if action.op == "click":
                self.desktop.click(action.x, action.y)
            elif action.op == "type":
                h.write(action.text)
            elif action.op == "key":
                keys = {
                    "ENTER": Keys.ENTER,
                    "TAB": Keys.TAB,
                    "ESCAPE": Keys.ESCAPE,
                    "BACKSPACE": Keys.BACKSPACE,
                    "CTRL+A": Keys.CONTROL + "a",
                    "UP": Keys.ARROW_UP,
                    "DOWN": Keys.ARROW_DOWN,
                    "LEFT": Keys.ARROW_LEFT,
                    "RIGHT": Keys.ARROW_RIGHT,
                }
                h.press(keys[action.text])
            elif action.op == "scroll":
                (h.scroll_down if action.dy >= 0 else h.scroll_up)(abs(action.dy))
            elif action.op == "back":
                self.driver.back()
            elif action.op == "finish":
                self.done = True
            if len(self.driver.window_handles) > 1:
                self.driver.switch_to.window(self.driver.window_handles[-1])
            time.sleep(action_delay(action.op))
        except Exception as exception:
            error = type(exception).__name__
        self.done |= self._state.step_count >= self.max_steps
        return self.observe(error)

    def observe(self, error=""):
        try:
            text = self.driver.execute_script("return document.body ? document.body.innerText : ''")
            if verification_block(text):
                self.done = True
                error = "blocked_by_bot_verification"
        except TimeoutException:
            error = error or "renderer_timeout"
        try:
            screenshot = self.driver.get_screenshot_as_base64()
        except TimeoutException:
            screenshot = ""
            self.done = True
            error = "renderer_timeout"
        return BrowserObservation(
            screenshot=screenshot,
            url=self.driver.current_url,
            error=error,
            done=self.done,
            reward=0,
        )

    def render_web(self, observation):
        """Draw the page screenshot with the URL and the last action error."""
        if not observation.get("screenshot"):
            return None
        parts = [
            '<div style="font-size:12px;opacity:0.7;word-break:break-all">'
            f"{html.escape(observation['url'])}</div>",
            f'<img alt="Helium Browser page" src="data:image/png;base64,{observation["screenshot"]}" '
            'style="display:block;width:520px;max-width:100%;border:1px solid var(--border-color-primary);border-radius:6px">',
        ]
        if observation["error"]:
            parts.append(
                f'<div style="color:var(--error-text-color)">{html.escape(observation["error"])}</div>'
            )
        return '<div style="display:flex;flex-direction:column;gap:6px">' + "".join(parts) + "</div>"

    @property
    def state(self):
        return self._state

    def close(self):
        try:
            if self.driver:
                self.driver.quit()
        finally:
            self.driver = None
            if self.desktop:
                self.desktop.close()
                self.desktop = None
            if self.profile:
                self.profile.cleanup()
                self.profile = None
