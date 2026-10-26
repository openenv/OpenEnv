# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Server implementation for the generic TextArena environment."""

from __future__ import annotations

import os
import re
import subprocess
import sys
import urllib.request
from typing import Any, Dict, Iterable, List, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment

try:
    # When running as installed package
    from textarena_env.models import (
        TextArenaAction,
        TextArenaMessage,
        TextArenaObservation,
        TextArenaState,
    )
    from textarena_env.rewards import build_reward_providers, RewardProvider
except ImportError:
    # When running uvicorn directly from textarena_env/
    from models import (
        TextArenaAction,
        TextArenaMessage,
        TextArenaObservation,
        TextArenaState,
    )
    from rewards import build_reward_providers, RewardProvider


_TEXTARENA_MODULE: Any | None = None
_TEXTARENA_IMPORT_ERROR: Exception | None = None
_NLTK_DOWNLOADED: bool = False

# TextArena's Wordle feedback, e.g. "You submitted [crane].\nFeedback:\nC R A N E\nG X Y X G".
_WORDLE_FEEDBACK = re.compile(
    r"You submitted \[(\w+)\]\.\nFeedback:\n[^\n]*\n([GYX ]+)"
)
_WORDLE_ATTEMPTS = re.compile(r"You have (\d+) attempts")
_WORDLE_WIN = re.compile(
    r"\[(\w+)\]\n\[GAME\] Congratulations! You guessed the word correctly!"
)
_WORDLE_TILES = {
    "G": "background:var(--color-accent);color:var(--background-fill-primary);border-color:var(--color-accent)",
    "Y": "color:var(--color-accent);border-color:var(--color-accent);border-style:dashed",
    "X": "background:var(--background-fill-primary);color:var(--body-text-color-subdued);border-color:var(--border-color-primary)",
    "-": "",
}


def _ensure_nltk_data() -> None:
    """Download NLTK data once per process."""
    global _NLTK_DOWNLOADED
    if _NLTK_DOWNLOADED:
        return
    try:
        import nltk
    except ImportError as exc:
        raise RuntimeError(
            "NLTK is required for TextArena environments. "
            "Install textarena_env dependencies (including nltk)."
        ) from exc
    packages = ("words", "averaged_perceptron_tagger_eng")
    opener = urllib.request._opener
    explicit_proxy = opener is not None and any(
        isinstance(handler, urllib.request.ProxyHandler)
        and any(scheme != "no" for scheme in handler.proxies)
        for handler in opener.handlers
    )
    if (
        nltk.__version__ == "3.10.3"
        and set(urllib.request.getproxies()) == {"no"}
        and not explicit_proxy
    ):
        # NLTK #3748 mistakes NO_PROXY alone for a carrying proxy. Isolate the
        # workaround so other server threads retain their proxy configuration.
        # Real proxies still use NLTK's normal, security-enforcing download path.
        download_env = os.environ.copy()
        download_env.pop("NO_PROXY", None)
        download_env.pop("no_proxy", None)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "nltk.downloader",
                "--quiet",
                "--exit-on-error",
                "--dir",
                nltk.downloader.Downloader().default_download_dir(),
                *packages,
            ],
            env=download_env,
            check=True,
            timeout=120,
        )
    else:
        for package in packages:
            nltk.download(package, quiet=True, raise_on_error=True)
    # NLTK 3.10.3's CLI can exit zero after a failed download. Verify the
    # resources in this process before caching successful initialization.
    nltk.data.find("corpora/words")
    nltk.data.find("taggers/averaged_perceptron_tagger_eng")
    _NLTK_DOWNLOADED = True


def _import_textarena() -> Any:
    """Import ``textarena`` lazily and cache the module reference."""

    global _TEXTARENA_MODULE, _TEXTARENA_IMPORT_ERROR

    if _TEXTARENA_MODULE is not None:
        return _TEXTARENA_MODULE

    if _TEXTARENA_IMPORT_ERROR is not None:
        raise _TEXTARENA_IMPORT_ERROR

    if sys.version_info < (3, 10):
        _TEXTARENA_IMPORT_ERROR = RuntimeError(
            "TextArena environments require Python 3.10 or newer; "
            f"current interpreter is {sys.version_info.major}.{sys.version_info.minor}"
        )
        raise _TEXTARENA_IMPORT_ERROR

    try:
        import textarena as ta  # type: ignore[import]
    except Exception as exc:  # pragma: no cover - surfaced to caller
        _TEXTARENA_IMPORT_ERROR = exc
        raise

    _TEXTARENA_MODULE = ta
    return ta


class TextArenaEnvironment(Environment):
    """Wrap any TextArena game behind the OpenEnv ``Environment`` API."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(
        self,
        env_id: str = "Wordle-v0",
        *,
        num_players: int = 1,
        max_turns: Optional[int] = None,
        download_nltk: bool = True,
        env_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()

        ta = _import_textarena()

        if download_nltk:
            _ensure_nltk_data()

        self.env_id = env_id
        self.num_players = num_players
        self.max_turns = max_turns
        self._env_kwargs = env_kwargs or {}

        self._ta_env = ta.make(env_id=env_id, **self._env_kwargs)

        self._state = TextArenaState(
            env_id=env_id,
            num_players=num_players,
            max_turns=max_turns,
        )

        self._reward_providers: List[RewardProvider] = build_reward_providers(env_id)
        self._last_reward_signals: Dict[str, float] = {}

        # Initialize environment state - TextArena envs require reset() to be called
        # before step() can be used, as the internal state object isn't created until reset.
        # This ensures the environment is always in a valid state after construction.
        self._ta_env.reset(num_players=self.num_players)

    # ------------------------------------------------------------------
    # Environment interface
    # ------------------------------------------------------------------
    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> TextArenaObservation:
        # TextArena observation wrappers (LLMObservationWrapper, etc.) accumulate
        # observations in self.full_observations across resets. Since we can't modify TextArena,
        # we need to manually clear this state to prevent history accumulation.
        env = self._ta_env
        while hasattr(env, "env"):
            if hasattr(env, "full_observations"):
                env.full_observations = {}
            env = env.env
        # Also check the final unwrapped env
        if hasattr(env, "full_observations"):
            env.full_observations = {}

        self._ta_env.reset(num_players=self.num_players)

        for provider in self._reward_providers:
            provider.reset()

        self._state.episode_id = episode_id if episode_id is not None else str(uuid4())
        self._state.step_count = 0
        self._state.turn = 0
        self._state.last_reward = 0.0
        self._state.last_info = {}
        self._state.raw_state = self._snapshot_state()
        self._last_reward_signals = {}

        observation = self._build_observation()
        observation.reward = 0.0
        observation.done = False

        return observation

    def step(self, action: TextArenaAction) -> TextArenaObservation:  # type: ignore[override]
        if not isinstance(action, TextArenaAction):
            raise TypeError(f"Expected TextArenaAction, received {type(action)!r}")

        done, info = self._ta_env.step(action.message)

        self._state.step_count += 1
        self._state.turn = getattr(self._ta_env.state, "turn", self._state.turn + 1)
        self._state.last_info = info or {}

        observation = self._build_observation()
        observation.done = done

        reward = self._extract_reward()
        observation.reward = reward
        self._state.last_reward = reward

        reward_signals = self._compute_reward_signals(
            action=action, observation=observation
        )
        if reward_signals:
            observation.info.setdefault("reward_signals", {}).update(reward_signals)
            observation.metadata.setdefault("reward_signals", {}).update(reward_signals)
        self._last_reward_signals = reward_signals
        if reward_signals:
            self._state.last_info = {
                **(self._state.last_info or {}),
                "reward_signals": reward_signals,
            }
        self._state.raw_state = self._snapshot_state()

        return observation

    def render_web(self, observation: Dict[str, Any]) -> Optional[str]:
        """Draw the Wordle board (one row per guess, empty rows for those left). Other games draw nothing."""
        if "messages" not in observation:
            return None
        text = "".join(message["content"] for message in observation["messages"])
        attempts = _WORDLE_ATTEMPTS.search(text)
        if attempts is None:
            return None
        guesses = _WORDLE_FEEDBACK.findall(text)
        length = len(guesses[0][0]) if guesses else 5
        rows = [(word.upper(), marks.split()) for word, marks in guesses]
        # A winning guess has no feedback line.
        won = _WORDLE_WIN.search(text)
        if won:
            length = len(won.group(1))
            rows.append((won.group(1).upper(), ["G"] * len(won.group(1))))
        rows += [(" " * length, ["-"] * length)] * (int(attempts.group(1)) - len(rows))
        tile = (
            '<span style="display:flex;align-items:center;justify-content:center;font-weight:700;font-size:18px;'
            'border:2px solid var(--border-color-primary);border-radius:4px;{}">{}</span>'
        )
        cells = [
            tile.format(_WORDLE_TILES[mark], letter)
            for word, marks in rows
            for letter, mark in zip(word, marks)
        ]
        return (
            f'<div role="img" aria-label="Wordle board" style="display:inline-grid;'
            f"grid-template-columns:repeat({length},38px);grid-auto-rows:38px;gap:5px;padding:10px;"
            'border:1px solid var(--border-color-primary);border-radius:10px;background:var(--background-fill-secondary)">'
            + "".join(cells)
            + "</div>"
        )

    @property
    def state(self) -> TextArenaState:
        return self._state

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _build_observation(self) -> TextArenaObservation:
        player_id, messages = self._ta_env.get_observation()

        # TextArena keeps how the game ended in game_info, out of the observation.
        state = self._ta_env.state
        reason = (
            state.game_info.get(player_id, {}).get("reason") if state.done else None
        )
        if reason and isinstance(messages, str):
            # LLMObservationWrapper clients diff this single cumulative message.
            messages += f"\n[GAME] {reason}"

        ta_messages = self._convert_messages(messages)
        if reason and not isinstance(messages, str):
            ta_messages.append(
                TextArenaMessage(
                    sender_id=-1,
                    content=f"\n[GAME] {reason}",
                    category="GAME_ADMIN",
                )
            )

        # Extract prompt from the appropriate messages.
        # TextArena PROMPT type messages contain the game instructions added during reset.
        # As a fallback for environments that don't use typed messages, use only the first
        # message if we're at turn 0 (fresh reset).
        prompt_lines = [msg.content for msg in ta_messages if msg.category == "PROMPT"]

        if not prompt_lines:
            # Fallback: use the first message only if at turn 0 (just after reset)
            # DO NOT use all messages as this causes history accumulation
            current_turn = getattr(self._ta_env.state, "turn", 0)
            if current_turn == 0 and ta_messages:
                prompt_lines = [ta_messages[0].content]
            else:
                # Use env_id as final fallback to avoid including game history
                prompt_lines = [self.env_id]

        prompt = "\n".join(prompt_lines).strip()

        info: Dict[str, Any] = {}
        info.update(getattr(self._ta_env.state, "step_info", {}))

        observation = TextArenaObservation(
            prompt=prompt,
            messages=ta_messages,
            current_player_id=player_id,
            legal_players=self._legal_players(),
            info=info,
            metadata={
                "env_id": self.env_id,
                "turn": getattr(self._ta_env.state, "turn", 0),
                "raw_messages": [
                    {
                        "sender_id": msg.sender_id,
                        "content": msg.content,
                        "category": msg.category,
                    }
                    for msg in ta_messages
                ],
            },
        )

        return observation

    def _legal_players(self) -> List[int]:
        role_mapping = getattr(self._ta_env.state, "role_mapping", {}) or {}
        players = [
            pid for pid in role_mapping.keys() if isinstance(pid, int) and pid >= 0
        ]
        return sorted(players)

    def _convert_messages(self, messages: Iterable[Any]) -> List[TextArenaMessage]:
        converted: List[TextArenaMessage] = []
        buffered_sender: int | None = None
        buffered_category: str | None = None
        buffered_content: List[str] = []

        def flush_buffer() -> None:
            nonlocal buffered_content, buffered_sender, buffered_category
            if not buffered_content:
                return
            converted.append(
                TextArenaMessage(
                    sender_id=buffered_sender if buffered_sender is not None else -1,
                    content="".join(buffered_content),
                    category=buffered_category or "MESSAGE",
                )
            )
            buffered_content = []
            buffered_category = None
            buffered_sender = None

        for entry in messages:
            if isinstance(entry, tuple) and len(entry) == 3:
                sender, content, category = entry
            elif isinstance(entry, tuple) and len(entry) == 2:
                sender, content = entry
                category = "MESSAGE"
            else:
                sender, content, category = -1, str(entry), "MESSAGE"

            category_name = getattr(category, "name", str(category))
            sender_id = int(sender) if isinstance(sender, (int, float)) else -1
            text = str(content)

            if (
                buffered_content
                and buffered_category == category_name
                and buffered_sender == sender_id
            ):
                buffered_content.append(text)
            else:
                flush_buffer()
                buffered_sender = sender_id
                buffered_category = category_name
                buffered_content = [text]

        flush_buffer()

        return converted

    def _extract_reward(self) -> float:
        rewards = getattr(self._ta_env.state, "rewards", None)
        if isinstance(rewards, dict):
            # Use current player reward if available, otherwise default to player 0.
            player_id = getattr(self._ta_env.state, "current_player_id", 0)
            if player_id in rewards:
                return float(rewards[player_id])
            if 0 in rewards:
                return float(rewards[0])
        return 0.0

    def _snapshot_state(self) -> Dict[str, Any]:
        state = self._ta_env.state
        snapshot: Dict[str, Any] = {
            "turn": getattr(state, "turn", 0),
            "game_state": getattr(state, "game_state", {}),
            "logs": list(getattr(state, "logs", [])),
            "rewards": getattr(state, "rewards", None),
            "done": getattr(state, "done", False),
            "role_mapping": getattr(state, "role_mapping", {}),
            "game_info": getattr(state, "game_info", {}),
            "step_info": getattr(state, "step_info", {}),
        }
        if self._last_reward_signals:
            snapshot["reward_signals"] = dict(self._last_reward_signals)
        return snapshot

    def _compute_reward_signals(
        self, *, action: TextArenaAction, observation: TextArenaObservation
    ) -> Dict[str, float]:
        if not self._reward_providers:
            return {}

        aggregated: Dict[str, float] = {}
        for provider in self._reward_providers:
            try:
                result = provider.compute(action=action, observation=observation)
            except Exception:  # pragma: no cover - defensive
                continue
            for key, value in result.items():
                aggregated[key] = float(value)
        return aggregated
