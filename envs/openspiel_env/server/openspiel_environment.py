# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
OpenSpiel Environment Server Implementation.

This module wraps OpenSpiel's rl_environment.Environment and exposes it
via the OpenEnv Environment interface.
"""

import uuid
from typing import Any, Dict, List, Optional, Tuple

# Support both in-repo and standalone imports
try:
    # In-repo imports (when running from OpenEnv repository)
    from openenv.core.env_server.interfaces import Environment

    from ..models import OpenSpielAction, OpenSpielObservation, OpenSpielState
    from .opponent_policies import get_opponent_policy, OpponentPolicy
except ImportError:
    from models import OpenSpielAction, OpenSpielObservation, OpenSpielState

    # Standalone imports (when environment is standalone with openenv from pip)
    from openenv.core.env_server.interfaces import Environment
    from server.opponent_policies import get_opponent_policy, OpponentPolicy

# Import OpenSpiel
try:
    import pyspiel
    from open_spiel.python import rl_environment
except ImportError as e:
    raise ImportError(
        "OpenSpiel is not installed. "
        "Please install it following instructions at: "
        "https://github.com/google-deepmind/open_spiel"
    ) from e


class OpenSpielEnvironment(Environment):
    """
    OpenSpiel Environment wrapper for OpenEnv.

    This environment wraps OpenSpiel games and provides a single-agent interface.
    For multi-player games, the agent controls one player while opponent(s) use
    a fixed policy (e.g., random).

    Supported games:
    - Single-player: catch, cliff_walking, 2048, blackjack
    - Multi-player: tic_tac_toe, kuhn_poker, connect_four

    Args:
        game_name: Name of the OpenSpiel game (e.g., "catch", "tic_tac_toe").
        agent_player: Which player ID the agent controls (default 0).
        opponent_policy: Policy for opponent players ("random", "first", etc.).
        game_params: Optional game-specific parameters.

    Example:
        >>> env = OpenSpielEnvironment("catch")
        >>> obs = env.reset()
        >>> print(obs.info_state)  # Agent's observation
        >>> obs = env.step(OpenSpielAction(action_id=1))
        >>> print(obs.reward)
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(
        self,
        game_name: str = "catch",
        agent_player: int = 0,
        opponent_policy: str = "random",
        game_params: Dict[str, Any] | None = None,
    ):
        """Initialize OpenSpiel environment."""
        super().__init__()

        self.game_name = game_name
        self.agent_player = agent_player
        self.game_params = game_params or {}

        # Create OpenSpiel environment
        try:
            self._ospiel_env = rl_environment.Environment(game_name, **self.game_params)
        except Exception as e:
            raise ValueError(
                f"Failed to create OpenSpiel game '{game_name}': {e}"
            ) from e

        self.num_players = self._ospiel_env.num_players
        self.is_turn_based = self._ospiel_env.is_turn_based

        # Validate agent_player
        if agent_player >= self.num_players:
            raise ValueError(
                f"agent_player={agent_player} >= num_players={self.num_players}"
            )

        # Set up opponent policy for multi-player games
        self.opponent_policy_fn: OpponentPolicy | None = None
        if self.num_players > 1:
            self.opponent_policy_fn = get_opponent_policy(opponent_policy)

        # Initialize state
        self._state = OpenSpielState(
            game_name=game_name,
            agent_player=agent_player,
            opponent_policy=opponent_policy,
            game_params=self.game_params,
            num_players=self.num_players,
        )

        # Track last opponent action for learning
        self._last_opponent_action: int | None = None

    def web_actions(
        self, observation: Dict[str, Any]
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """The legal moves as buttons, named for the games this env draws."""
        labels = _BUTTON_LABELS.get(self.game_name, [])
        return [
            (labels[a] if a < len(labels) else str(a), {"action_id": a})
            for a in observation.get("legal_actions") or []
        ]

    def render_web(self, observation: Dict[str, Any]) -> Optional[str]:
        """Draw the board or hand of Catch, 2048, Tic-Tac-Toe, Connect Four, Blackjack, Kuhn Poker or Cliff Walking."""
        draw = _DRAWINGS.get(self.game_name)
        return draw(observation) if draw else None

    def reset(self) -> OpenSpielObservation:
        """
        Reset the environment and return initial observation.

        For multi-player games, this will autoplay opponent turns until
        it's the agent's turn (or terminal state).

        Returns:
            Initial observation for the agent.
        """
        # Reset OpenSpiel environment
        time_step = self._ospiel_env.reset()

        # Reset state tracking
        self._state.episode_id = str(uuid.uuid4())
        self._state.step_count = 0
        self._last_opponent_action = None

        # Autoplay opponent turns until agent's turn
        time_step = self._auto_play_opponents(time_step)

        # Convert to OpenEnv observation
        return self._make_observation(time_step)

    def step(self, action: OpenSpielAction) -> OpenSpielObservation:  # type: ignore[override]
        """
        Execute agent's action and return resulting observation.

        For multi-player games, this will:
        1. Apply the agent's action
        2. Autoplay opponent turns until it's the agent's turn again
        3. Return the observation from the agent's perspective

        Args:
            action: OpenSpielAction containing the action_id to execute.

        Returns:
            Observation after action execution (and opponent turns if multi-player).

        Raises:
            ValueError: If action is not an OpenSpielAction.
        """
        if not isinstance(action, OpenSpielAction):
            raise ValueError(f"Expected OpenSpielAction, got {type(action)}")

        # Apply agent's action
        if self.is_turn_based:
            # Turn-based: single action
            time_step = self._ospiel_env.step([action.action_id])
        else:
            # Simultaneous-move: need actions for all players
            # For now, only support agent as player 0 in simultaneous games
            if self.agent_player != 0:
                raise NotImplementedError(
                    "Simultaneous-move games only support agent_player=0"
                )
            # Get opponent actions
            opponent_actions = []
            for player_id in range(self.num_players):
                if player_id == self.agent_player:
                    opponent_actions.append(action.action_id)
                else:
                    legal_actions = time_step.observations["legal_actions"][player_id]
                    opp_action = self.opponent_policy_fn.select_action(
                        legal_actions, time_step.observations
                    )
                    opponent_actions.append(opp_action)
            time_step = self._ospiel_env.step(opponent_actions)

        self._state.step_count += 1

        # Autoplay opponent turns (for turn-based games)
        if self.is_turn_based:
            time_step = self._auto_play_opponents(time_step)

        # Convert to OpenEnv observation
        return self._make_observation(time_step)

    @property
    def state(self) -> OpenSpielState:
        """Get current environment state."""
        return self._state

    def _auto_play_opponents(self, time_step) -> Any:
        """
        Autoplay opponent turns until it's the agent's turn or game is terminal.

        Args:
            time_step: Current TimeStep from OpenSpiel environment.

        Returns:
            Updated TimeStep after opponent moves.
        """
        # Single-player games: nothing to do
        if self.num_players == 1:
            return time_step

        # Multi-player games: play opponent turns
        while (
            not time_step.last()
            and time_step.observations["current_player"] != self.agent_player
        ):
            current_player = time_step.observations["current_player"]
            legal_actions = time_step.observations["legal_actions"][current_player]

            # Select opponent action
            opp_action = self.opponent_policy_fn.select_action(
                legal_actions, time_step.observations
            )
            self._last_opponent_action = opp_action

            # Apply opponent action
            time_step = self._ospiel_env.step([opp_action])
            self._state.step_count += 1

        return time_step

    def _make_observation(self, time_step) -> OpenSpielObservation:
        """
        Convert OpenSpiel TimeStep to OpenEnv Observation.

        Args:
            time_step: OpenSpiel TimeStep object.

        Returns:
            OpenSpielObservation for the agent.
        """
        # Extract agent's information
        info_state = time_step.observations["info_state"][self.agent_player]
        legal_actions = time_step.observations["legal_actions"][self.agent_player]
        current_player_id = time_step.observations["current_player"]

        # Determine game phase
        if time_step.last():
            game_phase = "terminal"
        elif time_step.first():
            game_phase = "initial"
        else:
            game_phase = "playing"

        # Get reward for agent
        reward = None
        if time_step.rewards is not None:
            reward = float(time_step.rewards[self.agent_player])

        # Create observation
        obs = OpenSpielObservation(
            info_state=info_state.tolist()
            if hasattr(info_state, "tolist")
            else list(info_state),
            legal_actions=legal_actions,
            game_phase=game_phase,
            current_player_id=current_player_id,
            opponent_last_action=self._last_opponent_action,
            done=time_step.last(),
            reward=reward,
        )

        return obs


_BUTTON_LABELS = {
    "catch": ["0 · left", "1 · stay", "2 · right"],
    "2048": ["0 · up", "1 · right", "2 · down", "3 · left"],
    "tic_tac_toe": [
        "0 · top left",
        "1 · top centre",
        "2 · top right",
        "3 · middle left",
        "4 · centre",
        "5 · middle right",
        "6 · bottom left",
        "7 · bottom centre",
        "8 · bottom right",
    ],
    "connect_four": [f"col {c}" for c in range(7)],  # same labels as connect4_env
    "blackjack": ["0 · hit", "1 · stand"],
    "kuhn_poker": ["0 · pass", "1 · bet"],
    "cliff_walking": ["0 · right", "1 · up", "2 · left", "3 · down"],
}


def _grid(
    label: str, columns: int, cells: List[str], size: int = 26, gap: int = 3
) -> str:
    """Lay out square cells in a bordered grid labelled for screen readers."""
    return (
        f'<div role="img" aria-label="{label}" style="display:inline-grid;'
        f"grid-template-columns:repeat({columns},{size}px);grid-auto-rows:{size}px;gap:{gap}px;padding:10px;"
        'border:1px solid var(--border-color-primary);border-radius:10px;background:var(--background-fill-secondary)">'
        + "".join(cells)
        + "</div>"
    )


def _cell(background: str, radius: str = "4px", text: str = "", style: str = "") -> str:
    """One grid cell with a theme colour background and optional centred text."""
    return (
        f'<span style="background:{background};border-radius:{radius};display:flex;'
        f'align-items:center;justify-content:center;{style}">{text}</span>'
    )


def _draw_catch(observation: Dict[str, Any]) -> Optional[str]:
    """Catch: 10 rows x 5 columns, the ball falls towards the paddle on the bottom row."""
    state = observation.get("info_state") or []
    if len(state) != 50:
        return None
    cells = []
    for i, value in enumerate(state):
        if not value:
            cells.append(_cell("var(--border-color-primary)"))
        elif i >= 45:
            cells.append(_cell("var(--body-text-color)"))
        else:
            cells.append(_cell("var(--color-accent)", "50%"))
    return _grid("Catch board", 5, cells)


def _draw_2048(observation: Dict[str, Any]) -> Optional[str]:
    """2048: 4x4 tiles, the shade grows with the tile value."""
    state = observation.get("info_state") or []
    if len(state) != 16:
        return None
    cells = []
    for value in state:
        value = int(value)
        if not value:
            cells.append(_cell("var(--border-color-primary)", "6px"))
            continue
        shade = min(12 * (value.bit_length() - 1), 96)
        text = (
            "var(--body-text-color)"
            if shade < 55
            else "var(--button-primary-text-color)"
        )
        cells.append(
            _cell(
                f"color-mix(in srgb,var(--color-accent) {shade}%,var(--background-fill-primary))",
                "6px",
                str(value),
                f"color:{text};font-weight:700;font-size:{18 if value < 1000 else 15}px",
            )
        )
    return _grid("2048 board", 4, cells, 56)


def _draw_tic_tac_toe(observation: Dict[str, Any]) -> Optional[str]:
    """Tic-Tac-Toe: 3x3 board, x in the accent colour and o in the text colour."""
    state = observation.get("info_state") or []
    if len(state) != 27:  # one-hot planes: empty, o, x
        return None
    cells = []
    for i in range(9):
        mark, colour = (
            ("x", "var(--color-accent)")
            if state[18 + i]
            else ("o", "var(--body-text-color)")
        )
        cells.append(
            _cell(
                "var(--border-color-primary)",
                "6px",
                mark if not state[i] else "",
                f"color:{colour};font-weight:700;font-size:30px",
            )
        )
    return _grid("Tic-Tac-Toe board", 3, cells, 52)


def _draw_connect_four(observation: Dict[str, Any]) -> Optional[str]:
    """Connect Four: 6x7 board, x in the accent colour and o in the text colour, columns numbered below."""
    state = observation.get("info_state") or []
    if len(state) != 126:  # one-hot planes: x, o, empty; row 0 is the bottom
        return None
    cells = []
    for row in reversed(range(6)):
        for col in range(7):
            i = row * 7 + col
            if state[i]:
                cells.append(_cell("var(--color-accent)", "50%"))
            elif state[42 + i]:
                cells.append(_cell("var(--body-text-color)", "50%"))
            else:
                cells.append(_cell("var(--border-color-primary)", "50%"))
    cells += [
        _cell("transparent", text=str(col), style="font-size:12px") for col in range(7)
    ]
    return _grid("Connect Four board", 7, cells, gap=4)  # same gap as connect4_env


def _card(text: str) -> str:
    """A playing card face with its rank and suit, or ?? when hidden."""
    colour = "var(--color-accent)" if text[-1] in "♥♦" else "var(--body-text-color)"
    return (
        '<span style="display:inline-flex;align-items:center;justify-content:center;width:38px;height:54px;'
        "border:1px solid var(--border-color-primary);border-radius:6px;"
        f'background:var(--background-fill-primary);color:{colour};font-weight:700;font-size:16px">{text}</span>'
    )


def _hand(label: str, title: str, rows: List[Tuple[str, List[str]]]) -> str:
    """Rows of labelled cards in a bordered box labelled for screen readers."""
    body = "".join(
        f'<div style="display:flex;align-items:center;gap:6px;margin:4px 0">'
        f'<span style="width:90px;font-size:13px">{name}</span>{"".join(_card(c) for c in cards)}</div>'
        for name, cards in rows
    )
    return (
        f'<div role="img" aria-label="{label}" style="display:inline-block;padding:10px 14px;'
        "border:1px solid var(--border-color-primary);border-radius:10px;background:var(--background-fill-secondary);"
        f'color:var(--body-text-color)"><div style="font-size:13px;margin-bottom:4px">{title}</div>{body}</div>'
    )


def _draw_blackjack(observation: Dict[str, Any]) -> Optional[str]:
    """Blackjack: your cards with their total, and the dealer's visible cards."""
    state = observation.get("info_state") or []
    if (
        len(state) != 189
    ):  # cards one-hot from 85 (player) and 137 (dealer), suits C, D, H, S
        return None
    player = [c for c in range(52) if state[85 + c]]
    dealer = [c for c in range(52) if state[137 + c]]
    total = sum(min(c % 13 + 1, 10) for c in player)
    if total <= 11 and any(c % 13 == 0 for c in player):
        total += 10
    ranks = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]
    dealer_cards = [ranks[c % 13] + "♣♦♥♠"[c // 13] for c in dealer]
    if observation.get("game_phase") != "terminal":
        dealer_cards.append("??")
    return _hand(
        "Blackjack board",
        "Blackjack",
        [
            (f"You ({total})", [ranks[c % 13] + "♣♦♥♠"[c // 13] for c in player]),
            ("Dealer", dealer_cards),
        ],
    )


def _draw_kuhn_poker(observation: Dict[str, Any]) -> Optional[str]:
    """Kuhn Poker: your card, the opponent's hidden card and the bets so far."""
    state = observation.get("info_state") or []
    if (
        len(state) != 11
    ):  # player one-hot, card one-hot (J, Q, K), then pass/bet for 3 rounds
        return None
    card = "JQK"[state[2:5].index(1)]
    bets = [
        ("pass", "bet")[int(state[6 + 2 * r])]
        for r in range(3)
        if state[5 + 2 * r] or state[6 + 2 * r]
    ]
    return _hand(
        "Kuhn Poker board",
        "Bets: " + (" → ".join(bets) or "none yet"),
        [("You", [card]), ("Opponent", ["?"])],
    )


def _draw_cliff_walking(observation: Dict[str, Any]) -> Optional[str]:
    """Cliff Walking: 4x8 grid, the walker starts bottom left, the cliff runs to the goal bottom right."""
    # The info state is a one-hot action history over 100 steps: right, up, left, down.
    state = observation.get("info_state") or []
    if len(state) != 400:
        return None
    row, col = 3, 0
    for i, value in enumerate(state):
        if value:
            d_row, d_col = ((0, 1), (-1, 0), (0, -1), (1, 0))[i % 4]
            row, col = min(max(row + d_row, 0), 3), min(max(col + d_col, 0), 7)
    cells = []
    for r in range(4):
        for c in range(8):
            if (r, c) == (row, col):
                cells.append(_cell("var(--color-accent)", "50%"))
            elif r == 3 and c in (0, 7):
                cells.append(
                    _cell(
                        "var(--border-color-primary)",
                        text="S" if c == 0 else "G",
                        style="font-weight:700",
                    )
                )
            elif r == 3 and c > 0:
                cells.append(_cell("var(--body-text-color)"))
            else:
                cells.append(_cell("var(--border-color-primary)"))
    return _grid("Cliff Walking board", 8, cells)


_DRAWINGS = {
    "catch": _draw_catch,
    "2048": _draw_2048,
    "tic_tac_toe": _draw_tic_tac_toe,
    "connect_four": _draw_connect_four,
    "blackjack": _draw_blackjack,
    "kuhn_poker": _draw_kuhn_poker,
    "cliff_walking": _draw_cliff_walking,
}
