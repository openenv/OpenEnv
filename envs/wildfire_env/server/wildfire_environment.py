import os
import random
import uuid
from typing import Any, Dict, Optional

from openenv.core.env_server.interfaces import Environment

from ..models import WildfireAction, WildfireObservation, WildfireState

# Helpers
DIRS_8 = {
    "N": (0, -1),
    "NE": (1, -1),
    "E": (1, 0),
    "SE": (1, 1),
    "S": (0, 1),
    "SW": (-1, 1),
    "W": (-1, 0),
    "NW": (-1, -1),
    "CALM": (0, 0),
}

# Web drawing: (label, cell style) per grid code. Fire and water need their own hues
# (no theme variable is orange or blue); the rest follow the theme.
CELL_STYLES = {
    2: ("burning", "background:hsl(16 90% 52%)"),
    1: ("fuel", "background:var(--color-accent);opacity:.45"),
    4: ("water", "background:hsl(210 80% 55%)"),
    3: ("firebreak", "background:var(--body-text-color)"),
    0: ("ash", "background:var(--body-text-color);opacity:.25"),
}
WIND_ARROWS = {
    "N": "↑",
    "NE": "↗",
    "E": "→",
    "SE": "↘",
    "S": "↓",
    "SW": "↙",
    "W": "←",
    "NW": "↖",
    "CALM": "·",
}


def idx(x: int, y: int, w: int) -> int:
    # Defensive type conversion to ensure all parameters are integers
    x, y, w = int(x), int(y), int(w)
    return y * w + x


def in_bounds(x: int, y: int, w: int, h: int) -> bool:
    # Defensive type conversion to ensure all parameters are integers
    x, y, w, h = int(x), int(y), int(w), int(h)
    return 0 <= x < w and 0 <= y < h


class WildfireEnvironment(Environment):
    """
    Weather-aware wildfire simulation.

    Grid encodings:
      0 = ash (burned out)
      1 = fuel / vegetation
      2 = burning
      3 = firebreak
      4 = watered / damp

    Each step:
      - agent acts (water/break/wait)
      - burning spreads to neighbors with wind + humidity effects
      - burning cells burn for multiple ticks, then become ash
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(
        self,
        width: int = 32,
        height: int = 32,
        base_ignite_prob: float = 0.30,
        wind_bias: float = 0.20,  # kept for compatibility (not directly used in B model)
        diag_factor: float = 0.7,  # kept for compatibility (not directly used in B model)
        humidity: float = 0.25,
        init_sources: int = 2,
        seed: int = 3407,
        max_steps: int = 128,
        water_capacity: int = 8,  # ↓ encourage strategic water use
        break_capacity: int = 50,
    ):
        super().__init__()

        # --- Env-var overrides (optional) ---
        width = int(os.environ.get("WILDFIRE_WIDTH", width))
        height = int(os.environ.get("WILDFIRE_HEIGHT", height))
        humidity = float(os.environ.get("WILDFIRE_HUMIDITY", humidity))
        forced_wind = os.environ.get("WILDFIRE_WIND", None)

        # Store config (ensure integers)
        self.w = int(width)
        self.h = int(height)
        self.base_ignite_prob = base_ignite_prob
        self.wind_bias = wind_bias
        self.diag_factor = diag_factor
        self.init_humidity = humidity
        self.init_sources = init_sources
        self.rng = random.Random(seed)
        self.max_steps = max_steps
        self.init_water = water_capacity
        self.init_breaks = break_capacity
        self.forced_wind = forced_wind

        # burn lifetime in ticks (balanced model)
        self.burn_lifetime = 3

        # Initialize state with minimal defaults (will be properly set in reset())
        # We can't use WildfireState() directly due to Pydantic/dataclass conflicts,
        # so we'll initialize it in reset() and handle None case in state property
        self._state: WildfireState | None = None

    # --- Core API ---

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        **kwargs: Any,
    ) -> WildfireObservation:
        """
        Start a new episode: a grid of fuel with new fires, wind and humidity.

        Args:
            seed (`int`, *optional*):
                Reseeds the wind, humidity, starting fires and fire spread. The same
                seed and actions give the same episode, and the same episodes on the
                unseeded resets that follow.
            episode_id (`str`, *optional*):
                Episode identifier. A UUID is generated when it is not given.
        """
        if seed is not None:
            self.rng = random.Random(seed)

        # Ensure w and h are integers (defensive type conversion)
        w, h = int(self.w), int(self.h)

        # Start with all fuel
        grid = [1] * (w * h)

        # Wind (forced if provided)
        if self.forced_wind and self.forced_wind in DIRS_8:
            wind_dir = self.forced_wind
        else:
            wind_dir = self.rng.choice(list(DIRS_8.keys()))

        # Humidity small variation around init
        humidity = min(
            1.0, max(0.0, self.init_humidity + self.rng.uniform(-0.05, 0.05))
        )

        # Place initial fires
        for _ in range(self.init_sources):
            x = self.rng.randrange(w)
            y = self.rng.randrange(h)
            i = idx(x, y, w)
            # Safety check: ensure index is within grid bounds
            if 0 <= i < len(grid):
                grid[i] = 2

        # Initialize burn timers before creating state
        burn_timers = [0] * (w * h)

        # Use model_construct to bypass Pydantic validation for dataclass/Pydantic compatibility
        self._state = WildfireState.model_construct(
            episode_id=episode_id if episode_id is not None else str(uuid.uuid4()),
            step_count=0,
            total_burned=0,
            total_extinguished=0,
            last_action="reset",
            width=w,
            height=h,
            wind_dir=wind_dir,
            humidity=humidity,
            remaining_water=self.init_water,
            remaining_breaks=self.init_breaks,
            grid=grid,
            burn_timers=burn_timers,
        )

        obs = self._make_observation(reward_hint=0.0)
        return obs

    def step(self, action: WildfireAction) -> WildfireObservation:
        st = self._state
        reward = 0.0

        # --- Agent action effects ---
        if (
            action.action == "water"
            and st.remaining_water > 0
            and action.x is not None
            and action.y is not None
        ):
            reward += self._apply_water(action.x, action.y)
        elif (
            action.action == "break"
            and st.remaining_breaks > 0
            and action.x is not None
            and action.y is not None
        ):
            reward += self._apply_break(action.x, action.y)
        elif action.action == "wait":
            pass
        else:
            reward -= 0.05  # invalid or exhausted resources

        # --- Natural fire dynamics ---
        prev_burning = self._burning_count()
        prev_burned = sum(1 for v in st.grid if v == 0)

        newly_burned = self._spread_fire()
        new_burning = self._burning_count()
        now_burned = sum(1 for v in st.grid if v == 0)

        st.total_burned += newly_burned
        st.step_count += 1
        st.last_action = action.action

        # --- Spread vs containment shaping ---
        spread_delta = new_burning - prev_burning
        burned_delta = now_burned - prev_burned

        # Strong penalty for spread
        if spread_delta > 0:
            reward -= 0.15 * spread_delta  # 🔥 focus on containment
        elif spread_delta < 0:
            reward += 0.10 * abs(spread_delta)  # reward shrinkage

        # Mild penalty for newly burned cells (area loss)
        if burned_delta > 0:
            reward -= 0.05 * burned_delta

        # Small time penalty to prefer fast control
        reward -= 0.01

        done = self._is_done()

        # --- End of episode bonuses ---
        if done:
            saved_ratio = self._saved_cells() / (self.w * self.h)
            burned_ratio = now_burned / (self.w * self.h)
            burning_left = self._burning_count()

            # Big containment bonus
            if burning_left == 0:
                reward += 0.5 + 0.5 * saved_ratio

            # Fallback proportional reward
            reward += 0.2 * (1.0 - burned_ratio)

        obs = self._make_observation(reward_hint=reward)
        obs.done = done
        obs.reward = reward
        return obs

    def render_web(self, observation: Dict[str, Any]) -> Optional[str]:
        """Draw the fire grid (one colour per cell state) with a legend, wind, humidity and resources."""
        if "grid" not in observation:
            return None
        width = observation["width"]
        size = max(6, min(22, 380 // width))
        max_width = f"max-width:{width * (size + 2)}px"
        cells = "".join(
            f'<span title="x={i % width}, y={i // width}: {CELL_STYLES[v][0]}" '
            f'style="{CELL_STYLES[v][1]};border-radius:3px"></span>'
            for i, v in enumerate(observation["grid"])
        )
        legend = "".join(
            f'<span style="display:inline-flex;align-items:center;gap:5px">'
            f'<span style="width:11px;height:11px;border-radius:3px;{style}"></span>{label}</span>'
            for label, style in CELL_STYLES.values()
        )
        wind = observation["wind_dir"]
        stats = " · ".join(
            f'<span style="white-space:nowrap">{item}</span>'
            for item in (
                f"step {observation['step']}",
                f"wind {wind} {WIND_ARROWS[wind]}",
                f"humidity {observation['humidity']:.2f}",
                f"water {observation['remaining_water']}",
                f"breaks {observation['remaining_breaks']}",
                f"burning {observation['burning_count']}",
                f"ash {observation['burned_count']}",
            )
        )
        return (
            '<div style="display:inline-flex;flex-direction:column;gap:8px;'
            "padding:10px;border:1px solid var(--border-color-primary);border-radius:10px;"
            'background:var(--background-fill-secondary);color:var(--body-text-color);font-size:12px">'
            f'<div role="img" aria-label="Wildfire board" style="display:grid;grid-template-columns:repeat({width},{size}px);'
            f'grid-auto-rows:{size}px;gap:2px">{cells}</div>'
            f'<div style="display:flex;flex-wrap:wrap;gap:4px 12px;{max_width}">{legend}</div>'
            f'<div style="font-family:var(--font-mono);color:var(--body-text-color-subdued);{max_width}">{stats}</div>'
            "</div>"
        )

    # --- Internal mechanics ---

    def _apply_water(self, x: int, y: int) -> float:
        st = self._state
        # Ensure x and y are integers (defensive type conversion)
        x, y = int(x), int(y)
        if not in_bounds(x, y, self.w, self.h):
            return -0.05

        # Strong penalty if no water left
        if st.remaining_water <= 0:
            return -0.5

        i = idx(x, y, self.w)
        # Safety check: ensure index is within grid bounds
        if i < 0 or i >= len(st.grid):
            return -0.05

        reward = 0.0

        if st.grid[i] == 2:
            st.grid[i] = 4  # extinguish & dampen
            st.burn_timers[i] = 0
            st.total_extinguished += 1
            reward += 0.25
        elif st.grid[i] == 1:
            st.grid[i] = 4  # dampen fuel (mild penalty to avoid spamming)
            st.burn_timers[i] = 0
            reward -= 0.10
        elif st.grid[i] == 4:
            # redundant watering
            reward -= 0.05
        else:
            # watering ash/break gives slight penalty
            reward -= 0.05

        st.remaining_water -= 1
        return reward

    def _apply_break(self, x: int, y: int) -> float:
        st = self._state
        # Ensure x and y are integers (defensive type conversion)
        x, y = int(x), int(y)
        if not in_bounds(x, y, self.w, self.h):
            return -0.05
        i = idx(x, y, self.w)
        # Safety check: ensure index is within grid bounds
        if i < 0 or i >= len(st.grid):
            return -0.05

        reward = 0.0

        if st.grid[i] in (1, 4):
            st.grid[i] = 3
            st.burn_timers[i] = 0
            reward += 0.15  # slightly more than before to make firebreaks attractive
        elif st.grid[i] == 2:
            st.grid[i] = 3
            st.burn_timers[i] = 0
            reward -= 0.02
        elif st.grid[i] == 3:
            reward -= 0.01
        else:
            reward -= 0.02

        st.remaining_breaks -= 1
        return reward

    def _spread_fire(self) -> int:
        """
        Balanced wildfire spread model:
          - burning cells persist for multiple ticks before turning to ash
          - 8-direction spread (diagonals weaker)
          - wind accelerates in wind direction, weakens upwind
          - humidity suppresses ignition probability
          - water (4) is IMMUNE to ignition while damp and reverts to fuel after several ticks
        """
        st = self._state
        new_grid = st.grid[:]
        newly_burned = 0

        # Ensure w and h are integers (defensive type conversion)
        w, h = int(self.w), int(self.h)

        # 8-neighbor model
        neighbors = [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
            (-1, -1),
            (1, -1),
            (-1, 1),
            (1, 1),
        ]
        wx, wy = DIRS_8.get(st.wind_dir, (0, 0))

        base = self.base_ignite_prob
        humidity_factor = 1.0 - st.humidity

        ignite_flags = [False] * (w * h)

        # First pass: evaluate ignitions, increment burn timers
        for y in range(h):
            for x in range(w):
                i = idx(x, y, w)
                # Safety check: ensure index is within grid bounds
                if i < 0 or i >= len(st.grid):
                    continue
                cell = st.grid[i]

                if cell == 2:  # burning
                    st.burn_timers[i] += 1

                    for dx, dy in neighbors:
                        nx, ny = x + dx, y + dy
                        if not in_bounds(nx, ny, w, h):
                            continue
                        ni = idx(nx, ny, w)
                        # Safety check: ensure neighbor index is within grid bounds
                        if ni < 0 or ni >= len(st.grid):
                            continue
                        target = st.grid[ni]

                        # Only fuel or water/damp can be candidates, but cells with code 4 (watered/damp) are immune to ignition
                        if target == 4:
                            # Watered/damp cells (code 4) do not ignite at all while in this state
                            continue
                        if target != 1:
                            continue

                        # Wind multiplier
                        if (dx, dy) == (wx, wy):
                            wind_mult = 2.0
                        elif (dx, dy) == (-wx, -wy):
                            wind_mult = 0.5
                        else:
                            wind_mult = 1.0

                        # Diagonals weaker
                        diag_mult = 0.6 if (dx != 0 and dy != 0) else 1.0

                        p = base * humidity_factor * wind_mult * diag_mult
                        p = max(0.0, min(1.0, p))
                        if self.rng.random() < p:
                            # Safety check: ensure ni is within ignite_flags bounds
                            if 0 <= ni < len(ignite_flags):
                                ignite_flags[ni] = True

        # Second pass: apply transitions
        for i, cell in enumerate(st.grid):
            # Safety check: ensure index is within bounds for all arrays
            if i < 0 or i >= len(new_grid) or i >= len(st.burn_timers):
                continue

            if cell == 2:
                # burns for burn_lifetime ticks before turning to ash
                if st.burn_timers[i] >= self.burn_lifetime:
                    new_grid[i] = 0  # ash
                    newly_burned += 1
                else:
                    new_grid[i] = 2  # keep burning
            elif i < len(ignite_flags) and ignite_flags[i] and new_grid[i] == 1:
                new_grid[i] = 2
                st.burn_timers[i] = 0
            elif cell == 4:
                # Water stays damp for several ticks before reverting to fuel
                st.burn_timers[i] += 1
                if st.burn_timers[i] >= 6:  # was 3; extend to make water useful
                    new_grid[i] = 1

        st.grid = new_grid
        return newly_burned

    def _burning_count(self) -> int:
        return sum(1 for v in self._state.grid if v == 2)

    def _saved_cells(self) -> int:
        # cells not turned to ash (includes fuel, burning, break, water)
        return sum(1 for v in self._state.grid if v in (1, 2, 3, 4))

    def _is_done(self) -> bool:
        return self._burning_count() == 0 or self._state.step_count >= self.max_steps

    def _make_observation(self, reward_hint: float = 0.0) -> WildfireObservation:
        st = self._state
        burning = self._burning_count()
        burned = sum(1 for v in st.grid if v == 0)
        # Use model_construct to bypass Pydantic validation for dataclass/Pydantic compatibility
        return WildfireObservation.model_construct(
            grid=st.grid[:],
            width=self.w,
            height=self.h,
            step=st.step_count,
            wind_dir=st.wind_dir,
            humidity=st.humidity,
            burning_count=burning,
            remaining_water=st.remaining_water,  # ✅ new
            remaining_breaks=st.remaining_breaks,  # ✅ new
            burned_count=burned,
            reward_hint=reward_hint,
        )

    # --- Required abstract property implementation ---
    @property
    def state(self) -> WildfireState:
        """Return the current environment state."""
        if self._state is None:
            # Initialize with minimal defaults if accessed before reset()
            # Use model_construct to bypass Pydantic validation for dataclass/Pydantic compatibility
            self._state = WildfireState.model_construct(
                episode_id="",
                step_count=0,
                total_burned=0,
                total_extinguished=0,
                last_action="reset",
                width=0,
                height=0,
                wind_dir="CALM",
                humidity=0.25,
                remaining_water=self.init_water,
                remaining_breaks=self.init_breaks,
                grid=[],
                burn_timers=[],
            )
        return self._state
