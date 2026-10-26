from typing import List, Optional

from openenv.core.env_server.interfaces import Action, Observation, State
from pydantic import Field

# Grid cell encoding:
# 0 = empty/ash, 1 = fuel (healthy), 2 = burning, 3 = firebreak, 4 = watered (damp)
# (You can tweak encodings, but keep them ints for compact obs.)


class WildfireAction(Action):
    """Action for wildfire environment.

    action: "break" (build firebreak), "water" (drop water), "wait"
    x, y: Optional coordinates for water/break actions
    """

    action: str
    x: Optional[int] = None
    y: Optional[int] = None


class WildfireObservation(Observation):
    """Observation from wildfire environment."""

    grid: List[int]  # flattened grid H*W, ints in {0..4}
    width: int
    height: int
    step: int
    wind_dir: str  # e.g. "N","NE","E","SE","S","SW","W","NW","CALM"
    humidity: float  # [0,1]
    burning_count: int
    burned_count: int  # total ash (0) cells (cumulative)
    reward_hint: float = 0.0  # optional shaping info
    remaining_water: int = 0
    remaining_breaks: int = 0


class WildfireState(State):
    """State for wildfire environment."""

    episode_id: str = ""
    step_count: int = 0
    total_burned: int = 0
    total_extinguished: int = 0
    last_action: str = "reset"
    # For visibility / debugging (not required by core):
    width: int = 0
    height: int = 0
    wind_dir: str = "CALM"
    humidity: float = 0.25
    remaining_water: int = 20  # simple resource constraint
    remaining_breaks: int = 50
    # internal full grid as flattened ints
    grid: List[int] = Field(default_factory=list)
    # burn timers for each cell (track how long cells have been burning/damp)
    burn_timers: List[int] = Field(default_factory=list)
