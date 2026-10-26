"""OpenEnv models for browser actions, observations, and state."""

from typing import Literal

from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field, model_validator


class BrowserAction(Action):
    op: Literal["click", "type", "key", "scroll", "back", "wait", "finish"]
    x: int | None = Field(default=None, ge=0, lt=800)
    y: int | None = Field(default=None, ge=0, lt=600)
    text: str = Field(default="", max_length=2000)
    dy: int = Field(default=0, ge=-1200, le=1200)

    @model_validator(mode="after")
    def validate_arguments(self):
        if self.op == "click" and (self.x is None or self.y is None):
            raise ValueError("click needs x and y")
        if self.op == "type" and not self.text:
            raise ValueError("type needs text")
        if self.op == "key" and self.text not in (
            "ENTER",
            "TAB",
            "ESCAPE",
            "BACKSPACE",
            "CTRL+A",
            "UP",
            "DOWN",
            "LEFT",
            "RIGHT",
        ):
            raise ValueError("unsupported key")
        return self


class BrowserObservation(Observation):
    screenshot: str
    url: str
    error: str = ""
    controls: list[dict] = Field(default_factory=list)
    page_text: str = ""


class BrowserState(State):
    pass
