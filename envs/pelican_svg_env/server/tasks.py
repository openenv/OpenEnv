# SPDX-License-Identifier: BSD-3-Clause

"""The task catalogue: which animal, on which vehicle.

Where these came from, because it matters:

- **`pelican_bicycle` is the real one.** "Generate an SVG of a pelican riding a
  bicycle" is Simon Willison's, and he has run it against nearly every model
  release since early 2025. It is the default task here and the only one with a
  public body of results behind it.
- **Every other combination is ours**, invented for this environment. They exist
  to check that the scorer measures drawing ability rather than one memorised
  picture: if a change to the geometry layer only works on pelicans and bicycles,
  these catch it.

What they are *not* is a contamination measurement. Dylan Castillo already ran
that study properly, over an 8 by 6 grid and 1008 SVGs from 7 frontier models,
and found little evidence that labs optimise for the pelican
(https://dylancastillo.co/posts/pelicanmaxxing.html). `held_out` here marks
"not the famous prompt", nothing stronger.

Simon's own conclusion is worth keeping in view when reading any score from this
environment: "the correlation between pelican performance and actual model
quality has been mostly severed now", and "don't go using pelicans to compare
models" (https://simonwillison.net/2026/Jul/16/kimi-k3/). This environment is
built to be a reproducible, trainable target, not a model ranking.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class Subject:
    """An animal that can be asked for.

    Attributes:
        name (`str`):
            Canonical noun used in the prompt.
        synonyms (`tuple[str, ...]`):
            Words that count as naming this animal exactly.
        family (`tuple[str, ...]`):
            Broader words that count as partial recognition, for example
            `bird` for a pelican.
        features (`tuple[str, ...]`):
            Distinguishing features a competent drawing should show. These are
            put to the judge as individual questions.
    """

    name: str
    synonyms: tuple[str, ...]
    family: tuple[str, ...]
    features: tuple[str, ...]


@dataclass(frozen=True)
class Vehicle:
    """A vehicle that can be asked for.

    Attributes:
        name (`str`):
            Canonical noun used in the prompt.
        synonyms (`tuple[str, ...]`):
            Words that count as naming this vehicle.
        wheels (`int`):
            Wheels visible in a side view, which is what the structural layer
            counts.
        features (`tuple[str, ...]`):
            Distinguishing features a competent drawing should show.
    """

    name: str
    synonyms: tuple[str, ...]
    wheels: int
    features: tuple[str, ...]


SUBJECTS: dict[str, Subject] = {
    "pelican": Subject(
        "pelican",
        ("pelican",),
        ("bird", "seabird", "waterbird"),
        ("a long beak", "a throat pouch under the beak"),
    ),
    "flamingo": Subject(
        "flamingo",
        ("flamingo",),
        ("bird", "waterbird"),
        ("a long thin neck", "long thin legs"),
    ),
    "capybara": Subject(
        "capybara",
        ("capybara",),
        ("rodent", "animal"),
        ("a blunt rectangular snout", "small rounded ears"),
    ),
    "axolotl": Subject(
        "axolotl",
        ("axolotl",),
        ("salamander", "amphibian", "animal"),
        ("feathery external gills on the head", "a wide flat smiling mouth"),
    ),
    "octopus": Subject(
        "octopus",
        ("octopus",),
        ("cephalopod", "animal"),
        ("a large bulbous head", "multiple curling tentacles"),
    ),
    "hedgehog": Subject(
        "hedgehog",
        ("hedgehog",),
        ("animal",),
        ("a coat of spines", "a small pointed snout"),
    ),
}

VEHICLES: dict[str, Vehicle] = {
    "bicycle": Vehicle(
        "bicycle",
        ("bicycle", "bike", "pushbike"),
        2,
        ("two wheels of similar size", "a frame joining the wheels", "handlebars"),
    ),
    "unicycle": Vehicle(
        "unicycle",
        ("unicycle",),
        1,
        ("a single wheel", "a seat post rising from the wheel"),
    ),
    "tandem bicycle": Vehicle(
        "tandem bicycle",
        ("tandem", "tandem bicycle", "bicycle built for two"),
        2,
        ("two wheels", "an extended frame with two saddles"),
    ),
    "scooter": Vehicle(
        "scooter",
        ("scooter", "kick scooter"),
        2,
        ("two small wheels", "a deck with an upright steering column"),
    ),
    "skateboard": Vehicle(
        "skateboard",
        ("skateboard", "board"),
        2,
        ("a flat deck", "wheels under the deck"),
    ),
}

# Simon Willison's original prompt, and the default task. Everything else in
# the grid is a variation of ours.
CANONICAL_PAIR = ("pelican", "bicycle")


@dataclass(frozen=True)
class Task:
    """One drawing request.

    Attributes:
        task_id (`str`):
            Stable identifier, `"{subject}_{vehicle}"` with spaces collapsed.
        subject ([`Subject`]):
            The animal to draw.
        vehicle ([`Vehicle`]):
            The vehicle it should be riding.
        held_out (`bool`):
            `False` only for Simon Willison's original pelican-and-bicycle
            prompt. `True` marks one of our own variations, which means "not the
            famous prompt" and nothing stronger.
    """

    task_id: str
    subject: Subject
    vehicle: Vehicle
    held_out: bool

    @property
    def prompt(self) -> str:
        """`str`: The instruction shown to the model under test."""
        return (
            f"Generate an SVG of a {self.subject.name} riding a "
            f"{self.vehicle.name}.\n\n"
            "Reply with the SVG document only. Draw it using vector primitives "
            "such as path, circle, ellipse, rect, line and polygon. Do not embed "
            "raster images, and do not write the answer as text inside the "
            "drawing."
        )

    @property
    def forbidden_terms(self) -> list[str]:
        """`list[str]`: Words that must not be written into the drawing."""
        return sorted(
            {
                *self.subject.synonyms,
                *self.subject.family,
                *self.vehicle.synonyms,
                "riding",
            }
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary of the task."""
        return {
            "task_id": self.task_id,
            "subject": self.subject.name,
            "vehicle": self.vehicle.name,
            "expected_wheels": self.vehicle.wheels,
            "held_out": self.held_out,
            "prompt": self.prompt,
        }


def make_task(subject: str, vehicle: str) -> Task:
    """Build a task from a subject and vehicle name.

    Args:
        subject (`str`):
            Key into [`SUBJECTS`].
        vehicle (`str`):
            Key into [`VEHICLES`].

    Returns:
        [`Task`]: The assembled task.

    Raises:
        KeyError: If either name is not in the catalogue.

    Examples:

    ```python
    task = make_task("pelican", "bicycle")
    print(task.prompt)
    ```
    """
    if subject not in SUBJECTS:
        raise KeyError(f"unknown subject {subject!r}; choose from {sorted(SUBJECTS)}")
    if vehicle not in VEHICLES:
        raise KeyError(f"unknown vehicle {vehicle!r}; choose from {sorted(VEHICLES)}")
    slug = f"{subject}_{vehicle}".replace(" ", "-")
    return Task(
        task_id=slug,
        subject=SUBJECTS[subject],
        vehicle=VEHICLES[vehicle],
        held_out=(subject, vehicle) != CANONICAL_PAIR,
    )


def all_tasks(held_out_only: bool = False) -> list[Task]:
    """Return every subject-by-vehicle combination in the catalogue.

    Args:
        held_out_only (`bool`, *optional*, defaults to `False`):
            Exclude the canonical pelican-and-bicycle pair.

    Returns:
        `list[Task]`: The full grid, in a stable order.
    """
    tasks = [
        make_task(subject, vehicle) for subject in SUBJECTS for vehicle in VEHICLES
    ]
    return [t for t in tasks if t.held_out] if held_out_only else tasks


def sample_task(seed: int | None = None, held_out_only: bool = False) -> Task:
    """Draw one task from the grid.

    Args:
        seed (`int`, *optional*):
            Seed for reproducible selection. Without it the choice is random.
        held_out_only (`bool`, *optional*, defaults to `False`):
            Exclude the canonical pair.

    Returns:
        [`Task`]: The selected task.
    """
    pool = all_tasks(held_out_only=held_out_only)
    rng = random.Random(seed)
    return rng.choice(pool)


def task_from_ids(task_ids: Sequence[str]) -> list[Task]:
    """Resolve task identifiers back into tasks, preserving order."""
    index = {task.task_id: task for task in all_tasks()}
    unknown = [task_id for task_id in task_ids if task_id not in index]
    if unknown:
        raise ValueError(f"Unknown task ids {unknown}. Valid ids: {sorted(index)}")
    return [index[task_id] for task_id in task_ids]
