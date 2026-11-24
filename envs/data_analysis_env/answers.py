# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Read the answer out of a model's final reply, for harnesses that accept a text answer.

The environment grades whatever is passed to `submit_answer`. A harness that also ends an episode
on a reply without a tool call (as many chat loops do) can read the answer from that reply with
`extract_answer` and submit it, so text answers and `submit_answer` are graded the same way.
Pure string handling: model text is never executed.
"""

from __future__ import annotations

import re
from typing import Optional

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.S | re.I)
_TOOL_CALL_BLOCK = re.compile(r"<tool_call>.*?(?:</tool_call>|$)", re.S | re.I)
_ANSWER_LINE = re.compile(
    r"(?:^[ \t>#*_-]*(?:final[ \t]+)?answer|\bfinal[ \t]+answer)[ \t*_]*(?:is)?[ \t*_]*[:\uff1a][ \t]*(.*)$",
    re.I | re.M,
)
_TEXT_WRAPPER = re.compile(r"\\(?:text|textbf|mathrm|mathbf)\{([^{}]*)\}")


def _boxed(text: str) -> Optional[str]:
    """Content of the last \\boxed{...}, braces balanced, or None if there is none or it never closes."""
    start = text.rfind("\\boxed{")
    if start < 0:
        return None
    i = start + len("\\boxed{")
    depth, out = 1, []
    while i < len(text):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return "".join(out)
        out.append(ch)
        i += 1
    return None


def _clean(value: str) -> str:
    value = _TEXT_WRAPPER.sub(r"\1", value).strip()
    # Markdown and LaTeX wrapping around a short answer: **42**, `42`, $42$, "yes".
    for _ in range(3):
        stripped = value.strip().strip("*_`").strip()
        if len(stripped) >= 2 and stripped[0] == stripped[-1] and stripped[0] in "$\"'":
            stripped = stripped[1:-1].strip()
        if stripped == value:
            break
        value = stripped
    if value.endswith(".") and not value.endswith(".."):
        value = value[:-1].rstrip()
    return value


def extract_answer(text: Optional[str]) -> Optional[str]:
    """
    The answer in a final assistant message, or None if it holds none.

    In order: the last `\\boxed{...}`, else what follows the last "Answer:" at the start of a line
    or "Final answer:" anywhere (or the next non-empty line when the answer sits below it), else
    the whole message. Reasoning in `<think>...</think>` and malformed `<tool_call>` blocks are
    dropped first.
    """
    if not text:
        return None
    text = str(text)
    if "</think>" in text.lower():
        text = re.split(r"</think>", text, flags=re.I)[-1]
    text = _THINK_BLOCK.sub("", text)
    if re.search(
        r"<think>", text, re.I
    ):  # reasoning that never closed: no answer in it
        text = re.split(r"<think>", text, flags=re.I)[0]
    text = _TOOL_CALL_BLOCK.sub("", text).strip()
    if not text:
        return None
    boxed = _boxed(text)
    if boxed is not None and _clean(boxed):
        return _clean(boxed)
    matches = list(_ANSWER_LINE.finditer(text))
    if matches:
        last = matches[-1]
        value = _clean(last.group(1))
        if not value:
            rest = [line for line in text[last.end() :].splitlines() if line.strip()]
            value = _clean(rest[0]) if rest else ""
        if value:
            return value
    return _clean(text) or None
