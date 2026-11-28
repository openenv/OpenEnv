"""Prompt text and reply parsing shared by every LLM-facing wrapper (verifiers env, API baselines)."""
from __future__ import annotations

import json

SYSTEM_PROMPT = """You run a small retail business for a few months. Each turn is one week.
You see the firm's state and inbox, and reply with ONE JSON object: your decisions for this week.
Think briefly if you like, then put the JSON action last in your reply, inside a ```json block.

Action fields (all optional; money in dollars):
{
  "prices":    {"A": 4.99},                          // shelf price per product
  "orders":    [{"supplier": "S1", "qty": 120}],     // purchase orders (respect min order)
  "pay":       [{"invoice": "INV-3"}],               // add "amount" for a partial payment
  "borrow":    500.0,                                // draw on the credit line (negative = repay)
  "hire": 1, "fire": 0,                              // staff (hires join next week)
  "liquidate": [{"sku": "A", "qty": 50}],            // fire-sale stock to a liquidator
  "accept":    ["M-12"],                             // accept an offer in your inbox
  "dispute":   ["INV-7"],                            // dispute an invoice you believe is wrong
  "replies":   [{"to": "M-12", "text": "..."}]       // free-text reply to a message
}
Customers buy less when prices are high or shelves are empty, and they remember both.
Suppliers charge late fees and stop extending credit to firms that pay late. Breaking the bank's
covenant gets the line frozen and called. Payroll, rent and interest are paid automatically;
if they can't be paid the firm goes bankrupt. Your goal: leave the business in the best long-run
shape you can: a healthy going concern, not just a big cash balance on the last day."""


def first_prompt(obs_text: str) -> str:
    return obs_text + "\n\nWhat are your decisions for this week? End with the JSON action."


def next_prompt(obs_text: str) -> str:
    return obs_text + "\n\nYour decisions for this week? End with the JSON action."


def extract_action(reply: str):
    """Return the last top-level JSON object in the reply (preferring ```json blocks), or the raw
    text if none parses (the engine then reports the error back to the agent)."""
    if not isinstance(reply, str):
        return reply
    text = reply
    if "```" in text:
        blocks = text.split("```")
        for b in reversed(blocks[1::2]):
            body = b[4:] if b.lstrip().lower().startswith("json") else b
            body = body.strip()
            if body.lower().startswith("json"):
                body = body[4:].strip()
            try:
                v = json.loads(body)
                if isinstance(v, dict):
                    return v
            except json.JSONDecodeError:
                pass
    # scan for balanced {...} from the end
    end = text.rfind("}")
    while end != -1:
        depth = 0
        for i in range(end, -1, -1):
            c = text[i]
            if c == "}":
                depth += 1
            elif c == "{":
                depth -= 1
                if depth == 0:
                    try:
                        v = json.loads(text[i:end + 1])
                        if isinstance(v, dict):
                            return v
                    except json.JSONDecodeError:
                        pass
                    break
        end = text.rfind("}", 0, end)
    return text
