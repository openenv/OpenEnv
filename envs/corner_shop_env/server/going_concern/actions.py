"""Parse and validate an agent action. Money arrives in dollars and becomes integer cents.

Schema (every key optional):
{
  "prices":    {"A": 4.99},                         # set shelf price per SKU
  "orders":    [{"supplier": "S1", "qty": 120}],    # purchase orders
  "pay":       [{"invoice": "INV-3", "amount": 250.0}],  # omit amount to pay in full
  "borrow":    500.0,                               # draw on the credit line; negative repays
  "repay":     200.0,                               # same as a negative borrow
  "hire": 1, "fire": 0,
  "liquidate": [{"sku": "A", "qty": 50}],           # fire-sale to a liquidator
  "accept":    ["M-12"],                            # accept an offer in the inbox
  "dispute":   ["INV-7"],                           # dispute an invoice
  "replies":   [{"to": "M-12", "text": "..."}]
}
Malformed parts are dropped and reported; they never crash the engine."""
from __future__ import annotations

import json
import math

MAX_QTY = 1_000_000
MAX_MONEY_CENTS = 10_000_000_000  # $100M
MAX_LIST = 50
MAX_TEXT = 2000


def _money(x, errors, where, allow_negative=False):
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
        errors.append(f"{where}: not a number")
        return None
    if abs(x) > MAX_MONEY_CENTS / 100:  # check before scaling: 1e307 * 100 overflows
        errors.append(f"{where}: amount too large")
        return None
    c = int(round(x * 100))
    if abs(c) > MAX_MONEY_CENTS:
        errors.append(f"{where}: amount too large")
        return None
    if c < 0 and not allow_negative:
        errors.append(f"{where}: negative amount")
        return None
    return c


def _qty(x, errors, where):
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
        errors.append(f"{where}: not a number")
        return None
    if abs(x) > MAX_QTY:
        errors.append(f"{where}: quantity too large")
        return None
    if x != int(x):
        errors.append(f"{where}: quantity must be a whole number")
        return None
    q = int(x)
    if q < 0:
        errors.append(f"{where}: negative quantity")
        return None
    if q > MAX_QTY:
        errors.append(f"{where}: quantity too large")
        return None
    return q


def _list(v, errors, where):
    if not isinstance(v, list):
        errors.append(f"{where}: expected a list")
        return []
    if len(v) > MAX_LIST:
        errors.append(f"{where}: list truncated to {MAX_LIST}")
    return v[:MAX_LIST]


def parse_action(raw) -> dict:
    errors: list[str] = []
    if raw is None:
        raw = {}
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as e:
            return {"_errors": [f"invalid JSON: {e.msg}"]}
    if not isinstance(raw, dict):
        return {"_errors": ["action must be a JSON object"]}
    known = {"prices", "orders", "pay", "borrow", "repay", "hire", "fire", "liquidate", "accept",
             "dispute", "replies"}
    for k in raw:
        if k not in known:
            errors.append(f"unknown field {k!r} ignored")
    act: dict = {}

    if "prices" in raw:
        if isinstance(raw["prices"], dict):
            act["prices"] = {}
            for sku, p in list(raw["prices"].items())[:MAX_LIST]:
                c = _money(p, errors, f"price {sku}")
                if c is not None:
                    if c < 1:
                        errors.append(f"price {sku}: must be at least $0.01")
                    else:
                        act["prices"][str(sku)] = c
        else:
            errors.append("prices: expected an object")

    act["orders"] = []
    for o in _list(raw.get("orders", []), errors, "orders"):
        if not isinstance(o, dict) or "supplier" not in o or "qty" not in o:
            errors.append("order: needs supplier and qty")
            continue
        q = _qty(o["qty"], errors, f"order {o.get('supplier')}")
        if q:
            act["orders"].append({"supplier": str(o["supplier"]), "qty": q})

    act["pay"] = []
    for p in _list(raw.get("pay", []), errors, "pay"):
        if isinstance(p, str):
            p = {"invoice": p}
        if not isinstance(p, dict) or "invoice" not in p:
            errors.append("pay: needs invoice")
            continue
        amt = None
        if p.get("amount") is not None:
            amt = _money(p["amount"], errors, f"pay {p['invoice']}")
            if amt is None:
                continue
        act["pay"].append({"invoice": str(p["invoice"]), "amount": amt})

    borrow = 0
    if "borrow" in raw:
        c = _money(raw["borrow"], errors, "borrow", allow_negative=True)
        borrow += c or 0
    if "repay" in raw:
        c = _money(raw["repay"], errors, "repay")
        borrow -= c or 0
    act["borrow"] = borrow

    for k in ("hire", "fire"):
        if k in raw:
            q = _qty(raw[k], errors, k)
            act[k] = q or 0

    act["liquidate"] = []
    for lq in _list(raw.get("liquidate", []), errors, "liquidate"):
        if not isinstance(lq, dict) or "sku" not in lq or "qty" not in lq:
            errors.append("liquidate: needs sku and qty")
            continue
        q = _qty(lq["qty"], errors, f"liquidate {lq.get('sku')}")
        if q:
            act["liquidate"].append({"sku": str(lq["sku"]), "qty": q})

    for k in ("accept", "dispute"):
        act[k] = [str(x) for x in _list(raw.get(k, []), errors, k) if isinstance(x, (str, int))]

    act["replies"] = []
    for r in _list(raw.get("replies", []), errors, "replies"):
        if isinstance(r, dict) and "to" in r and isinstance(r.get("text"), str):
            act["replies"].append({"to": str(r["to"]), "text": r["text"][:MAX_TEXT]})
        else:
            errors.append("reply: needs to and text")

    act["_errors"] = errors
    return act
