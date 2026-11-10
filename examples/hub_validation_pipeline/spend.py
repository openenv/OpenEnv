"""Offline shared spend ledger; amounts are decimal strings, Decimal, or integers."""

import argparse
import fcntl
import json
import os
import re
import tempfile
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
from decimal import Decimal
from fractions import Fraction
from pathlib import Path


def _amount(value):
    if type(value) not in (str, Decimal, int) or not re.fullmatch(
        r"\+?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?", str(value)
    ):
        raise ValueError("Amount must be a nonnegative finite decimal")
    return Fraction(Decimal(value))


def _text(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("ID, purpose and evidence must be nonempty strings")
    return value


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError("Duplicate ledger key")
        result[key] = value
    return result


def _report(data):
    if not isinstance(data, dict) or data.get("schema") != "openenv-shared-spend-v1":
        raise ValueError("Invalid ledger schema")
    limit, reserved, spent = _amount(data["limit_usd"]), Fraction(0), Fraction(0)
    changes = data.get("limit_changes", [])
    if not isinstance(changes, list):
        raise ValueError("Invalid limit change history")
    previous = None
    for change in changes:
        if (
            not isinstance(change, dict)
            or not {"previous_limit_usd", "limit_usd", "evidence", "changed_utc"}
            <= change.keys()
        ):
            raise ValueError("Invalid limit change history")
        old, new = _amount(change["previous_limit_usd"]), _amount(change["limit_usd"])
        _text(change["evidence"])
        changed = datetime.fromisoformat(_text(change["changed_utc"]))
        if (
            changed.utcoffset() != timezone.utc.utcoffset(None)
            or old == new
            or (previous is not None and previous != old)
        ):
            raise ValueError("Invalid limit change history")
        previous = new
    if changes and previous != limit:
        raise ValueError("Limit change history differs from current limit")
    over = data.get("over_budget", False)
    statuses = {}
    for identifier, row in data["reservations"].items():
        amount = _amount(row["reserved_usd"])
        statuses[identifier] = row["status"]
        if row["status"] == "reserved" and row["spent_usd"] is None:
            reserved += amount
        elif row["status"] == "settled":
            _text(row.get("evidence"))
            actual = _amount(row["spent_usd"])
            spent += actual
            over = over or actual > amount
        else:
            raise ValueError("Invalid reservation status")
    totals = dict(limit=limit, reserved=reserved, spent=spent, charged=reserved + spent)
    totals["remaining"] = limit - totals["charged"]
    result = {
        "over_budget": bool(over or totals["charged"] > limit),
        "statuses": statuses,
    }
    for key, value in totals.items():
        micros = -((-value.numerator * 1000000) // value.denominator)
        sign, whole = ("-" if micros < 0 else ""), abs(micros)
        result[f"{key}_usd_exact"] = str(value)
        result[f"{key}_usd_rounded_up"] = (
            f"{sign}{whole // 1000000}.{whole % 1000000:06d}"
        )
    return result


@contextmanager
def _ledger(path, limit=None):
    path = Path(path).resolve()
    with path.with_name(path.name + ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        source = None
        if path.exists():
            source = path.read_bytes().decode("utf-8")
            data = json.loads(source, object_pairs_hook=_pairs)
        elif limit is not None:
            data = dict(
                schema="openenv-shared-spend-v1",
                limit_usd=str(limit),
                reservations={},
                created_utc=datetime.now(timezone.utc).isoformat(),
            )
        else:
            raise ValueError("Ledger must be initialized first")
        _report(data)
        before = deepcopy(data)
        yield data
        if data != before or not path.exists():
            data["updated_utc"] = datetime.now(timezone.utc).isoformat()
            with tempfile.NamedTemporaryFile(
                mode="w", dir=path.parent, delete=False
            ) as out:
                temporary = Path(out.name)
                try:
                    rendered = json.dumps(data, indent=2, allow_nan=False)
                    if (
                        source is not None
                        and data["reservations"] == before["reservations"]
                    ):
                        # Decode top-level values to retain the original reservation bytes.
                        decoder = json.JSONDecoder()
                        index = source.index("{") + 1
                        for _ in before:
                            index += re.match(r"\s*", source[index:]).end()
                            key, index = decoder.raw_decode(source, index)
                            index = source.index(":", index) + 1
                            index += re.match(r"\s*", source[index:]).end()
                            _, end = decoder.raw_decode(source, index)
                            if key == "reservations":
                                field = json.dumps(
                                    {"reservations": data["reservations"]}, indent=2
                                )[1:-2]
                                rendered = rendered.replace(
                                    field, '\n  "reservations": ' + source[index:end], 1
                                )
                                break
                            index = source.index(",", end) + 1
                    out.write(rendered)
                    out.write("\n")
                    out.flush()
                    os.fsync(out.fileno())
                    os.replace(temporary, path)
                finally:
                    temporary.unlink(missing_ok=True)
        # A prior replace may have succeeded before directory syncing failed.
        # Idempotent retries must confirm that durability before acknowledging.
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)


def init(path, limit_usd):
    """Initialize once; an existing ledger must have the same limit."""
    limit = _amount(limit_usd)
    with _ledger(path, str(limit_usd)) as data:
        if _amount(data["limit_usd"]) != limit:
            raise ValueError("Existing ledger limit differs")
        return _report(data)


def set_limit(path, limit_usd, evidence):
    """Record an evidenced limit decision, retaining all holds and overrun blocks."""
    limit = _amount(limit_usd)
    _text(evidence)
    with _ledger(path) as data:
        report = _report(data)
        if _amount(data["limit_usd"]) == limit:
            return report
        data.setdefault("limit_changes", []).append(
            dict(
                previous_limit_usd=data["limit_usd"],
                limit_usd=str(limit_usd),
                evidence=evidence,
                changed_utc=datetime.now(timezone.utc).isoformat(),
            )
        )
        data["limit_usd"] = str(limit_usd)
        if report["over_budget"] or _report(data)["over_budget"]:
            data["over_budget"] = True
        return _report(data)


def reserve(path, id, amount_usd, purpose, evidence=None):
    """Reserve before paid work; settled IDs cannot be reused."""
    amount = _amount(amount_usd)
    _text(id), _text(purpose)
    if evidence is not None:
        _text(evidence)
    with _ledger(path) as data:
        report = _report(data)
        rows = data["reservations"]
        if id in rows:
            if (
                rows[id]["status"] == "reserved"
                and _amount(rows[id]["reserved_usd"]) == amount
            ):
                return report
            raise ValueError("Reservation ID already used; use a fresh ID")
        if report["over_budget"] or amount > Fraction(report["remaining_usd_exact"]):
            raise ValueError("Shared budget exceeded or blocked")
        rows[id] = dict(
            reserved_usd=str(amount_usd),
            spent_usd=None,
            status="reserved",
            purpose=purpose,
            evidence=evidence,
        )
        return _report(data)


def settle(path, id, spent_usd, evidence):
    """Record actual cost, freezing new reservations on any reservation/cap overrun."""
    actual = _amount(spent_usd)
    _text(id), _text(evidence)
    with _ledger(path) as data:
        rows = data["reservations"]
        if id not in rows:
            raise ValueError("Unknown reservation ID")
        row = rows[id]
        if row["status"] == "settled":
            if _amount(row["spent_usd"]) == actual:
                return _report(data)
            raise ValueError("Different settlement requires manual reconciliation")
        row.update(
            status="settled",
            spent_usd=str(spent_usd),
            evidence=evidence,
            settled_utc=datetime.now(timezone.utc).isoformat(),
        )
        report = _report(data)
        data["over_budget"] = report["over_budget"]
        return report


def summary(path):
    """Return exact fraction totals, upward-rounded six-place displays and ID statuses."""
    with _ledger(path) as data:
        return _report(data)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("summary")
    sub = commands.add_parser("set-limit")
    sub.add_argument("--usd", required=True)
    sub.add_argument("--evidence", required=True)
    for command in ("reserve", "settle"):
        sub = commands.add_parser(command)
        sub.add_argument("--id", required=True)
        sub.add_argument("--usd", required=True)
        sub.add_argument("--evidence", required=command == "settle")
        if command == "reserve":
            sub.add_argument("--purpose", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "set-limit":
            result = set_limit(args.ledger, args.usd, args.evidence)
        elif args.command == "reserve":
            result = reserve(
                args.ledger, args.id, args.usd, args.purpose, args.evidence
            )
        elif args.command == "settle":
            result = settle(args.ledger, args.id, args.usd, args.evidence)
        else:
            result = summary(args.ledger)
        print(json.dumps(result, sort_keys=True))
    except (ValueError, TypeError, KeyError, AttributeError, OSError):
        parser.exit(
            2, "Spend operation rejected; inspect ledger and arguments locally.\n"
        )


if __name__ == "__main__":
    main()
