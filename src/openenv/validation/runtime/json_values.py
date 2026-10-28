"""Exact JSON values for byte-bounded runtime documents and redacted bundles."""

import json
from decimal import Decimal


def loads(text):
    # Match the stdlib's BOM handling for bytes after UTF-8 text retention.
    if isinstance(text, str):
        text = text.removeprefix("\ufeff")
    return json.loads(text, parse_float=Decimal, parse_constant=Decimal)


def typed_json(value):
    """Flatten values so traversal and equality do not recurse through containers."""
    result, pending = [], [value]
    while pending:
        child = pending.pop()
        if isinstance(child, dict):
            result.append((dict, len(child)))
            for key in sorted(child, reverse=True):
                pending.extend((child[key], key))
        elif isinstance(child, list):
            result.append((list, len(child)))
            pending.extend(reversed(child))
        else:
            if isinstance(child, Decimal) and not child.is_finite():
                raise ValueError("non-finite JSON constant")
            result.append((type(child), child))
    return tuple(result)


def dumps(value):
    """Serialize depth-bounded, redacted values without rounding Decimal leaves."""
    if isinstance(value, dict):
        return (
            "{"
            + ",".join(
                json.dumps(key) + ":" + dumps(value[key]) for key in sorted(value)
            )
            + "}"
        )
    if isinstance(value, list):
        return "[" + ",".join(dumps(child) for child in value) + "]"
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError("non-finite JSON constant")
        number = str(value)
        # Decimal('1') came from a decimal/exponent token, never an integer.
        return number if "." in number or "e" in number.lower() else number + ".0"
    return json.dumps(value, allow_nan=False)
