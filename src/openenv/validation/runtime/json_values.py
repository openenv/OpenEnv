"""Exact JSON values for byte-bounded runtime documents and redacted bundles."""

import json
from decimal import Decimal


def loads(text, *, object_pairs_hook=None):
    # Match the stdlib's BOM handling for bytes after UTF-8 text retention.
    if isinstance(text, str):
        text = text.removeprefix("\ufeff")
    return json.loads(
        text,
        parse_float=Decimal,
        parse_constant=Decimal,
        object_pairs_hook=object_pairs_hook,
    )


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


def dumps(value, *, ensure_ascii=True):
    """Serialize exact JSON numbers without recursing through containers."""
    chunks, pending = [], [(False, value)]
    while pending:
        literal, child = pending.pop()
        if literal:
            chunks.append(child)
        elif isinstance(child, dict):
            chunks.append("{")
            pending.append((True, "}"))
            for index, key in reversed(list(enumerate(sorted(child)))):
                pending.extend(
                    [
                        (False, child[key]),
                        (True, ":"),
                        (True, json.dumps(key, ensure_ascii=ensure_ascii)),
                    ]
                )
                if index:
                    pending.append((True, ","))
        elif isinstance(child, list):
            chunks.append("[")
            pending.append((True, "]"))
            for index, item in reversed(list(enumerate(child))):
                pending.append((False, item))
                if index:
                    pending.append((True, ","))
        elif isinstance(child, Decimal):
            if not child.is_finite():
                raise ValueError("non-finite JSON constant")
            number = str(child)
            # Decimal('1') came from a decimal/exponent token, never an integer.
            chunks.append(
                number if "." in number or "e" in number.lower() else number + ".0"
            )
        else:
            chunks.append(json.dumps(child, allow_nan=False, ensure_ascii=ensure_ascii))
    return "".join(chunks)
