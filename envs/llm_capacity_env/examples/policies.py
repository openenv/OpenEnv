"""Causal baselines using only the history in the public tool observation."""

from math import ceil


def quantile(xs, q=0.75):
    values = sorted(xs)
    return values[max(0, ceil(q * len(values)) - 1)]


def reservations(view, policy="rolling_quantile", frozen=None):
    models = tuple(view["unit_cost_per_1000_tokens"])
    if policy == "zero":
        return {m: [0] * 24 for m in models}
    if policy == "fixed_quantile" and frozen is not None:
        return frozen
    if policy not in ("rolling_quantile", "fixed_quantile", "previous_day"):
        raise ValueError("Unknown policy")
    if policy == "previous_day":
        return {m: list(view["history"][-1]["tokens"][m]) for m in models}
    # Cost minimizer of empirical newsvendor objective with 4x overflow price: q=0.75.
    return {
        m: [quantile([day["tokens"][m][h] for day in view["history"]]) for h in range(24)]
        for m in models
    }
