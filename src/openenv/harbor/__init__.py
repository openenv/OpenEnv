"""Harbor integration: run Harbor tasks as OpenEnv environments, for eval and training.

Harbor owns what it is good at — task datasets, sandbox backends, coding agents, verifiers, trial
concurrency, pass@k. This package adds the OpenEnv side and nothing more:

    tasks.py          dataset discovery over the Task API (HF repo | local dir | Harbor registry)
    seams.py          how each agent is pointed at the capture proxy — the only per-agent knowledge
    install_fixes.py  subclasses for agents whose Harbor wrapper cannot be configured as shipped
    atif.py           cross-check captured tokens against Harbor's own ATIF trajectory
    models.py         wire types

Generic capture lives in `openenv.core.harness.capture` and knows nothing about Harbor. The
dependency runs one way only: `openenv.harbor` imports capture, never the reverse. ATIF is here
rather than there because it is Harbor's trace format, not a general one.

Harbor itself is an optional dependency (`pip install openenv[harbor]`), imported lazily so that
importing `openenv` never requires it.
"""
