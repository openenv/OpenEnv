# Example baseline trajectories

Two completed local MCP episodes, seed 23, test split: zero reservation and
rolling empirical 0.75 quantile. Every JSONL contains reset metadata and all 14
observation/action/reward/terminal/next-observation transitions. Both were
independently replayed through the environment with exact equality.

These are deterministic heuristic-policy runs, not LLM agent or reasoning traces.
They do not certify automatic server-side session telemetry or all RFC 008 checks.
They were run on the center cluster, not on the local controller.

Reproduce from the environment root:

```bash
python examples/record_trace.py outputs/traces
```

The seed was reused from the protocol smoke test. This single episode is only
an instrumentation/replay example, not a representative difficulty estimate.
For aggregate baselines, see validation-evidence/BASELINES.json.
