# Validation evidence

CI_MATRIX.json records the Python 3.11/3.12 monorepo checks, source hashes,
retries and dependency setup on 2026-11-02. SUMMARY.json and SOURCE_MANIFEST.json
record the separate OpenEnv 0.11.0 / Python 3.13 standalone container-validation
snapshot; paths in that manifest refer to that standalone archive.

- 21 environment/MCP/trajectory tests passed; Ruff lint passed.
- Official static manifest validation: one registered static check passed.
- Official URL/API validation: 6/6 checks passed.
- Actual pinned-base Docker build and 14-decision network episode passed.
- Concurrent isolated sessions, invalid/terminal reward handling, seeded reset
  and the web UI passed.
- The actual API server emitted per-session JSONL files with reset, action,
  observation, reward, terminal flag and step count; a complete 14-decision
  trajectory was inspected. See SERVER_TRAJECTORIES.json.
- All three upstream raw releases reproduced the exact aggregate SHA-256.
- Wheel build, installed client/data and offline official push staging passed.

No upload or upstream acceptance is implied. These checks are not full semantic
or hub certification, and do not establish frontier-agent difficulty. Costs are
modeled accounting units; the perfect-information row reads future demand and
is only an offline lower bound. Public partitions are not sealed holdouts.
