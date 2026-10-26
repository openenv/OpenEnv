# Reference Echo environment canary

The lab copies `envs/echo_env` from the tested checkout byte for byte and records
its source hashes. This directory supplies only the runtime execution declaration
and public action plan. It does not contain a substitute Echo implementation.

The canary reuses the shared probe's digest-pinned base, exact OpenEnv wheel,
hash-checked wheelhouse and offline installation. Its image recipe changes only
the subject COPY and entrypoint. The validator never imports Echo on the host.

The acceptance test requires real tool responses and a continuing episode with
state counts 0, 1 and 2. Echo explicitly advertises its base reset observation
separately from CallToolObservation step responses. Its null rewards denote
unscored, nonterminal tool calls permitted by the core API but rejected by Level
Two's stricter numeric-step-reward requirement. Startup, state and observation
schema checks must pass; reward must FAIL, making the report FAIL with CLI exit 1.
This is an expected certification finding, not a broken canary or a reason to
coerce null to zero. Malformed reset observations and legacy single-schema
mismatches remain failures in dedicated regressions.
