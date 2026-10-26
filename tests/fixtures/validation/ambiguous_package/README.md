A package that carries two well-known files (`openenv.yaml` and `task.toml`).
`openenv validate` must refuse it as ambiguous (exit 2, SignatureError) — never
guess which format to parse.
