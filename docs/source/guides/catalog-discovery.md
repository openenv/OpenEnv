# Catalog Discovery

`openenv catalog` and `openenv discover` find an environment for a task from committed metadata only. They never install, import, pull or run candidate environments. To load an environment you already know, use [`AutoEnv`](auto-discovery) instead.

## Build a catalog

From a clone of the OpenEnv repository:

```bash
openenv catalog build \
  --repository . \
  --repository-uri https://github.com/huggingface/OpenEnv.git \
  --revision HEAD \
  --publisher example.org \
  --output catalog.json
```

- The catalog lists every tracked `openenv.yaml` in a direct child of `envs/` (change it with `--root`) at the resolved commit. Untracked files are ignored.
- `--publisher` names the publication authority. `example.org` is a placeholder. Setting it does not verify identity.
- Each card reads the environment's `openenv.yaml`, `pyproject.toml`, README front matter and optional `discovery.json`, and records the source URI, path and full revision.
- If any environment has invalid metadata, the command exits nonzero and the report has `complete: false`. An incomplete catalog can't be loaded for search.

## Search and inspect

```bash
openenv discover "client smoke test" --catalog catalog.json
openenv discover "client smoke test" --catalog catalog.json --json
openenv discover "" --catalog catalog.json --filter license=BSD-3-Clause
openenv catalog inspect "<identifier from the result>" --catalog catalog.json
```

Ranking is case-insensitive token overlap with descriptions, names, tags, capabilities and representative queries. Scores measure that lexical match only. `--filter` takes exact `field=value` matches on `license`, `provider`, `artifact_availability`, `name`, `tags` or `type`, and `--limit` caps the results (default 20).

`catalog inspect` returns the complete entry for an exact identifier from the snapshot. It never fetches URLs or visits deployments.

## Add a `discovery.json`

An optional `discovery.json` next to an environment's `openenv.yaml` adds metadata the other files lack. It accepts `description`, `tags`, `representative_queries` (empty, or two to five), `artifact_availability`, `license`, `license_source` and `agent_tools`. Echo's declaration:

```json
{
  "tags": ["openenv", "smoke-test"],
  "representative_queries": [
    "find an environment that echoes a message",
    "find a minimal environment for checking OpenEnv tool calls",
    "find an environment for a client smoke test"
  ],
  "agent_tools": {
    "protocol": "mcp",
    "names": ["echo_message", "echo_with_length"],
    "source": "envs/echo_env/server/echo_environment.py"
  }
}
```

`agent_tools.source` points to the file that defines the tools. Don't list simulation controls (such as `reset` or `step`) as agent tools. The build rejects known control names.

`discovery.json` takes precedence over `openenv.yaml`, the package description and README front matter. Conflicting license declarations become `unknown` with a warning.

Coding, BrowserGym, Calendar, Chess and Reasoning Gym also ship a `discovery.json`. For example, try `openenv discover "Python snippets and standard error" --catalog catalog.json`.

## The `0.1-draft` profile

This is the contract `openenv catalog` and `openenv discover` enforce, the first repository profile of
[RFC 011](https://github.com/huggingface/OpenEnv/blob/main/rfcs/011-ard-catalog-discovery.md).

### Inventory and producer

The inventory is exactly the regular tracked `openenv.yaml` files in direct
child directories of `envs/` (or `--root`) at the resolved Git commit. It
excludes untracked files, submodules, deployments, and community repositories
outside that tree. It is not a global environment census.

The producer reads the commit's manifest, project metadata, README frontmatter
and optional `discovery.json`. Frontmatter accepts LF, CRLF and CR line endings.
An opening frontmatter marker without a closing marker is invalid metadata. Each
card and its Git artifact keep a source URI, environment path and full revision.
The snapshot has a separate content digest. Builds from identical committed
inputs and publisher settings are byte-identical.

An unreadable or invalid eligible environment produces an error in the build
report and `complete: false`, and the command exits nonzero. The partial report
is inspectable but cannot be loaded as a complete catalog or used to infer
withdrawals. Unknown license or omitted tool evidence remains unknown.

### Search semantics

The baseline ranks literal, case-insensitive query-token overlap over authored
descriptions, names, tags, capabilities and representative queries. Scores
describe this lexical match only, not training quality, validation, reputation
or execution approval. Unsupported filters are errors rather than empty
success. An identifier is resolved only in the configured snapshot, missing
identifiers fail explicitly, and no URL is guessed from an identifier. Neither
the CLI nor the library fetches metadata URLs, visits deployments or passes
source IDs into a Hub resolver.

### Metadata precedence and license rules

Description precedence is `discovery.json`, `openenv.yaml`, package description,
then README frontmatter. An explicit reviewed license declaration takes
precedence. Conflicting package and README declarations remain `unknown` with a
diagnostic. Otherwise the package or README declaration is used, then the
repository's declared source license. Mappings are retained in
`metadata.provenance`. This is a source declaration, not a legal certification.

A package `license = {file = "LICENSE"}` table remains `unknown`: a file pointer
alone does not identify an SPDX or custom license, and the producer does not
classify the referenced file or replace the unknown with a repository-wide
license. A license table cannot contain both `file` and `text`.

Custom license text is retained internally when comparing package and README
declarations, and the emitted card uses `other`. Two different custom texts do
not match merely because both normalize to that category. Identical custom
text, allowing for line endings and outer whitespace, is a consistent
declaration. Two bare `other` markers do not establish a shared license
identity. Ambiguous or conflicting declarations produce `unknown` and a
`license_conflict` warning.

Tool declarations name repository-relative evidence within the environment and
remain `declared`. Listing a tool name does not establish semantic safety.
Simulation controls (`reset`, `step`, `state`, `get_state`, in any case) are
rejected as agent tools, but name checks do not prove the boundary. Discovery
never invokes a tool to inspect it.

`manifest_spec_version` is a manifest marker. `framework_requirement` is a
source-declared package requirement. Neither is a runtime-protocol version or a
compatibility certificate.

### Profile and schemas

The `0.1-draft` profile is declaration-only, GitHub-source, revision-bound, and
inline. Unsupported profile versions and validated-interface claims are
rejected. It does not replace RFC 008's normalized validation manifest, report
contract, or graders, and does not require the validation block.

Schemas are packaged in `openenv/discovery/schemas/0.1-draft/`. Regenerate them
with `PYTHONPATH=src python scripts/generate_discovery_schemas.py` (`--check`
detects drift). The Pydantic models also enforce relational invariants such as
matching artifact revisions and inventory accounting.

The resource media type is `application/vnd.openenv.environment-card+json`. It
describes an environment source definition, not an installable MCP-server
configuration. Clients dispatch on the ARD entry `type` and check the card's
`data.schema_version`. The ARD entry carries search-facing fields
(`displayName`, `description`, `tags`, `capabilities`,
`representativeQueries`), and its inline `data` is the Environment Card.
Preserve both rather than reconstructing the card from search snippets.

| Packaged schema | Applies to |
|-----------------|------------|
| `environment-card.schema.json` | One `entry.data` Environment Card |
| `catalog.schema.json` | A complete repository snapshot. Its `DiscoveryEntry` definition describes each entry |
| `declaration.schema.json` | Producer-side `discovery.json` input, not a discovered resource |

The schema `$id` is an identifier, not an instruction to fetch it or a
guarantee that a draft is published there. Pin the agreed schema/profile
revision during review. Loading these self-contained schemas
does not require fetching candidate resource URLs.

### Consumer validation

JSON Schema validation is necessary but not sufficient. The schemas enforce
object shape, required fields, relative-path safety, supported literals,
conditional artifact and license-evidence presence, and exactly one
orchestration interface. Relative paths permit printable UTF-8 (including
spaces) but reject C0, DEL, C1, and Unicode line and paragraph separators. Other
rules require semantic validation:

| Subject | Additional rule |
|---------|-----------------|
| Repository source | Parse a credential-free GitHub HTTPS Git URI with no query, fragment or encoded path components. Its repository identity must equal `source.id` |
| Artifact binding | Every artifact's `uri`, `path` and `revision` must equal the selected `data.source` tuple |
| Agent-tool declaration | `source_revision` must equal the environment revision. Agent-tool protocols must be unique |
| License | Parse an SPDX expression or the explicit `other`/`unknown` sentinel. Evidence URLs must be credential-free HTTPS references |
| Framework requirement | Parse the declared package requirement and verify that its normalized package name is `openenv`. This is not runtime compatibility evidence |
| Entry identity | Require the canonical `urn:air:` prefix and the selected full source revision suffix. Within a snapshot, the publisher must also match |
| Capabilities | Require a source-bound agent-tool declaration. Known simulation-control names are forbidden regardless of case |
| Snapshot | Check source and publisher consistency, unique identities and paths, direct-child inventory scope, complete accounting and the snapshot digest before trusting a refresh |

The Python models enforce these rules. Consumers in other languages must
implement equivalent checks, and passing JSON Schema alone is not full
conformance. A malformed or unsupported card must not become a resolved,
validated or installable environment through guessed defaults.

### Source retrieval and setup

Discovery reads metadata only. Source acquisition, installation and execution
are separate actions:

1. Select and validate a complete card, preserving its full identifier and
   snapshot provenance.
2. Only `artifact_availability: resolvable` supplies an immutable artifact
   locator. `external` and `unknown` do not authorize a guessed download.
3. After the caller's policy permits retrieval, use the Git artifact's `uri`,
   full `revision` and repository-relative `path`, keeping any required monorepo
   build context. The URN is not a URL, and the GitHub repository ID is not a
   Hub Space identifier.
4. Review the environment's setup instructions at the same revision. Installing
   the repository root does not necessarily install the selected environment.

The card does not specify an installer, OCI image digest, dependency lock,
launch arguments, resource budgets or execution permissions. A declared MCP
agent-tool interface is not enough to construct an MCP-server install action. A
`resolvable` card does not establish caller access, a running deployment,
reproducible build output, validated interfaces or approval to execute.

### Identifiers and publication

The identifier is publisher-scoped and revision-qualified. Its locator component
is SHA-256 over the declared repository URI, a newline and the environment path.
It is not a global canonical identity. A publisher transfer requires an explicit
mapping.

Publish the complete generated JSON through an owner-controlled, versioned
metadata channel. Consumers pin the profile they understand and inspect the
whole `entries[].data` card. A third-party adapter may extract entries but must
retain source, path, revision and the snapshot reference. Publication alone does
not guarantee admission to or indexing by any finder.

The `Discovery catalog` workflow generates a revision-named GitHub Actions
artifact from the checked-out source, using the hosting GitHub domain and
repository-owner namespace without claiming verified publisher status. It is a
reviewable output, not a hosted registry. On PRs it identifies the PR merge
revision.

`compare_catalogs(previous, current)` distinguishes added listings, withdrawn
listings, superseded revision cards and corrected metadata for the same
revision. Both inputs must be complete snapshots of the same publisher, source
and inventory scope. Failed reads cannot authorize removal. Historical snapshots
remain usable as explicit historical metadata.
