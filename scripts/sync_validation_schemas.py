#!/usr/bin/env python3
"""Sync committed validation JSON Schemas with their pydantic models.

The pydantic models in src/openenv/validation/ are the source of truth;
src/openenv/validation/schemas/*.schema.json are committed exports. This script
keeps them in sync (same pattern as scripts/sync_env_docs.py).

Modes:
  --check : Exit non-zero if out of sync (for CI)
  --fix   : Regenerate the committed schema files from the models

Run with: PYTHONPATH=src python scripts/sync_validation_schemas.py --check
"""

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

SCHEMAS_DIR = os.path.join(ROOT, "src", "openenv", "validation", "schemas")


def rendered_schemas():
    """Return {filename: rendered JSON text} for every exported schema."""
    from openenv.validation.manifest import NormalizedManifest, NormalizedManifestV2
    from openenv.validation.report import ValidationReport, ValidationReportV2
    from openenv.validation.runtime.contracts import RuntimePlan

    exports = {
        "manifest.schema.json": NormalizedManifest,
        "report.schema.json": ValidationReport,
        "manifest-v2.schema.json": NormalizedManifestV2,
        "report-v2.schema.json": ValidationReportV2,
        "runtime-plan.schema.json": RuntimePlan,
    }
    return {
        fname: json.dumps(model.model_json_schema(), indent=2, sort_keys=True) + "\n"
        for fname, model in exports.items()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--check", action="store_true", help="exit non-zero if out of sync"
    )
    mode.add_argument("--fix", action="store_true", help="regenerate committed schemas")
    args = parser.parse_args()

    os.makedirs(SCHEMAS_DIR, exist_ok=True)
    stale = []
    for fname, rendered in rendered_schemas().items():
        path = os.path.join(SCHEMAS_DIR, fname)
        existing = None
        if os.path.exists(path):
            with open(path) as f:
                existing = f.read()
        if existing == rendered:
            continue
        if args.fix:
            with open(path, "w") as f:
                f.write(rendered)
            print(f"wrote {os.path.relpath(path, ROOT)}")
        else:
            stale.append(fname)

    if stale:
        print("committed schemas are out of sync with the pydantic models:")
        for fname in stale:
            print(f"  {fname}")
        print("run: PYTHONPATH=src python scripts/sync_validation_schemas.py --fix")
        return 1
    if args.check:
        print("validation schemas in sync")
    return 0


if __name__ == "__main__":
    sys.exit(main())
