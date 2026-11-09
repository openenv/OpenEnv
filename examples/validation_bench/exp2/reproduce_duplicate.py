#!/usr/bin/env python3
"""Reproduce production artifact loss with a real local protocol server.

PYTHONPATH=src:envs .venv/bin/python examples/validation_bench/exp2/reproduce_duplicate.py --surface wire
Use --surface http for the independent HTTP serialization path. No Docker/HF.
Exit zero means the oracle FAIL -> replay PASS defect was reproduced.
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from incomplete import run_case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--surface", choices=("wire", "http"), default="wire")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/openenv-validation-bench-20261109/exp2/minimal-repros"),
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix=args.surface + "-", dir=args.output))
    row = run_case(directory, f"duplicate_{args.surface}_decisive_evidence", 17, 0)
    changed = row["replay"]["status_mismatches"]
    reproduced = any(
        c["check_id"] == "runtime.oracle_containment"
        and c["original"] == "fail"
        and c["replayed"] == "pass"
        for c in changed
    )
    print(
        json.dumps(
            {
                "reproduced": reproduced,
                "changes": changed,
                "bundle_checksums_valid": row["integrity"]["all_checksums_match"],
                "omission_or_redaction_flags": row["replay"]["limits"],
                "evidence": str(directory),
            },
            indent=2,
        )
    )
    return 0 if reproduced else 1


if __name__ == "__main__":
    raise SystemExit(main())
