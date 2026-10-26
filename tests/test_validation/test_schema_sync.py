import json
from pathlib import Path

import pytest
from openenv.validation.manifest import NormalizedManifest, NormalizedManifestV2
from openenv.validation.report import ValidationReport, ValidationReportV2
from openenv.validation.runtime.contracts import RuntimePlan

SCHEMAS_DIR = (
    Path(__file__).parent.parent.parent / "src" / "openenv" / "validation" / "schemas"
)

EXPORTS = {
    "manifest.schema.json": NormalizedManifest,
    "report.schema.json": ValidationReport,
    "manifest-v2.schema.json": NormalizedManifestV2,
    "report-v2.schema.json": ValidationReportV2,
    "runtime-plan.schema.json": RuntimePlan,
}


@pytest.mark.parametrize("fname", sorted(EXPORTS))
def test_committed_schema_matches_model(fname):
    committed = (SCHEMAS_DIR / fname).read_text()
    rendered = (
        json.dumps(EXPORTS[fname].model_json_schema(), indent=2, sort_keys=True) + "\n"
    )
    assert committed == rendered, (
        f"{fname} is stale; run: PYTHONPATH=src python scripts/sync_validation_schemas.py --fix"
    )


def test_report_schema_allows_hub_and_statistical_check_ids():
    schema = ValidationReport.model_json_schema()
    pattern = schema["$defs"]["CheckResult"]["properties"]["check_id"]["pattern"]
    import re

    for reserved in ("hub.cross_host_determinism", "statistical.adversarial_floor"):
        assert re.match(pattern, reserved), reserved
