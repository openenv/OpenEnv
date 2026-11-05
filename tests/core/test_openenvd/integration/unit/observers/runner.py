"""Runner observer: reads the agent's workspace, never sees assets."""

import json
import os

workspace = os.environ.get("OPENENVD_WORKSPACE", "/workspace")
assets = os.environ.get("OPENENVD_ASSETS", "/assets")
out = os.environ.get("OPENENVD_OUT", "/out")
report = {
    "workspace_files": sorted(os.listdir(workspace))
    if os.path.isdir(workspace)
    else [],
    "assets_visible": bool(os.path.isdir(assets) and os.listdir(assets)),
}
with open(os.path.join(out, "report.json"), "w") as f:
    json.dump(report, f)
with open(os.path.join(out, "verdict.json"), "w") as f:
    json.dump({"ran": True, **report}, f)
