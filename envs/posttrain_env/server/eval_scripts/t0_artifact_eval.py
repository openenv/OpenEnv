# /// script
# requires-python = ">=3.10"
# ///
# SPDX-License-Identifier: BSD-3-Clause

"""Score `t0_artifact`: 1.0 if `/model/answer.txt` holds the expected text."""

import json
import os
from pathlib import Path

expected = os.environ["POSTTRAIN_EXPECTED"]
answer = Path("/model/answer.txt")

if answer.is_file():
    text = answer.read_text(encoding="utf-8", errors="replace").strip()
    result = {"status": "ok", "score": float(text == expected)}
else:
    result = {"status": "invalid", "score": 0.0, "feedback": "answer.txt not found"}

print("POSTTRAIN_SCORE " + json.dumps(result), flush=True)
