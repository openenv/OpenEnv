# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""`probe_logprobs_mode` must say "unknown", not "raw", when the distribution is saturated.

The probe decides raw-vs-processed from how the top-two logprob GAP scales with temperature: `T1/T2`
when processed, `1.0` when raw. It measures at completion position 1 because that is the only
position independent of sampling.

But a reasoning model's chat template FORCES its opening token -- `<think>` for Qwen3, at p~1.0 with
every alternative at -inf. Engines report -inf as a sentinel (vLLM: -9999), and +-inf is unchanged by
division, so the gap is identical at both temperatures and the ratio is 1.0. The probe then reports
"raw", `validate_llm` demotes `capture_level` from `tokens` to `logprobs`, and every rollout comes
back 409 -- which reads as "this model cannot train" when the truth is "this probe cannot measure
here".

Measured on a live Qwen3-8B served WITH `--logprobs-mode processed_logprobs`:

    position 1 (forced `<think>`)         gap 9999.0 @T=1.0 -> 9999.0 @T=2.0   ratio 1.000
    position 1, prefilled past `<think>`  gap 3.7500 @T=1.0 -> 1.8750 @T=2.0   ratio 0.500

So "raw" was wrong about a correctly-configured engine. "unknown" is what the function documents for
this case: not a failure, only an absence of evidence -- and unlike "raw" it does not demote the tier.
"""

from __future__ import annotations

import pytest

validate_llm = pytest.importorskip("openenv.core.harness.capture.validate_llm")


def _payload(top: list[float]) -> dict:
    """A chat completion whose first position carries `top` as its `top_logprobs`."""
    return {
        "choices": [
            {"logprobs": {"content": [{"top_logprobs": [{"logprob": v} for v in top]}]}}
        ]
    }


def _probe_with(monkeypatch, per_temperature: list[list[float]]) -> str:
    """Run the probe against canned responses, one per temperature it asks about."""
    answers = list(per_temperature)

    def fake_post(url, body, timeout, api_key=None, auth_header="Authorization"):
        return _payload(answers.pop(0))

    monkeypatch.setattr(validate_llm, "_post", fake_post)
    return validate_llm.probe_logprobs_mode("http://engine", "some/model")


def test_saturated_gap_is_unknown_not_raw(monkeypatch):
    # The real shape: top token at 0.0, runners-up at the -inf sentinel, unchanged by temperature.
    mode = _probe_with(monkeypatch, [[0.0, -9999.0, -9999.0], [0.0, -9999.0, -9999.0]])
    assert mode == "unknown", (
        "a sentinel gap is an absence of evidence; calling it 'raw' demotes a correctly-configured "
        "engine to the eval tier and every rollout then 409s"
    )


def test_genuinely_raw_is_still_detected(monkeypatch):
    # The guard must not blind the check it lives in: a real, unchanging gap is still raw.
    mode = _probe_with(monkeypatch, [[-1.0, -7.75], [-1.0, -7.75]])
    assert mode == "raw"


def test_processed_is_still_detected(monkeypatch):
    # Gap halves as temperature doubles -> processed. Matches the measured 3.75 -> 1.875.
    mode = _probe_with(monkeypatch, [[-1.0, -4.75], [-1.0, -2.875]])
    assert mode == "processed"


def test_flat_distribution_remains_unknown(monkeypatch):
    # The pre-existing guard at the other extreme, kept honest alongside the new one.
    mode = _probe_with(monkeypatch, [[-1.0, -1.2], [-1.0, -1.1]])
    assert mode == "unknown"


def test_sentinel_threshold_admits_real_tail_values(monkeypatch):
    # A deep but REAL tail value must still be measured, or the guard would swallow valid data.
    mode = _probe_with(monkeypatch, [[-1.0, -41.0], [-1.0, -21.0]])
    assert mode == "processed"
