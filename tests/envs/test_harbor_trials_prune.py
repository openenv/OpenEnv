# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""`prune_trials`: opt-in cleanup of finished trial directories during long training runs.

Every rollout leaves a Harbor trial directory under `OPENENV_HARBOR_TRIALS_DIR` (default `/tmp`). A
200-step GRPO run with 8 rollouts per step left 6,300 of them (8.8 GB) and filled a shared disk
mid-run. The rollout's trace is already in the returned result, so for training the directory is
scratch. `OPENENV_HARBOR_TRIALS_MAX_AGE_S` enables pruning; unset keeps every directory, as before.
"""

from __future__ import annotations

import os

import pytest

env_mod = pytest.importorskip("openenv.harbor.environment")
prune_trials = env_mod.prune_trials


def trial(root, name, age_s, now, *, inner_age_s=None):
    """A trial directory whose files were last written `age_s` (or `inner_age_s`) seconds ago."""
    d = root / name
    (d / "agent").mkdir(parents=True)
    f = d / "agent" / "trajectory.json"
    f.write_text("{}")
    t = now - (age_s if inner_age_s is None else inner_age_s)
    os.utime(f, (t, t))
    os.utime(d / "agent", (now - age_s, now - age_s))
    os.utime(d, (now - age_s, now - age_s))
    return d


NOW = 1_800_000_000.0


def test_old_trials_are_removed_and_recent_ones_kept(tmp_path):
    old = trial(tmp_path, "oe-old", 3600, NOW)
    new = trial(tmp_path, "oe-new", 60, NOW)
    assert prune_trials(tmp_path, 1800, now=NOW) == 1
    assert not old.exists()
    assert new.exists()


def test_a_trial_still_being_written_is_kept(tmp_path):
    """The directory's own mtime can be old while a file inside was just written."""
    live = trial(tmp_path, "oe-live", 3600, NOW, inner_age_s=5)
    assert prune_trials(tmp_path, 1800, now=NOW) == 0
    assert live.exists()


@pytest.mark.parametrize("max_age_s", [0, -1])
def test_zero_or_negative_age_is_a_no_op(tmp_path, max_age_s):
    old = trial(tmp_path, "oe-old", 10**6, NOW)
    assert prune_trials(tmp_path, max_age_s, now=NOW) == 0
    assert old.exists()


def test_only_direct_subdirectories_are_touched(tmp_path):
    root = tmp_path / "trials"
    root.mkdir()
    stray = root / "notes.txt"
    stray.write_text("keep me")
    os.utime(stray, (NOW - 10**6, NOW - 10**6))
    old = trial(root, "oe-old", 10**6, NOW)
    assert prune_trials(root, 1800, now=NOW) == 1
    assert not old.exists()
    assert stray.exists(), "plain files in the trials dir are not trials"


def test_a_symlinked_directory_is_not_followed(tmp_path):
    target = tmp_path / "elsewhere"
    target.mkdir()
    (target / "data").write_text("x")
    os.utime(target / "data", (NOW - 10**6, NOW - 10**6))
    os.utime(target, (NOW - 10**6, NOW - 10**6))
    root = tmp_path / "trials"
    root.mkdir()
    (root / "link").symlink_to(target, target_is_directory=True)
    assert prune_trials(root, 1800, now=NOW) == 0
    assert (target / "data").exists()


def test_missing_directory_is_a_no_op(tmp_path):
    assert prune_trials(tmp_path / "absent", 60, now=NOW) == 0


def test_max_age_env_parsing(monkeypatch):
    monkeypatch.delenv("OPENENV_HARBOR_TRIALS_MAX_AGE_S", raising=False)
    assert env_mod._trials_max_age_s() == 0.0
    monkeypatch.setenv("OPENENV_HARBOR_TRIALS_MAX_AGE_S", "1800")
    assert env_mod._trials_max_age_s() == 1800.0
    for bad in ("soon", "-5"):
        monkeypatch.setenv("OPENENV_HARBOR_TRIALS_MAX_AGE_S", bad)
        with pytest.raises(ValueError):
            env_mod._trials_max_age_s()
