"""Offline spend accounting, durability and real process/thread coordination."""

import json
import multiprocessing
import stat
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from decimal import Decimal, localcontext
from fractions import Fraction

import pytest
from examples.hub_validation_pipeline import spend


@pytest.mark.parametrize("operation", ["reserve", "set_limit"])
def test_idempotent_retry_must_confirm_directory_durability(
    ledger, monkeypatch, operation
):
    real_fsync = spend.os.fsync
    calls = []

    def fail_directory(fd):
        directory = stat.S_ISDIR(spend.os.fstat(fd).st_mode)
        calls.append("directory" if directory else "file")
        if directory:
            raise OSError("injected directory fsync failure")
        return real_fsync(fd)

    monkeypatch.setattr(spend.os, "fsync", fail_directory)

    def retry():
        if operation == "reserve":
            return spend.reserve(ledger, "durability", "1", "work")
        return spend.set_limit(ledger, "5000", "reviewed example budget increase")

    with pytest.raises(OSError):
        retry()
    assert calls == ["file", "directory"]
    before = ledger.read_bytes()
    calls.clear()
    with pytest.raises(OSError):
        retry()
    assert calls == ["directory"]
    assert ledger.read_bytes() == before
    monkeypatch.setattr(spend.os, "fsync", real_fsync)
    report = retry()
    assert ledger.read_bytes() == before
    if operation == "reserve":
        assert report["charged_usd_exact"] == "1"
    else:
        assert report["limit_usd_exact"] == "5000"
        assert len(json.loads(ledger.read_text())["limit_changes"]) == 1


@pytest.mark.parametrize("operation", ["summary", "reserve", "settle", "set_limit"])
def test_saved_settlement_without_evidence_cannot_release_capacity(ledger, operation):
    spend.reserve(ledger, "unfinished", "1", "unknown cost")
    data = json.loads(ledger.read_text())
    data["reservations"]["unfinished"].update(status="settled", spent_usd="0")
    ledger.write_text(json.dumps(data))
    original = ledger.read_bytes()
    with pytest.raises(ValueError):
        if operation == "summary":
            spend.summary(ledger)
        elif operation == "reserve":
            spend.reserve(ledger, "new", "1", "work")
        elif operation == "set_limit":
            spend.set_limit(ledger, "5000", "reviewed example budget increase")
        else:
            spend.settle(ledger, "unfinished", "0", "new receipt")
    assert ledger.read_bytes() == original


def _reserve_worker(path, barrier, results, amount, attempts, limit=None):
    barrier.wait(timeout=10)
    if limit is not None:
        spend.set_limit(path, limit, "reviewed example budget increase")
    successes = 0
    for index in range(attempts):
        try:
            spend.reserve(
                path,
                f"{multiprocessing.current_process().name}-{index}",
                amount,
                "race",
            )
        except ValueError:
            continue
        successes += 1
    results.put(successes)


@pytest.fixture
def ledger(tmp_path):
    path = tmp_path / "ledger.json"
    spend.init(path, "10")
    return path


def test_existing_schema_allocations_and_unknown_metadata_are_preserved(tmp_path):
    path = tmp_path / "ledger.json"
    data = {
        "schema": "openenv-shared-spend-v1",
        "limit_usd": "500",
        "created_utc": "original creation",
        "updated_utc": "original update",
        "checked_utc": "keep this timestamp",
        "metadata": {"nested": [1, "kept", None]},
        "reservations": {
            identifier: dict(
                reserved_usd=amount,
                spent_usd=None,
                status="reserved",
                purpose="existing allocation",
                evidence=None,
                reserved_utc="original reservation",
                metadata={"preserve": True},
            )
            for identifier, amount in (
                ("cohort-a", "48.123456"),
                ("cohort-b", "150"),
                ("image-probe", "5"),
                ("generation-cohort", "100"),
            )
        },
    }
    path.write_text(json.dumps(data))
    before = path.read_bytes()
    report = spend.summary(path)
    assert Fraction(report["charged_usd_exact"]) == Fraction("303.123456")
    assert Fraction(report["remaining_usd_exact"]) == Fraction("196.876544")
    assert report["remaining_usd_rounded_up"] == "196.876544"
    assert spend.init(path, "500.000") == report
    assert path.read_bytes() == before
    with pytest.raises(ValueError, match="limit differs"):
        spend.init(path, "501")
    assert path.read_bytes() == before
    spend.reserve(path, "last", "196.876544", "remaining allocation")
    before = path.read_bytes()
    with pytest.raises(ValueError):
        spend.reserve(path, "too-much", "1e-300", "must fail exactly")
    assert path.read_bytes() == before
    reservations = json.loads(before)["reservations"]
    report = spend.set_limit(path, "5000", "reviewed example budget increase")
    assert report["remaining_usd_exact"] == "4500"
    raised = json.loads(path.read_text())
    assert raised["reservations"] == reservations
    assert raised["reservations"]["generation-cohort"]["reserved_usd"] == "100"
    # The existing writer's reservation serialization is unchanged by a limit update.
    reservation_bytes = json.dumps({"reservations": reservations}, indent=2)[1:-2]
    assert reservation_bytes.encode() in before
    assert reservation_bytes.encode() in path.read_bytes()
    history = raised["limit_changes"]
    assert len(history) == 1
    assert history[0] == {
        "previous_limit_usd": "500",
        "limit_usd": "5000",
        "evidence": "reviewed example budget increase",
        "changed_utc": history[0]["changed_utc"],
    }
    assert datetime.fromisoformat(history[0]["changed_utc"]).tzinfo == timezone.utc
    spend.reserve(path, "additional", "4500", "newly approved capacity")
    before = path.read_bytes()
    with pytest.raises(ValueError):
        spend.reserve(path, "still-too-much", "1e-300", "must fail exactly")
    assert path.read_bytes() == before
    spend.settle(path, "cohort-a", "40", "receipt")
    saved = json.loads(path.read_text())
    assert saved["limit_usd"] == "5000"
    assert saved["limit_changes"] == history
    for key in ("schema", "created_utc", "checked_utc", "metadata"):
        assert saved[key] == data[key]
    assert saved["updated_utc"] != data["updated_utc"]
    for identifier, original in data["reservations"].items():
        row = saved["reservations"][identifier]
        if identifier != "cohort-a":
            assert row == original
        else:
            assert row["reserved_utc"] == original["reserved_utc"]
            assert row["metadata"] == original["metadata"]
            assert row["reserved_usd"] == original["reserved_usd"]


def test_reservation_and_settlement_are_idempotent_without_reusing_ids(ledger):
    first = spend.reserve(ledger, "run", "1.0", "purpose", "plan")
    before = ledger.read_bytes()
    assert spend.reserve(ledger, "run", Decimal("1.000"), "retry", "other") == first
    assert ledger.read_bytes() == before
    with pytest.raises(ValueError, match="fresh ID"):
        spend.reserve(ledger, "run", "2", "purpose")
    assert ledger.read_bytes() == before
    settled = spend.settle(ledger, "run", "0.7", "receipt")
    before = ledger.read_bytes()
    assert spend.settle(ledger, "run", "0.700", "retry receipt") == settled
    assert ledger.read_bytes() == before
    with pytest.raises(ValueError, match="manual reconciliation"):
        spend.settle(ledger, "run", "0.8", "different receipt")
    with pytest.raises(ValueError, match="fresh ID"):
        spend.reserve(ledger, "run", "1.0", "reused")
    assert ledger.read_bytes() == before


def test_actual_overspend_is_recorded_and_blocks_new_reservations(ledger):
    spend.reserve(ledger, "a", "6", "first")
    spend.reserve(ledger, "b", "4", "second")
    report = spend.settle(ledger, "a", "12.000000000000000000000000000001", "receipt")
    actual = Fraction("12.000000000000000000000000000001")
    assert Fraction(report["charged_usd_exact"]) == actual + 4
    assert Fraction(report["remaining_usd_exact"]) == 6 - actual
    assert report["over_budget"] is True
    saved = json.loads(ledger.read_text())
    assert saved["over_budget"] is True
    assert Fraction(saved["reservations"]["a"]["spent_usd"]) == actual
    before = ledger.read_bytes()
    with pytest.raises(ValueError, match="blocked"):
        spend.reserve(ledger, "c", "0", "blocked even at zero")
    assert ledger.read_bytes() == before
    # Other real costs can still be settled after the cap is exceeded.
    report = spend.settle(ledger, "b", "0", "cancelled before billing")
    assert Fraction(report["charged_usd_exact"]) == actual
    assert report["statuses"] == {"a": "settled", "b": "settled"}
    for key in ("limit", "reserved", "spent", "charged", "remaining"):
        exact = Fraction(report[f"{key}_usd_exact"])
        displayed = Fraction(report[f"{key}_usd_rounded_up"])
        assert exact <= displayed < exact + Fraction(1, 1000000)


def test_reservation_overrun_below_shared_cap_also_freezes_new_work(ledger):
    spend.reserve(ledger, "a", "1", "first")
    report = spend.settle(ledger, "a", "2", "receipt")
    assert report["over_budget"] is True
    assert report["remaining_usd_exact"] == "8"
    before = json.loads(ledger.read_text())
    report = spend.set_limit(ledger, "5000", "reviewed example budget increase")
    assert report["over_budget"] is True
    assert report["remaining_usd_exact"] == "4998"
    saved = json.loads(ledger.read_text())
    assert saved["over_budget"] is True
    assert saved["reservations"] == before["reservations"]
    with pytest.raises(ValueError):
        spend.reserve(ledger, "b", "1", "blocked")


def test_normal_settlement_releases_unused_reservation(ledger):
    spend.reserve(ledger, "a", "10", "first")
    assert spend.settle(ledger, "a", "2", "receipt")["over_budget"] is False
    report = spend.reserve(ledger, "b", "8", "released capacity")
    assert report["charged_usd_exact"] == "10"
    assert report["reserved_usd_exact"] == "8"
    assert report["spent_usd_exact"] == "2"


@pytest.mark.parametrize(
    "bad",
    [
        True,
        False,
        0.1,
        float("nan"),
        float("inf"),
        Decimal("NaN"),
        Decimal("sNaN"),
        Decimal("Infinity"),
        "NaN",
        "Infinity",
        "-1",
        "-0",
        "1/2",
        "1_0",
        " 1 ",
        "",
        "malformed",
        None,
        [],
        {},
    ],
)
def test_invalid_money_is_rejected_without_partial_writes(ledger, tmp_path, bad):
    missing = tmp_path / "never-created.json"
    with pytest.raises(ValueError):
        spend.init(missing, bad)
    assert not missing.exists()
    spend.reserve(ledger, "a", "1", "valid")
    before = ledger.read_bytes()
    with pytest.raises(ValueError):
        spend.reserve(ledger, "bad", bad, "invalid")
    with pytest.raises(ValueError):
        spend.settle(ledger, "a", bad, "receipt")
    with pytest.raises(ValueError):
        spend.set_limit(ledger, bad, "reviewed example budget increase")
    assert ledger.read_bytes() == before


def test_tiny_values_large_integers_and_decimal_context_do_not_round(tmp_path):
    path = tmp_path / "precise.json"
    with localcontext() as context:
        context.prec = 2
        spend.init(path, Decimal("1.0000000000000000000000000000000000000001"))
        spend.reserve(path, "whole", "1", "exact")
        spend.reserve(path, "remainder", "1e-40", "last exact digit")
        before = path.read_bytes()
        with pytest.raises(ValueError):
            spend.reserve(path, "tiny-over", "1e-300", "cannot round down")
        assert path.read_bytes() == before
        tiny = tmp_path / "tiny.json"
        spend.init(tiny, "1e-300")
        spend.reserve(tiny, "one", "1e-301", "tiny")
        report = spend.reserve(tiny, "nine", "9e-301", "tiny")
        assert report["remaining_usd_exact"] == "0"
        assert Fraction(report["charged_usd_exact"]) == Fraction(Decimal("1e-300"))
        assert report["charged_usd_rounded_up"] == "0.000001"
        large = tmp_path / "large.json"
        spend.init(large, 9007199254740993)
        report = spend.reserve(large, "large", 9007199254740992, "exact integer")
        assert report["remaining_usd_exact"] == "1"


def test_limit_changes_are_exact_append_only_and_idempotent(ledger, tmp_path):
    missing = tmp_path / "missing.json"
    with pytest.raises(ValueError, match="initialized"):
        spend.set_limit(missing, "5000", "reviewed example budget increase")
    assert not missing.exists()
    with localcontext() as context:
        context.prec = 2
        exact = Decimal("5000.0000000000000000000000000000000000000001")
        report = spend.set_limit(ledger, exact, "reviewed example budget increase")
        assert Fraction(report["limit_usd_exact"]) == Fraction(exact)
        before = ledger.read_bytes()
        assert spend.set_limit(ledger, str(exact), "retry evidence") == report
        assert ledger.read_bytes() == before
        first = json.loads(before)["limit_changes"][0]
        report = spend.set_limit(ledger, "1e-300", "another decision")
        assert Fraction(report["limit_usd_exact"]) == Fraction("1e-300")
        changes = json.loads(ledger.read_text())["limit_changes"]
        assert len(changes) == 2
        assert changes[0] == first
        assert changes[1]["previous_limit_usd"] == str(exact)
        report = spend.set_limit(ledger, 9007199254740993, "large exact decision")
        assert report["limit_usd_exact"] == "9007199254740993"


def test_limit_reduction_latches_over_budget_and_raise_cannot_clear_it(ledger):
    spend.reserve(ledger, "held", "5", "pending billing")
    assert spend.set_limit(ledger, "4", "reduced ceiling")["over_budget"] is True
    assert spend.set_limit(ledger, "5000", "reviewed example budget increase")[
        "over_budget"
    ]
    with pytest.raises(ValueError, match="blocked"):
        spend.reserve(ledger, "blocked", "0", "still blocked")


@pytest.mark.parametrize("saved_flag", [False, True])
def test_saved_over_budget_condition_survives_limit_increase(ledger, saved_flag):
    spend.reserve(ledger, "held", "10", "existing hold")
    data = json.loads(ledger.read_text())
    data["limit_usd"] = "9" if not saved_flag else "10"
    if saved_flag:
        data["over_budget"] = True
    ledger.write_text(json.dumps(data))
    assert spend.set_limit(ledger, "5000", "reviewed example budget increase")[
        "over_budget"
    ]
    assert json.loads(ledger.read_text())["over_budget"] is True


@pytest.mark.parametrize(
    "history",
    [
        None,
        {},
        "invalid",
        [None],
        [{}],
        [{"previous_limit_usd": "500", "limit_usd": "10", "evidence": "decision"}],
    ],
)
def test_malformed_limit_history_fails_closed(ledger, history):
    data = json.loads(ledger.read_text())
    data["limit_changes"] = history
    ledger.write_text(json.dumps(data))
    before = ledger.read_bytes()
    with pytest.raises(ValueError):
        spend.set_limit(ledger, "5000", "reviewed example budget increase")
    with pytest.raises(ValueError):
        spend.summary(ledger)
    assert ledger.read_bytes() == before


@pytest.mark.parametrize(
    "corruption",
    [
        {"limit_usd": 10.0},
        {"previous_limit_usd": "-1"},
        {"evidence": " "},
        {"changed_utc": "bad timestamp"},
        {"changed_utc": "2026-11-10T00:00:00"},
        {"changed_utc": "2026-11-10T00:00:00+01:00"},
        {"limit_usd": "9"},
        {"previous_limit_usd": "10"},
    ],
)
def test_corrupt_limit_change_row_fails_closed(ledger, corruption):
    change = dict(
        previous_limit_usd="500",
        limit_usd="10",
        evidence="decision",
        changed_utc="2026-11-10T00:00:00+00:00",
    )
    test_malformed_limit_history_fails_closed(ledger, [change | corruption])


def test_discontinuous_limit_history_fails_closed(ledger):
    change = dict(
        previous_limit_usd="500",
        limit_usd="9",
        evidence="decision",
        changed_utc="2026-11-10T00:00:00+00:00",
    )
    test_malformed_limit_history_fails_closed(
        ledger, [change, change | {"previous_limit_usd": "8", "limit_usd": "10"}]
    )


@pytest.mark.parametrize(
    "indent,newline,ensure_ascii", [(None, "\n", True), (4, "\r\n", False)]
)
def test_limit_change_preserves_original_reservation_bytes(
    tmp_path, indent, newline, ensure_ascii
):
    path = tmp_path / "ledger.json"
    reservations = json.dumps(
        {
            "tasksmith": dict(
                reserved_usd="100.0000",
                spent_usd=None,
                status="reserved",
                purpose='original "reservations": note',
                evidence="original évidence",
                metadata={"keep": 1.25},
            )
        },
        indent=indent,
        ensure_ascii=ensure_ascii,
    )
    reservations = reservations.replace("\n", newline).replace(
        "1.25", "1.25000000000000000000000000001"
    )
    path.write_text(
        '{"schema":"openenv-shared-spend-v1","limit_usd":"500",'
        '"metadata":{"reservations":"unrelated"},"reservations":' + reservations + "}"
    )
    spend.set_limit(path, "5000", "reviewed example budget increase")
    assert reservations.encode() in path.read_bytes()
    assert spend.summary(path)["reserved_usd_exact"] == "100"


@pytest.mark.parametrize("field", ["limit_usd", "reserved_usd"])
def test_float_ledger_amounts_cannot_be_rewritten_by_limit_change(ledger, field):
    spend.reserve(ledger, "held", "1", "existing hold")
    data = json.loads(ledger.read_text())
    if field == "limit_usd":
        data[field] = 10.0
    else:
        data["reservations"]["held"][field] = 1.0
    ledger.write_text(json.dumps(data))
    before = ledger.read_bytes()
    with pytest.raises(ValueError):
        spend.set_limit(ledger, "5000", "reviewed example budget increase")
    assert ledger.read_bytes() == before


@pytest.mark.parametrize("amount,attempts,expected", [("0.6", 1, 1), ("0.03", 20, 33)])
def test_process_budget_race_and_no_lost_updates(tmp_path, amount, attempts, expected):
    path = tmp_path / "shared.json"
    spend.init(path, "1")
    context = multiprocessing.get_context("spawn")
    barrier, results = context.Barrier(2), context.Queue()
    processes = [
        context.Process(
            target=_reserve_worker, args=(path, barrier, results, amount, attempts)
        )
        for _ in range(2)
    ]
    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=15)
            assert process.exitcode == 0
        successes = sum(results.get(timeout=2) for _ in processes)
        report = spend.summary(path)
        assert successes == expected == len(report["statuses"])
        assert Fraction(report["charged_usd_exact"]) == expected * Fraction(amount) <= 1
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=2)
        results.close()
        results.join_thread()


def test_threads_share_the_same_budget(ledger):
    def reserve(index):
        try:
            spend.reserve(ledger, f"thread-{index}", "1", "thread race")
            return 1
        except ValueError:
            return 0

    with ThreadPoolExecutor(max_workers=8) as executor:
        successes = sum(executor.map(reserve, range(32)))
    report = spend.summary(ledger)
    assert successes == 10 == len(report["statuses"])
    assert report["charged_usd_exact"] == "10"


def test_concurrent_limit_retry_and_reservations_share_the_lock(tmp_path):
    path = tmp_path / "shared.json"
    spend.init(path, "500")
    spend.reserve(path, "tasksmith", "100", "existing allocation")
    context = multiprocessing.get_context("spawn")
    barrier, results = context.Barrier(2), context.Queue()
    processes = [
        context.Process(
            target=_reserve_worker,
            args=(path, barrier, results, "3000", 1, "5000"),
        )
        for _ in range(2)
    ]
    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=15)
            assert process.exitcode == 0
        assert sum(results.get(timeout=2) for _ in processes) == 1
        report = spend.summary(path)
        assert report["charged_usd_exact"] == "3100"
        saved = json.loads(path.read_text())
        assert len(saved["limit_changes"]) == 1
        assert saved["reservations"]["tasksmith"]["reserved_usd"] == "100"
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=2)
        results.close()
        results.join_thread()


def test_concurrent_different_limit_changes_keep_all_history(ledger):
    with ThreadPoolExecutor(max_workers=8) as executor:
        list(
            executor.map(
                lambda limit: spend.set_limit(ledger, limit, "reviewed decision"),
                range(11, 27),
            )
        )
    data = json.loads(ledger.read_text())
    changes = data["limit_changes"]
    assert len(changes) == 16
    assert {row["limit_usd"] for row in changes} == {str(n) for n in range(11, 27)}
    previous = "10"
    for row in changes:
        assert row["previous_limit_usd"] == previous
        previous = row["limit_usd"]
    assert data["limit_usd"] == previous


@pytest.mark.parametrize("evidence", [None, "", " ", True, {}])
def test_settlement_requires_evidence_and_existing_id(ledger, evidence):
    spend.reserve(ledger, "a", "1", "purpose")
    before = ledger.read_bytes()
    with pytest.raises(ValueError):
        spend.settle(ledger, "a", "1", evidence)
    with pytest.raises(ValueError, match="Unknown reservation"):
        spend.settle(ledger, "missing", "1", "receipt")
    with pytest.raises(ValueError):
        spend.set_limit(ledger, "5000", evidence)
    with pytest.raises(ValueError):
        spend.set_limit(ledger, "10", evidence)
    assert ledger.read_bytes() == before


@pytest.mark.parametrize("operation", ["reserve", "set_limit"])
def test_failed_atomic_replace_keeps_original_and_cleans_temporary_file(
    ledger, monkeypatch, operation
):
    before, files = ledger.read_bytes(), set(ledger.parent.iterdir())

    def fail_replace(*args):
        raise OSError("private diagnostic payload")

    monkeypatch.setattr(spend.os, "replace", fail_replace)
    with pytest.raises(OSError):
        if operation == "reserve":
            spend.reserve(ledger, "a", "1", "purpose")
        else:
            spend.set_limit(ledger, "5000", "reviewed example budget increase")
    assert ledger.read_bytes() == before
    assert set(ledger.parent.iterdir()) == files


def test_duplicate_keys_and_corrupt_costs_fail_closed(ledger):
    ledger.write_text(
        '{"schema":"openenv-shared-spend-v1","limit_usd":"10","limit_usd":"100","reservations":{}}'
    )
    before = ledger.read_bytes()
    with pytest.raises(ValueError, match="Duplicate"):
        spend.reserve(ledger, "a", "1", "purpose")
    assert ledger.read_bytes() == before
    ledger.write_text(
        json.dumps(
            dict(
                schema="openenv-shared-spend-v1",
                limit_usd="10",
                reservations={
                    "bad": dict(status="reserved", reserved_usd=True, spent_usd=None)
                },
            )
        )
    )
    before = ledger.read_bytes()
    with pytest.raises(ValueError):
        spend.reserve(ledger, "a", "1", "purpose")
    assert ledger.read_bytes() == before


def test_module_cli_outputs_totals_and_statuses_without_freeform_evidence(
    ledger, capsys
):
    command = [
        sys.executable,
        "-B",
        "-m",
        "examples.hub_validation_pipeline.spend",
        "--ledger",
        str(ledger),
    ]
    result = subprocess.run(
        command
        + [
            "reserve",
            "--id",
            "cli",
            "--usd",
            "3",
            "--purpose",
            "purpose",
            "--evidence",
            "private evidence",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    assert json.loads(result.stdout)["charged_usd_exact"] == "3"
    assert "private evidence" not in result.stdout
    assert not result.stderr
    result = subprocess.run(
        command
        + ["settle", "--id", "cli", "--usd", "2", "--evidence", "private receipt"],
        check=True,
        text=True,
        capture_output=True,
    )
    assert json.loads(result.stdout)["statuses"] == {"cli": "settled"}
    result = subprocess.run(
        command + ["summary"], check=True, text=True, capture_output=True
    )
    assert json.loads(result.stdout)["spent_usd_exact"] == "2"
    result = subprocess.run(
        command
        + [
            "set-limit",
            "--usd",
            "5000",
            "--evidence",
            "reviewed example budget increase",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    assert json.loads(result.stdout)["limit_usd_exact"] == "5000"
    assert json.loads(result.stdout)["spent_usd_exact"] == "2"
    assert "user decision" not in result.stdout
    before = ledger.read_bytes()
    result = subprocess.run(
        command + ["set-limit", "--usd", "6000"], text=True, capture_output=True
    )
    assert result.returncode == 2
    assert ledger.read_bytes() == before
    with pytest.raises(SystemExit) as error:
        spend.main(
            [
                "--ledger",
                str(ledger),
                "reserve",
                "--id",
                "bad",
                "--usd",
                "private invalid amount",
                "--purpose",
                "purpose",
            ]
        )
    assert error.value.code == 2
    output = capsys.readouterr()
    assert not output.out
    assert "private invalid amount" not in output.err
