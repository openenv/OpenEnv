"""Run a bounded operator coding audit and retain its evidence bundle."""

import argparse
import signal
import sys
import threading
from contextlib import contextmanager
from pathlib import Path

from openenv.validation.runtime.json_values import dumps, loads

from .runner import run_coding_validation


@contextmanager
def _sigterm_as_interrupt():
    if threading.current_thread() is not threading.main_thread():
        yield
        return

    def interrupt(signum, frame):
        raise KeyboardInterrupt

    previous = signal.signal(signal.SIGTERM, interrupt)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New directory for the report, evidence and checksummed artifacts",
    )
    parser.add_argument("--json", action="store_true", dest="json_output")
    args = parser.parse_args(argv)
    try:
        config = loads(args.config.read_text())
        with _sigterm_as_interrupt():
            report = run_coding_validation(args.task, config, args.output)
    except (OSError, ValueError) as error:
        print(
            f"Coding audit configuration/input failure ({type(error).__name__})",
            file=sys.stderr,
        )
        return 2
    if args.json_output:
        print(dumps(report))
    else:
        print(f"Coding audit: {report['verdict'].upper()} ({report['profile']})")
        for name, check in report["checks"].items():
            print(f"  {check['status'].upper():5s} {name}")
    return 0 if report["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
