# SPDX-License-Identifier: BSD-3-Clause

"""`python -m openenv.core.openenvd`: run openenvd as the unit's PID 1."""

import sys

from .daemon import main

if __name__ == "__main__":
    sys.exit(main())
