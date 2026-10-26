"""
Default configuration for AWM environment server.

Can be overridden via environment variables
"""

import os

# -- Server capacity --
MAX_CONCURRENT_ENVS: int = 10000

# -- Subprocess startup --
READY_TIMEOUT: float = float(os.environ.get("OPENENV_AWM_READY_TIMEOUT", "180"))
READY_POLL_INTERVAL: float = 0.5  # polling interval during startup check
MAX_PORT_RETRIES: int = int(
    os.environ.get("OPENENV_AWM_MAX_PORT_RETRIES", "5")
)  # port-retry attempts on startup failure
RETRY_READY_TIMEOUT: float = float(
    os.environ.get("OPENENV_AWM_RETRY_READY_TIMEOUT", "30.0")
)  # shorter timeout for retry attempts

# -- Idle cleanup --
MAX_IDLE_TIME: float = float(os.environ.get("OPENENV_AWM_MAX_IDLE_TIME", "600"))
ALLOWED_IDLE_SESSIONS: int = int(
    os.environ.get("OPENENV_AWM_ALLOWED_IDLE_SESSIONS", "3000")
)  # max sessions before considering subprocess idle
CLEANUP_INTERVAL: float = float(
    os.environ.get("OPENENV_AWM_CLEANUP_INTERVAL", "5.0")
)  # how often the daemon thread scans

# Default reward config: complete=1.0, incomplete=0.1, format_error=-1.0, others=0.0
DEFAULT_REWARD_CONFIG: dict[str, float] = {
    "complete": 1.0,
    "incomplete": 0.1,
    "format_error": -1.0,
}
