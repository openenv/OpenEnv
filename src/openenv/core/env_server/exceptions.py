# SPDX-License-Identifier: BSD-3-Clause

"""Custom exceptions for environment server operations."""

from typing import Optional


class OpenEnvError(Exception):
    """Base exception for all OpenEnv errors."""

    pass


class ConcurrencyConfigurationError(OpenEnvError):
    """
    Raised when an environment is misconfigured for concurrent sessions.

    This error is raised during server startup when max_concurrent_envs > 1
    is specified for an environment that is not marked as SUPPORTS_CONCURRENT_SESSIONS.
    """

    def __init__(
        self,
        environment_name: str,
        max_concurrent_envs: int,
        message: Optional[str] = None,
    ):
        self.environment_name = environment_name
        self.max_concurrent_envs = max_concurrent_envs

        if message is None:
            message = (
                f"Environment '{environment_name}' is not marked as SUPPORTS_CONCURRENT_SESSIONS. "
                f"Cannot run with max_concurrent_envs={max_concurrent_envs}. "
                f"Either set max_concurrent_envs=1 or ensure the environment "
                f"properly isolates session state and set SUPPORTS_CONCURRENT_SESSIONS=True."
            )

        super().__init__(message)


class SessionCapacityError(OpenEnvError):
    """
    Raised when the server cannot accept new sessions due to capacity limits.

    This error is raised when a new WebSocket connection is attempted but
    the server has already reached max_concurrent_envs active sessions.
    """

    def __init__(
        self,
        active_sessions: int,
        max_sessions: int,
        message: Optional[str] = None,
    ):
        self.active_sessions = active_sessions
        self.max_sessions = max_sessions

        if message is None:
            message = (
                f"Server at capacity: {active_sessions}/{max_sessions} sessions active. "
                f"Cannot accept new connections."
            )

        super().__init__(message)


class SessionNotFoundError(OpenEnvError):
    """Raised when attempting to access a session that does not exist."""

    def __init__(self, session_id: str, message: Optional[str] = None):
        self.session_id = session_id

        if message is None:
            message = f"Session '{session_id}' not found."

        super().__init__(message)


class SessionCreationError(OpenEnvError):
    """Raised when a session cannot be created."""

    def __init__(self, reason: str, message: Optional[str] = None):
        self.reason = reason

        if message is None:
            message = f"Failed to create session: {reason}"

        super().__init__(message)


class EnvironmentFactoryError(OpenEnvError):
    """Raised when the environment factory fails to create an instance."""

    def __init__(self, factory_name: str, message: Optional[str] = None):
        self.factory_name = factory_name

        if message is None:
            message = f"Environment factory '{factory_name}' failed to create instance."

        super().__init__(message)
