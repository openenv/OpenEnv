"""Expose the trusted ThinkingBox server environment.

Importing this package loads only server and shared wire code; the OpenEnv
client remains lazy at the top-level package boundary.
"""

from .thinkingbox_environment import ThinkingBoxEnvironment

__all__ = ["ThinkingBoxEnvironment"]
