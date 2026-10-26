# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
OpenEnv Wrapper - Generic MCP Integration Package

This package provides a fully generic OpenEnv integration that can be
copied to any MCP project. See README.md for usage instructions.
"""

from .client import MCPEnvClient
from .config import MCP_NAME
from .custom_http_server import MCPHTTPEnvServer
from .data_models import ListToolsAction, MCPAction, MCPObservation, ToolCallAction
from .mcp_env_environment import MCPEnvironment

__all__ = [
    # Configuration
    "MCP_NAME",
    # Data Models
    "MCPAction",
    "MCPObservation",
    "ListToolsAction",
    "ToolCallAction",
    # Environment
    "MCPEnvironment",
    # HTTP Server
    "MCPHTTPEnvServer",
    # Client
    "MCPEnvClient",
]

__version__ = "1.0.0"
__author__ = "OpenEnv MCP Integration"
__description__ = "Generic OpenEnv wrapper for any MCP integration"
