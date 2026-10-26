# SPDX-License-Identifier: BSD-3-Clause

"""
Route configuration utilities for declarative FastAPI route registration.

This module provides utilities to reduce boilerplate in route registration
by using configuration objects instead of repeated function calls.
"""

from dataclasses import dataclass
from typing import Callable, List, Type

from fastapi import FastAPI
from pydantic import BaseModel


@dataclass
class GetEndpointConfig:
    """Configuration for a simple GET endpoint."""

    path: str
    handler: Callable[[], BaseModel | dict]
    response_model: Type[BaseModel] | type[dict]
    tag: str
    summary: str
    description: str


def _make_get_endpoint(
    handler: Callable[[], BaseModel | dict],
) -> Callable[[], BaseModel | dict]:
    """Wrap a sync GET handler in the async endpoint FastAPI expects."""

    async def endpoint() -> BaseModel | dict:
        return handler()

    return endpoint


def register_get_endpoints(app: FastAPI, configs: List[GetEndpointConfig]) -> None:
    """
    Register multiple GET endpoints from configuration.

    Args:
        app ([`~fastapi.FastAPI`]):
            FastAPI application instance.
        configs (`List[GetEndpointConfig]`):
            List of GET endpoint configurations.
    """
    for config in configs:
        app.get(
            config.path,
            response_model=config.response_model,
            tags=[config.tag],
            summary=config.summary,
            description=config.description,
        )(_make_get_endpoint(config.handler))
