# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the OpenApp Environment.

The OpenApp environment provides a simulated web application environment
for training and evaluating UI agents that interact with various apps
(calendar, todo, messenger, maps, etc.) using browser actions.
"""

from typing import Any, Dict, List, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class OpenAppAction(Action):
    """
    Action for the OpenApp environment.

    Supports BrowserGym-style actions for web interaction:
    - click: Click on an element (requires bid - BrowserGym ID)
    - fill: Fill a text field (requires bid and text)
    - select_option: Select from dropdown (requires bid and value)
    - goto: Navigate to URL (requires url)
    - scroll: Scroll the page (requires direction)
    - send_keys: Send keyboard input (requires text)
    - noop: No operation

    Attributes:
        action_type: Type of action to perform
        bid: BrowserGym element ID (for click, fill, select_option)
        text: Text content (for fill, send_keys)
        value: Value to select (for select_option)
        url: URL to navigate to (for goto)
        direction: Scroll direction - 'up' or 'down' (for scroll)
    """

    action_type: str = Field(
        ...,
        description="Type of action: click, fill, select_option, goto, scroll, send_keys, noop",
    )
    bid: Optional[str] = Field(default=None, description="BrowserGym element ID")
    text: Optional[str] = Field(
        default=None, description="Text content for fill or send_keys"
    )
    value: Optional[str] = Field(default=None, description="Value for select_option")
    url: Optional[str] = Field(default=None, description="URL for goto action")
    direction: Optional[str] = Field(
        default=None, description="Scroll direction: 'up' or 'down'"
    )


class OpenAppObservation(Observation):
    """
    Observation from the OpenApp environment.

    Provides comprehensive state information about the web apps and browser state.

    Attributes:
        html: Current page HTML content
        url: Current page URL
        open_pages_urls: List of all open page URLs
        active_page_index: Index of currently active page
        screenshot: Base64-encoded screenshot (optional)
        axtree_txt: Accessibility tree as text (for element interaction)
        app_state: Current state of all apps (calendar, todo, messenger, map)
        task_info: Information about the current task (if any)
        last_action_error: Error message from last action (if failed)
    """

    html: str = Field(default="", description="Current page HTML content")
    url: str = Field(default="", description="Current page URL")
    open_pages_urls: List[str] = Field(
        default_factory=list, description="List of all open page URLs"
    )
    active_page_index: int = Field(
        default=0, ge=0, description="Index of currently active page"
    )
    screenshot: Optional[str] = Field(
        default=None, description="Base64-encoded screenshot"
    )
    axtree_txt: str = Field(default="", description="Accessibility tree as text")
    app_state: Dict[str, Any] = Field(
        default_factory=dict, description="State of all apps"
    )
    task_info: Optional[Dict[str, Any]] = Field(
        default=None, description="Current task information"
    )
    last_action_error: Optional[str] = Field(
        default=None, description="Error from last action"
    )
