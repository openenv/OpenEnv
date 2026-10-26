# Copyright 2025 Yuan He. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Inspired by: https://github.com/THUDM/slime/tree/main/examples/search-r1

from __future__ import annotations

import random

import chardet
import requests
from models import WebContent, WebSearchAction, WebSearchObservation


class WebSearchTool:
    """A tool for searching the web using Google Search API (via Serper.dev)."""

    def __init__(
        self,
        api_key: str | None = None,
        top_k: int = 5,
        timeout: int = 60,
        snippet_only: bool = False,
        proxy: str | None = None,
    ):
        self.api_key = api_key
        self.top_k = top_k
        self.timeout = timeout
        self.snippet_only = snippet_only
        self.proxy = proxy

    def execute(self, web_search_action: WebSearchAction) -> WebSearchObservation:
        """
        Execute a web search based on the query.
        """
        query = web_search_action.query.strip()
        api_key = web_search_action.temp_api_key or self.api_key
        try:
            web_contents = self.google_search(
                api_key=api_key,
                query=query,
                top_k=self.top_k,
                timeout=self.timeout,
                snippet_only=self.snippet_only,
            )
            if web_contents:
                return WebSearchObservation(
                    content=self.format_web_contents(web_contents, query),
                    web_contents=web_contents,
                    done=False,
                    metadata={"query": query},
                )
            else:
                return WebSearchObservation(
                    content=f"[ERROR] No search results found for query: {query}",
                    web_contents=[],
                    done=False,
                    metadata={"query": query, "error": "No search results found"},
                )

        except Exception as e:
            import traceback

            tb_str = traceback.format_exc()
            return WebSearchObservation(
                content=f"[ERROR] Search failed due to: {str(e)}\nTraceback:\n{tb_str}",
                web_contents=[],
                done=False,
                metadata={"query": query, "error": str(e), "traceback": tb_str},
            )

    def google_search(
        self,
        api_key: str,
        query: str,
        top_k: int = 5,
        timeout: int = 60,
        snippet_only: bool = False,
    ) -> list[WebContent]:
        """
        Perform a Google search using Serper.dev API.

        Args:
            api_key: Serper.dev API key.
            query: Search query string.
            top_k: Number of results to return.
            timeout: Request timeout in seconds.
            snippet_only: If `True`, return only snippets; if `False`, fetch full webpage content.

        Returns:
            list[dict[str, Any]]: List of search results with titles and content.
        """
        proxies = {"http": self.proxy, "https": self.proxy} if self.proxy else None

        resp = requests.post(
            "https://google.serper.dev/search",
            json={
                "q": query,
                "num": top_k,
                "gl": "us",
                "hl": "en",
            },
            headers={
                "Content-Type": "application/json",
                "X-API-KEY": api_key,
            },
            timeout=timeout,
            proxies=proxies,
        )
        resp.raise_for_status()
        response = resp.json()
        items = response.get("organic", [])

        web_contents = []
        if snippet_only:
            # Quick mode: just use snippets
            for item in items:
                title = item.get("title", "")
                snippet = item.get("snippet", "")
                context = " ".join(self.parse_search_snippet(snippet))

                if title or context:
                    title = title or "No title."
                    context = context or "No snippet available."
                    web_contents.append(
                        WebContent(
                            title=title, content=context, url=item.get("link", "")
                        )
                    )
        else:
            # Deep mode: fetch full page content
            links = [item.get("link", "") for item in items if "link" in item]
            raw_contents = self.fetch_web_contents(links)

            for i, item in enumerate(items):
                title = item.get("title", "")
                snippet = item.get("snippet", "")

                # Extract relevant context from the full page
                context = (
                    self.expand_search_snippet(snippet, raw_contents[i])
                    if i < len(raw_contents) and raw_contents[i]
                    else snippet
                )

                if title or context:
                    title = title or "No title."
                    context = context or "No content available."
                    web_contents.append(
                        WebContent(
                            title=title, content=context, url=item.get("link", "")
                        )
                    )

        return web_contents

    @staticmethod
    def fetch_web_contents(urls: list[str], limit: int = 8) -> list[str]:
        """
        Fetch multiple web contents concurrently with rate limiting.

        Args:
            urls (list[str]): List of URLs to fetch.
            limit (int): Maximum concurrent requests.

        Returns:
            list[str]: List of page contents (empty string for failed requests).
        """

        def _fetch(url: str) -> str:
            if url == "":
                return ""

            user_agents = [
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                "Mozilla/5.0 (compatible; Googlebot/2.1; +https://www.google.com/bot.html)",
            ]
            headers = {"User-Agent": random.choice(user_agents)}

            try:
                response = requests.get(url, headers=headers, timeout=10)
                raw = response.content
                detected = chardet.detect(raw)
                encoding = detected.get("encoding") or "utf-8"
                return raw.decode(encoding, errors="ignore")
            except Exception:
                # Silently fail for individual pages
                return ""

        # Fetch URLs sequentially
        results = []
        for url in urls:
            results.append(_fetch(url))

        return results

    @staticmethod
    def parse_search_snippet(snippet: str) -> list[str]:
        """
        Parse a search snippet into meaningful segments.

        Args:
            snippet: The snippet text with ellipsis separators.

        Returns:
            List of text segments with at least 5 words.
        """
        segments = snippet.split("...")
        return [s.strip() for s in segments if len(s.strip().split()) > 5]

    @staticmethod
    def expand_search_snippet(snippet: str, web_content: str) -> str:
        """
        Finds snippet segments in the web content and expands them to full paragraphs.

        Args:
            snippet (str): The search snippet with key phrases.
            web_content (str): The full web content text.

        Returns:
            str: The expanded full context of the snippet.
        """
        snippets = WebSearchTool.parse_search_snippet(snippet)
        ctx_paras = []

        for s in snippets:
            # Find snippet in document
            pos = web_content.replace("\n", " ").find(s)
            if pos == -1:
                continue

            # Expand to paragraph boundaries
            sta = pos
            while sta > 0 and web_content[sta] != "\n":
                sta -= 1

            end = pos + len(s)
            while end < len(web_content) and web_content[end] != "\n":
                end += 1

            para = web_content[sta:end].strip()
            if para and para not in ctx_paras:
                ctx_paras.append(para)

        return "\n".join(ctx_paras)

    @staticmethod
    def format_web_contents(web_contents: list[WebContent], query: str) -> str:
        """
        Format search results into a readable string.

        Args:
            results (list[dict[str, Any]]): List of search result dictionaries.
            query (str): Original search query.

        Returns:
            str: Formatted string representation of results.
        """
        lines = [f"Search results for: {query}\n"]

        for i, result in enumerate(web_contents, 1):
            lines.append(f"[{i}] {result.title}")
            lines.append(f"    URL: {result.url or 'N/A'}")
            lines.append(
                f"    {result.content[:500]}{'...' if len(result.content) > 500 else ''}"
            )
            lines.append("")

        return "\n".join(lines)
