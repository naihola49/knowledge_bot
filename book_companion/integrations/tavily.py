"""Tavily API adapter for search + extract operations"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from tavily import TavilyClient

from book_companion.config import TAVILY_API_KEY_ENV_VAR


class TavilyApiAdapter:
    """Small adapter to normalize Tavily client usage across the app"""

    def __init__(self, client: TavilyClient) -> None:
        self._client = client

    def search(self, *, query: str, max_results: int) -> Mapping[str, Any]:
        return self._client.search(query=query, max_results=max_results)

    def extract(self, *, urls: list[str], extract_depth: str | None = None) -> Mapping[str, Any]:
        if extract_depth:
            return self._client.extract(urls=urls, extract_depth=extract_depth)
        return self._client.extract(urls=urls)


def get_tavily_client(api_key_env_var: str = TAVILY_API_KEY_ENV_VAR) -> TavilyApiAdapter:
    """Build an authenticated Tavily adapter from environment configuration"""
    api_key = os.getenv(api_key_env_var)
    if not api_key:
        raise RuntimeError(f"Missing required environment variable: {api_key_env_var}")
    return TavilyApiAdapter(TavilyClient(api_key=api_key))

