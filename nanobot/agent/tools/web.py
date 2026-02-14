"""Web tools: web_search, web_fetch, understand_image."""

import base64
import html
import json
import mimetypes
import os
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

from nanobot.agent.tools.base import Tool

# Shared constants
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_7_2) AppleWebKit/537.36"
MAX_REDIRECTS = 5  # Limit redirects to prevent DoS attacks
DEFAULT_MINIMAX_MCP_HOST = "https://api.minimax.chat"
DEFAULT_MINIMAX_TIMEOUT = 15.0
MAX_IMAGE_SOURCE_BYTES = 10 * 1024 * 1024  # 10MB


def _strip_tags(text: str) -> str:
    """Remove HTML tags and decode entities."""
    text = re.sub(r'<script[\s\S]*?</script>', '', text, flags=re.I)
    text = re.sub(r'<style[\s\S]*?</style>', '', text, flags=re.I)
    text = re.sub(r'<[^>]+>', '', text)
    return html.unescape(text).strip()


def _normalize(text: str) -> str:
    """Normalize whitespace."""
    text = re.sub(r'[ \t]+', ' ', text)
    return re.sub(r'\n{3,}', '\n\n', text).strip()


def _validate_url(url: str) -> tuple[bool, str]:
    """Validate URL: must be http(s) with valid domain."""
    try:
        p = urlparse(url)
        if p.scheme not in ('http', 'https'):
            return False, f"Only http/https allowed, got '{p.scheme or 'none'}'"
        if not p.netloc:
            return False, "Missing domain"
        return True, ""
    except Exception as e:
        return False, str(e)


class _MiniMaxMCPClient:
    """Lightweight client for MiniMax Coding Plan MCP APIs."""

    def __init__(self, api_key: str, api_host: str, timeout: float = DEFAULT_MINIMAX_TIMEOUT):
        self.api_key = api_key.strip()
        self.api_host = api_host.rstrip("/")
        self.timeout = timeout

    @property
    def enabled(self) -> bool:
        return bool(self.api_key and self.api_host)

    async def search(self, query: str) -> dict[str, Any]:
        payload = {"q": query}
        return await self._post_json("/v1/coding_plan/search", payload)

    async def understand_image(self, prompt: str, image_source: str) -> dict[str, Any]:
        image_url = await self._to_data_url(image_source)
        payload = {"prompt": prompt, "image_url": image_url}
        return await self._post_json("/v1/coding_plan/vlm", payload)

    async def _post_json(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.enabled:
            raise RuntimeError("MiniMax MCP is not configured")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "MM-API-Source": "Minimax-MCP",
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(f"{self.api_host}{endpoint}", headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()

        base_resp = data.get("base_resp") if isinstance(data, dict) else None
        if isinstance(base_resp, dict):
            status_code = base_resp.get("status_code")
            if status_code not in (None, 0):
                status_msg = base_resp.get("status_msg", "unknown")
                raise RuntimeError(f"MiniMax MCP API error: {status_code} - {status_msg}")
        return data

    async def _to_data_url(self, image_source: str) -> str:
        source = (image_source or "").strip()
        if not source:
            raise RuntimeError("image_source is required")
        if source.startswith("@"):
            source = source[1:]
        if source.startswith("data:"):
            return source

        if source.startswith(("http://", "https://")):
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(source)
                response.raise_for_status()
                image_bytes = response.content
                if len(image_bytes) > MAX_IMAGE_SOURCE_BYTES:
                    raise RuntimeError("image_source is too large (>10MB)")
                mime = response.headers.get("content-type", "").split(";")[0].strip().lower()
                if mime not in {"image/jpeg", "image/jpg", "image/png", "image/webp"}:
                    # API supports jpg/png/webp. Fall back to jpeg if unknown.
                    mime = "image/jpeg"
            b64 = base64.b64encode(image_bytes).decode("utf-8")
            return f"data:{mime};base64,{b64}"

        path = Path(source).expanduser()
        if not path.exists() or not path.is_file():
            raise RuntimeError(f"image file not found: {source}")
        image_bytes = path.read_bytes()
        if len(image_bytes) > MAX_IMAGE_SOURCE_BYTES:
            raise RuntimeError("image_source is too large (>10MB)")
        mime, _ = mimetypes.guess_type(str(path))
        mime = (mime or "image/jpeg").lower()
        if mime not in {"image/jpeg", "image/jpg", "image/png", "image/webp"}:
            ext = path.suffix.lower()
            if ext == ".png":
                mime = "image/png"
            elif ext == ".webp":
                mime = "image/webp"
            else:
                mime = "image/jpeg"
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        return f"data:{mime};base64,{b64}"


class WebSearchTool(Tool):
    """Search the web using Brave or MiniMax MCP search API."""
    
    name = "web_search"
    description = "Search the web. Returns titles, URLs, and snippets."
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "count": {"type": "integer", "description": "Results (1-10)", "minimum": 1, "maximum": 10}
        },
        "required": ["query"]
    }
    
    def __init__(
        self,
        api_key: str | None = None,
        max_results: int = 5,
        *,
        minimax_enabled: bool = False,
        minimax_api_key: str | None = None,
        minimax_api_host: str | None = None,
        minimax_timeout: float = DEFAULT_MINIMAX_TIMEOUT,
    ):
        self.api_key = api_key or os.environ.get("BRAVE_API_KEY", "")
        self.max_results = max_results
        resolved_minimax_key = minimax_api_key or os.environ.get("MINIMAX_API_KEY", "")
        host = (minimax_api_host or DEFAULT_MINIMAX_MCP_HOST).strip() or DEFAULT_MINIMAX_MCP_HOST
        self._minimax = _MiniMaxMCPClient(
            api_key=resolved_minimax_key,
            api_host=host,
            timeout=minimax_timeout,
        ) if minimax_enabled else None

    def is_configured(self) -> bool:
        return bool(self.api_key) or bool(self._minimax and self._minimax.enabled)
    
    async def execute(self, query: str, count: int | None = None, **kwargs: Any) -> str:
        try:
            n = min(max(count or self.max_results, 1), 10)
            if self._minimax and self._minimax.enabled:
                data = await self._minimax.search(query)
                return self._format_minimax_results(query, data, n)

            if self.api_key:
                async with httpx.AsyncClient() as client:
                    r = await client.get(
                        "https://api.search.brave.com/res/v1/web/search",
                        params={"q": query, "count": n},
                        headers={"Accept": "application/json", "X-Subscription-Token": self.api_key},
                        timeout=10.0
                    )
                    r.raise_for_status()
                return self._format_brave_results(query, r.json(), n)

            return "Error: web search not configured (set BRAVE_API_KEY or MiniMax MCP key)"
        except Exception as e:
            return f"Error: {e}"

    @staticmethod
    def _format_brave_results(query: str, data: dict[str, Any], n: int) -> str:
        results = data.get("web", {}).get("results", [])
        if not results:
            return f"No results for: {query}"

        lines = [f"Results for: {query}\n"]
        for i, item in enumerate(results[:n], 1):
            lines.append(f"{i}. {item.get('title', '')}\n   {item.get('url', '')}")
            if desc := item.get("description"):
                lines.append(f"   {desc}")
        return "\n".join(lines)

    @staticmethod
    def _format_minimax_results(query: str, data: dict[str, Any], n: int) -> str:
        results = data.get("organic", [])
        if not results:
            return f"No results for: {query}"

        lines = [f"Results for: {query}\n"]
        for i, item in enumerate(results[:n], 1):
            title = item.get("title", "")
            url = item.get("link", "")
            snippet = item.get("snippet", "")
            date = item.get("date", "")
            lines.append(f"{i}. {title}\n   {url}")
            if snippet:
                lines.append(f"   {snippet}")
            if date:
                lines.append(f"   Date: {date}")
        return "\n".join(lines)


class UnderstandImageTool(Tool):
    """Understand an image via MiniMax Coding Plan VLM API."""

    name = "understand_image"
    description = "Analyze an image (local path, URL, or data URL) with a prompt."
    parameters = {
        "type": "object",
        "properties": {
            "prompt": {"type": "string", "description": "Instruction for image analysis"},
            "image_source": {"type": "string", "description": "Local file path, http(s) URL, or data URL"},
        },
        "required": ["prompt", "image_source"],
    }

    def __init__(
        self,
        *,
        minimax_api_key: str | None = None,
        minimax_api_host: str | None = None,
        timeout: float = DEFAULT_MINIMAX_TIMEOUT,
    ):
        key = minimax_api_key or os.environ.get("MINIMAX_API_KEY", "")
        host = (minimax_api_host or DEFAULT_MINIMAX_MCP_HOST).strip() or DEFAULT_MINIMAX_MCP_HOST
        self._minimax = _MiniMaxMCPClient(
            api_key=key,
            api_host=host,
            timeout=timeout,
        )

    def is_configured(self) -> bool:
        return self._minimax.enabled

    async def execute(self, prompt: str, image_source: str, **kwargs: Any) -> str:
        if not self._minimax.enabled:
            return "Error: understand_image is not configured (set MiniMax MCP key)"
        try:
            data = await self._minimax.understand_image(prompt=prompt, image_source=image_source)
            content = data.get("content")
            if isinstance(content, str) and content.strip():
                return content
            return json.dumps(data, ensure_ascii=False, indent=2)
        except Exception as e:
            return f"Error: {e}"


class WebFetchTool(Tool):
    """Fetch and extract content from a URL using Readability."""
    
    name = "web_fetch"
    description = "Fetch URL and extract readable content (HTML → markdown/text)."
    parameters = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "URL to fetch"},
            "extractMode": {"type": "string", "enum": ["markdown", "text"], "default": "markdown"},
            "maxChars": {"type": "integer", "minimum": 100}
        },
        "required": ["url"]
    }
    
    def __init__(self, max_chars: int = 50000):
        self.max_chars = max_chars
    
    async def execute(self, url: str, extractMode: str = "markdown", maxChars: int | None = None, **kwargs: Any) -> str:
        from readability import Document

        max_chars = maxChars or self.max_chars

        # Validate URL before fetching
        is_valid, error_msg = _validate_url(url)
        if not is_valid:
            return json.dumps({"error": f"URL validation failed: {error_msg}", "url": url})

        try:
            async with httpx.AsyncClient(
                follow_redirects=True,
                max_redirects=MAX_REDIRECTS,
                timeout=30.0
            ) as client:
                r = await client.get(url, headers={"User-Agent": USER_AGENT})
                r.raise_for_status()
            
            ctype = r.headers.get("content-type", "")
            
            # JSON
            if "application/json" in ctype:
                text, extractor = json.dumps(r.json(), indent=2), "json"
            # HTML
            elif "text/html" in ctype or r.text[:256].lower().startswith(("<!doctype", "<html")):
                doc = Document(r.text)
                content = self._to_markdown(doc.summary()) if extractMode == "markdown" else _strip_tags(doc.summary())
                text = f"# {doc.title()}\n\n{content}" if doc.title() else content
                extractor = "readability"
            else:
                text, extractor = r.text, "raw"
            
            truncated = len(text) > max_chars
            if truncated:
                text = text[:max_chars]
            
            return json.dumps({"url": url, "finalUrl": str(r.url), "status": r.status_code,
                              "extractor": extractor, "truncated": truncated, "length": len(text), "text": text})
        except Exception as e:
            return json.dumps({"error": str(e), "url": url})
    
    def _to_markdown(self, html: str) -> str:
        """Convert HTML to markdown."""
        # Convert links, headings, lists before stripping tags
        text = re.sub(r'<a\s+[^>]*href=["\']([^"\']+)["\'][^>]*>([\s\S]*?)</a>',
                      lambda m: f'[{_strip_tags(m[2])}]({m[1]})', html, flags=re.I)
        text = re.sub(r'<h([1-6])[^>]*>([\s\S]*?)</h\1>',
                      lambda m: f'\n{"#" * int(m[1])} {_strip_tags(m[2])}\n', text, flags=re.I)
        text = re.sub(r'<li[^>]*>([\s\S]*?)</li>', lambda m: f'\n- {_strip_tags(m[1])}', text, flags=re.I)
        text = re.sub(r'</(p|div|section|article)>', '\n\n', text, flags=re.I)
        text = re.sub(r'<(br|hr)\s*/?>', '\n', text, flags=re.I)
        return _normalize(_strip_tags(text))
