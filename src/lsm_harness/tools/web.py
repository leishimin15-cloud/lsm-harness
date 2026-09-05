"""Web tools: search and fetch.

- ``web_search`` uses DuckDuckGo instant answer API (no key needed).
- ``web_fetch`` fetches a URL and extracts readable text.
"""

from __future__ import annotations

import html
import re
import urllib.parse
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

from lsm_harness.coding_agent.tools import ToolDefinition
from lsm_harness.agent.tools import ToolResult


# ── HTML → text ───────────────────────────────────────────────────

_SCRIPT_RE = re.compile(r"<script[^>]*>.*?</script>", re.DOTALL | re.IGNORECASE)
_STYLE_RE = re.compile(r"<style[^>]*>.*?</style>", re.DOTALL | re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")
_ENTITY_RE = re.compile(r"&[a-zA-Z]+;|&#\d+;|&#x[0-9a-fA-F]+;")
_WHITESPACE_RE = re.compile(r"\n{3,}")


def _html_to_text(raw: str) -> str:
    """Strip HTML tags and extract readable text."""
    text = _SCRIPT_RE.sub("", raw)
    text = _STYLE_RE.sub("", text)
    text = _TAG_RE.sub("\n", text)
    text = _ENTITY_RE.sub(lambda m: html.unescape(m.group()), text)
    # Collapse whitespace
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    text = "\n".join(lines)
    text = _WHITESPACE_RE.sub("\n\n", text)
    return text.strip()


# ── web_search ────────────────────────────────────────────────────


def _web_search(query: str, max_results: int = 5, timeout: int = 10) -> str | ToolResult:
    """Search DuckDuckGo and return results.

    Uses the DuckDuckGo Instant Answer API. Falls back to HTML scraping
    if the API returns no results.
    """
    max_results = max(1, min(int(max_results), 10))
    timeout = max(3, min(int(timeout), 30))

    # First try the instant answer API
    try:
        api_url = "https://api.duckduckgo.com/"
        params = urllib.parse.urlencode({
            "q": query,
            "format": "json",
            "no_html": "1",
            "skip_disambig": "1",
        })
        req = Request(
            f"{api_url}?{params}",
            headers={"User-Agent": "lsm-harness/1.0"},
        )
        with urlopen(req, timeout=timeout) as resp:
            import json
            data = json.loads(resp.read().decode("utf-8"))

        parts: list[str] = []

        # Abstract (instant answer)
        abstract = (data.get("AbstractText") or "").strip()
        if abstract:
            source = data.get("AbstractSource") or data.get("AbstractURL") or ""
            parts.append(f"📌 {abstract}")
            if source:
                parts.append(f"   Source: {source}")
            parts.append("")

        # Related topics
        related = data.get("RelatedTopics", [])
        results_count = 0
        for topic in related:
            if results_count >= max_results:
                break
            text = (topic.get("Text") or "").strip()
            url = (topic.get("FirstURL") or "").strip()
            if text and url:
                # Remove HTML from the text field
                text = _html_to_text(text)
                parts.append(f"🔗 {text}")
                parts.append(f"   {url}")
                results_count += 1
                parts.append("")

        if parts:
            return f"Search results for: {query}\n\n" + "\n".join(parts)
        else:
            return f"No results found for: {query}"

    except (URLError, OSError, ValueError) as exc:
        return ToolResult(
            output=f"Search failed: {type(exc).__name__}: {exc}",
            is_error=True,
        )


# ── web_fetch ─────────────────────────────────────────────────────


def _web_fetch(url: str, timeout: int = 15, max_chars: int = 8000) -> str | ToolResult:
    """Fetch a URL and extract readable text content.

    Args:
        url: The URL to fetch (must be http/https).
        timeout: Request timeout in seconds (default 15, max 30).
        max_chars: Maximum characters to return (default 8000, max 30000).
    """
    timeout = max(3, min(int(timeout), 30))
    max_chars = max(500, min(int(max_chars), 30000))

    # Validate URL
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return ToolResult(
            output=f"Error: unsupported URL scheme '{parsed.scheme}'. Only http/https allowed.",
            is_error=True,
        )

    # Basic SSRF prevention
    hostname = (parsed.hostname or "").lower()
    if hostname in ("127.0.0.1", "localhost", "0.0.0.0", "::1"):
        return ToolResult(
            output="Error: fetching localhost is not allowed.",
            is_error=True,
        )
    if hostname.startswith("10.") or hostname.startswith("192.168.") or hostname.startswith("172.16."):
        return ToolResult(
            output="Error: fetching private IP ranges is not allowed.",
            is_error=True,
        )

    try:
        req = Request(
            url,
            headers={
                "User-Agent": "lsm-harness/1.0 (personal agent)",
                "Accept": "text/html,text/plain,*/*",
            },
        )
        with urlopen(req, timeout=timeout) as resp:
            # Check content type
            content_type = resp.headers.get("Content-Type", "")
            raw = resp.read()

            if "text/html" in content_type:
                text = _html_to_text(raw.decode("utf-8", errors="replace"))
            elif "text/" in content_type:
                text = raw.decode("utf-8", errors="replace")
            elif "application/json" in content_type:
                text = raw.decode("utf-8", errors="replace")
            else:
                return ToolResult(
                    output=(
                        f"Fetched {len(raw)} bytes of {content_type}. "
                        f"Cannot render as text — use for binary data only."
                    ),
                    is_error=True,
                )

        if len(text) > max_chars:
            text = text[:max_chars] + f"\n\n... (truncated, {len(text) - max_chars} more chars)"

        header = f"Fetched: {url}  ({len(text)} chars)\n\n"
        return header + text

    except HTTPError as exc:
        return ToolResult(
            output=f"HTTP error: {exc.code} {exc.reason} for {url}",
            is_error=True,
        )
    except URLError as exc:
        return ToolResult(
            output=f"Failed to fetch {url}: {exc.reason}",
            is_error=True,
        )
    except Exception as exc:
        return ToolResult(
            output=f"Fetch error: {type(exc).__name__}: {exc}",
            is_error=True,
        )


# ── tool factory ──────────────────────────────────────────────────


def make_tools(default_max_chars: int = 8000) -> list[ToolDefinition]:
    """Build web search and fetch tools."""

    return [
        ToolDefinition(
            name="web_search",
            label="搜索网页",
            description=(
                "在 DuckDuckGo 上搜索网页。返回标题、摘要和链接。"
                "不需要 API key。用于查找最新信息、文档和资料。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索查询"},
                    "max_results": {
                        "type": "integer",
                        "description": "最大结果数，默认 5，上限 10",
                    },
                },
                "required": ["query"],
            },
            execute=lambda query, max_results=5: _web_search(query, max_results),
            effect="read",
            execution_mode="parallel",
            timeout=15.0,
        ),
        ToolDefinition(
            name="web_fetch",
            label="读取网页",
            description=(
                "抓取一个 URL 的内容并提取为纯文本。"
                "用于阅读文档、API 响应、文章等。"
                "不会加载 JavaScript 或执行脚本。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "完整 URL（http 或 https）"},
                    "max_chars": {
                        "type": "integer",
                        "description": "返回的最大字符数，默认 8000，上限 30000",
                    },
                },
                "required": ["url"],
            },
            execute=lambda url, max_chars=default_max_chars: _web_fetch(url, max_chars=max_chars),
            effect="read",
            execution_mode="parallel",
            timeout=30.0,
        ),
    ]
