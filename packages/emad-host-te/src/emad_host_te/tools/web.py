"""Web tools — search and read web pages.

Uses duckduckgo-search for search and crawl4ai for page reading.
Always available to the Imperator.
"""

import asyncio
import ipaddress
import logging
from urllib.parse import urlparse

from langchain_core.tools import tool

_log = logging.getLogger("emad_host.tools.web")


@tool
async def web_search(query: str, max_results: int = 5) -> str:
    """Search the web using DuckDuckGo.

    Use this to find current information, documentation, or answers
    that aren't in the conversation history or domain knowledge.

    Args:
        query: Search query.
        max_results: Maximum results to return (default 5, max 20).
    """
    try:
        from duckduckgo_search import DDGS

        max_results = min(max_results, 20)
        loop = asyncio.get_running_loop()
        ddgs = DDGS()
        results = await loop.run_in_executor(
            None, lambda: ddgs.text(query, max_results=max_results)
        )
        if not results:
            return "No search results found."
        lines = [f"Found {len(results)} results:"]
        for r in results:
            lines.append(f"- **{r.get('title', 'Untitled')}**")
            lines.append(f"  {r.get('href', '')}")
            lines.append(f"  {r.get('body', '')[:200]}")
        return "\n".join(lines)
    except ImportError:
        return "Web search unavailable — duckduckgo-search not installed."
    except (RuntimeError, OSError, ValueError) as exc:
        return f"Search error: {exc}"


@tool
async def web_read(url: str, max_chars: int = 10000) -> str:
    """Read and extract content from a web page.

    Use this to read documentation, articles, or any web content.
    Returns cleaned text content, not raw HTML.

    Args:
        url: URL to read.
        max_chars: Maximum characters to return (default 10000).
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return f"Invalid URL scheme: {parsed.scheme}. Only http and https are supported."
    # Block private/internal IPs
    hostname = parsed.hostname or ""
    if hostname == "localhost":
        return "Access denied: cannot access internal/private addresses."
    try:
        addr = ipaddress.ip_address(hostname)
        if addr.is_private or addr.is_loopback or addr.is_link_local or addr.is_reserved:
            return "Access denied: cannot access internal/private addresses."
    except ValueError:
        pass  # hostname is a DNS name, not an IP — allow (DNS resolution happens later)
    # Also block metadata endpoint by hostname
    if hostname == "169.254.169.254":
        return "Access denied: cannot access metadata endpoint."

    try:
        import httpx
        import ssl

        async with httpx.AsyncClient(
            follow_redirects=True, verify=ssl.create_default_context()
        ) as client:
            resp = await client.get(url, timeout=30)
            resp.raise_for_status()
            content_type = resp.headers.get("content-type", "")

            if "text/html" in content_type:
                # Try crawl4ai for HTML
                try:
                    from crawl4ai import AsyncWebCrawler

                    async with AsyncWebCrawler() as crawler:
                        result = await crawler.arun(url=url)
                        text = result.markdown or result.text or ""
                        if text:
                            return text[:max_chars]
                except (ImportError, OSError, RuntimeError):
                    # crawl4ai may fail if playwright browsers not installed.
                    # Fall through to basic HTML stripping.
                    pass

                # Fallback: basic HTML stripping
                import re

                html = resp.text
                text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL)
                text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL)
                text = re.sub(r"<[^>]+>", " ", text)
                text = re.sub(r"\s+", " ", text).strip()
                return text[:max_chars]
            else:
                # Plain text or other
                return resp.text[:max_chars]
    except ImportError:
        return "Web reading unavailable — httpx not installed."
    except (httpx.HTTPError, RuntimeError, OSError, ValueError) as exc:
        return f"Error reading {url}: {exc}"


def get_tools() -> list:
    """Return all web tools."""
    return [web_search, web_read]
