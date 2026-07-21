# =========================================================
# WEB SEARCH SERVICE (DuckDuckGo)
# =========================================================
# Small wrapper around the DuckDuckGo search library.
# Kept in its own file so the LangGraph backend doesn't need
# to know *how* search works, only that it can call search().

from ddgs import DDGS


def search(query: str, max_results: int = 5) -> str:
    """
    Run a DuckDuckGo search and return a plain-text summary
    of the top results (title + snippet + link) that can be
    dropped straight into an LLM prompt as context.

    Returns an empty-results message instead of raising, so a
    flaky network / rate limit never crashes the chat.
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
    except Exception as e:
        return f"(Web search failed: {e})"

    if not results:
        return "(No web results found.)"

    formatted = []
    for i, r in enumerate(results, start=1):
        title = r.get("title", "Untitled")
        snippet = r.get("body", "")
        link = r.get("href", "")
        formatted.append(f"{i}. {title}\n   {snippet}\n   Source: {link}")

    return "\n\n".join(formatted)
