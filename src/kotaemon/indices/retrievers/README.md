# External retrievers

Optional web-search retriever integrations.

## Important files

- `jina_web_search.py` - Jina web search retriever.
- `tavily_web_search.py` - Tavily web search retriever.

## How it connects

The Chat page can import a configured web-search backend from `KH_WEB_SEARCH_BACKEND`. Guest submission policy strips web search commands for the reserved guest account.

## Before changing

- Do not expose web search to guests without updating product policy and tests.
- Keep API keys in environment/settings, not source.

## Verification

```bash
pytest -q tests/test_guest_search_scope.py
python -m compileall src/kotaemon/indices/retrievers
```
