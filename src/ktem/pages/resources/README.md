# Resources pages

UI pages for managing model/resource records.

## Important files

- `__init__.py` - Resources tab assembly.
- `user.py` - user-management helpers, including admin user creation.

## How it connects

Only admins should see Resources when user management is enabled. Resource managers under `src/ktem/llms`, `embeddings`, `rerankings`, and `mcp` provide the tab contents.

## Before changing

- Preserve guest/admin visibility expectations from `src/ktem/main.py`.
- Do not expose resource management to the reserved guest account.

## Verification

```bash
pytest -q tests/test_guest_search_scope.py
python -m compileall src/ktem/pages/resources
```
