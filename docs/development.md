# Development

## Setup

Use Python 3.11+ and install from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements_gerageragera39.txt
pip install -e .
```

Windows PowerShell activation:

```powershell
.venv\Scripts\Activate.ps1
```

## Run

```bash
python app.py
```

## Tests and smoke checks

```bash
pytest tests
python -m compileall src
```

Focused guest-scope tests live in `tests/test_guest_search_scope.py`.

## CI

GitHub Actions should run from the repository root on Python 3.11+ and should not changing into an upstream `libs/*` path. Dependency installation is intentionally pip-based unless a valid lockfile workflow is refreshed for this flat layout.

## Repository conventions

- Keep user-facing branding as **KURAGa**.
- Keep internal imports as `kotaemon` and `ktem`.
- Do not commit `ktem_app_data/`, `.env`, uploaded files, vector stores, or SQLite DBs.
- Use `CLEANUP_NOTES.md` for uncertain obsolete files instead of deleting them blindly.
- Prefer small, reversible patches and focused tests.
