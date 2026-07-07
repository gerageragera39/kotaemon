# Database models

SQLModel database definitions and engine setup for KURAGa runtime metadata.

## Important files

- `engine.py` - database engine creation from settings.
- `base_models.py` - base SQLModel table definitions such as users, conversations, settings, and issue reports.
- `models.py` - concrete table classes selected from `flowsettings.py` settings and `SQLModel.metadata.create_all` when Alembic is disabled.

## How it connects

The app stores users, conversations, settings, issue reports, file/index metadata, and feedback-related data in runtime databases under `ktem_app_data/` by default. Login, chat history, resources, settings, and file index code all depend on these models.

## Before changing

- Treat schema changes as persistent-data changes. Provide migrations or backward-compatible defaults for existing `ktem_app_data/` users.
- Keep the reserved `guest` user non-admin.
- Do not commit local SQLite databases.

## Verification

```bash
pytest -q tests/test_guest_search_scope.py tests/test_feedback_repair.py
python -m compileall src/ktem/db
```
