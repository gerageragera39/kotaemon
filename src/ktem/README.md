# `ktem` application layer

`ktem` is the Gradio application package inherited from Kotaemon and adapted for KURAGa. It owns UI pages, app lifecycle, settings state, index/resource managers, user/guest behavior, and evaluation entry points.

## Important files and folders

- `app.py` - `BaseApp` lifecycle: load assets, register extensions/reasonings, initialize indices, create Gradio blocks.
- `main.py` - KURAGa tab layout and guest/admin tab visibility policy.
- `settings.py` - typed settings groups used to flatten UI/runtime settings.
- `pages/` - Gradio pages for login, chat, docs, files, evaluation, resources, settings, setup, and help.
- `index/` - index manager and file-index implementation.
- `reasoning/` - answer-generation pipelines that call retrievers and QA components.
- `evaluation/` - RAG/RAGAS evaluation implementation.
- `utils/` - guest-scope, feedback repair, export, rendering, and support helpers.
- `db/` - SQLModel tables and engine setup.
- `assets/` - CSS, JS, images, in-app Markdown, and PDF viewer integration.
- `embeddings/`, `llms/`, `rerankings/`, `mcp/` - UI/database managers for configurable resources.

## How it connects

`app.py` at the repository root creates `ktem.main.App`. That app reads `flowsettings.py`, initializes `IndexManager`, and registers the file index (`ktem.index.file.FileIndex`). Chat page events call reasoning pipelines, which call file retrieval pipelines, which call `kotaemon` retrievers and QA/citation code.

## KURAGa-specific behavior

- Guest login is implemented in `pages/login.py`; a reserved `guest` user is created with a random password and accessed only through the guest button.
- Guest tab visibility is restricted in `main.py` to Chat and Project Documentation.
- Guest submissions are sanitized in `utils/guest_scope.py` so guests search all admin-indexed documents and cannot submit URLs/web-search commands through the guest flow.
- Evaluation exports and feedback repair behavior are KURAGa additions around local university RAG quality workflows.

## Before changing

- UI state often travels through flattened setting keys such as `index.options.files.num_retrieval`; update tests when changing these contracts.
- Keep guest restrictions enforced in testable helper functions, not only in Gradio visibility.
- Avoid putting model secrets or runtime data in source; resource specs should come from settings, DB rows, or `.env`.

## Verification

```bash
pytest -q tests/test_guest_search_scope.py tests/test_feedback_repair.py tests/test_rag_evaluation_modes.py
python -m compileall src/ktem
```
