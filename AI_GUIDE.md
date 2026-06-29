# AI guide for KURAGa

Short instructions for Codex, Cursor, and other AI agents. Read `PROJECT_CONTEXT.md` first, then inspect only the files needed for the task.

## What this is

**KURAGa** — KU Retrieval-Augmented Guide Assistant — is a KU Digital Projects university-document RAG chatbot. It started from Cinnamon/kotaemon, but the current repository is project-specific and user-facing text should describe KURAGa, not a generic Kotaemon fork.

Internal import packages are still **`kotaemon`** and **`ktem`**. Do not rename them unless the task explicitly includes a safe package-migration plan.

## Read-first map

| Task | Files to inspect first |
| --- | --- |
| App launch/config | `app.py`, `flowsettings.py`, `.env.example`, `pyproject.toml` |
| Chat behavior | `src/ktem/pages/chat/__init__.py`, `src/ktem/pages/chat/chat_panel.py`, `src/ktem/pages/chat/control.py` |
| Guest access/scope | `src/ktem/pages/login.py`, `src/ktem/main.py`, `src/ktem/index/file/ui.py`, `src/ktem/utils/guest_scope.py`, `tests/test_guest_search_scope.py` |
| In-app docs tab | `src/ktem/pages/project_docs.py`, `src/ktem/pages/help.py`, `docs/in_app_guest_docs.md`, `docs/project_overview.md` |
| RAG pipeline | `src/ktem/reasoning/simple.py`, `src/ktem/index/file/pipelines.py`, `src/kotaemon/indices/qa/citation_qa.py`, `src/kotaemon/indices/qa/format_context.py` |
| Indexing/PDF handling | `src/ktem/index/file/ui.py`, `src/ktem/index/file/ingestion_v2.py`, `src/kotaemon/indices/ingests/files.py`, `src/kotaemon/indices/splitters/university_pdf.py`, `src/kotaemon/loaders/docling_loader.py` |
| Evaluation | `src/ktem/evaluation/ragas_eval.py`, `scripts/run_rag_eval.py`, `scripts/eval_university_retrieval.py`, `scripts/evaluate_llm_judge.py`, `rag_eval_dataset*.json`, `dataset/` |
| Docker/CI | `Dockerfile`, `docker-compose.yml`, `launch.sh`, `Makefile`, `.github/workflows/` |
| Documentation | `README.md`, `docs/`, `mkdocs.yml`, `NOTICE.md` |

## Do not edit/delete without explicit reason

- `.env` or secrets.
- `ktem_app_data/` runtime data.
- SQLite DBs, vector stores, `user_data`, uploaded files, markdown/chunk caches.
- Course/evaluation datasets under `dataset/` and `rag_eval_dataset*.json` unless the task is specifically about evaluation data.
- User uploads or generated stores, even if they look obsolete.

Generated caches such as `__pycache__/`, `.pytest_cache/`, and `*.pyc` can be removed when they are local/generated.

## Current assumptions to preserve

- Python **3.11+**.
- Install from repo root with `pip install -r requirements_gerageragera39.txt` and `pip install -e .`.
- No the upstream `libs/*` layout layout in this fork.
- User-facing name: **KURAGa**.
- Full name: **KU Retrieval-Augmented Guide Assistant**.
- Short tagline: **A university-document RAG chatbot for KU Digital Projects**.
- KURAGa is not an official KU service.

## Common commands

```bash
python -m venv .venv
pip install -r requirements_gerageragera39.txt
pip install -e .
python app.py

pytest tests
python -m compileall src
rg "upstream-only path or branding pattern"
```
