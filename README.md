# KURAGa — KU Retrieval-Augmented Guide Assistant

**KURAGa** is a student-built retrieval-augmented generation (RAG) assistant for querying university and programme documents. It was developed for the **KU / WFI Digital Projects** course and adapts the open-source Kotaemon document-chat architecture into a university-document chatbot.

> **Status and disclaimer**
> KURAGa is a course project, not an official KU service. Answers can be incomplete or wrong; always verify important academic, legal, or administrative decisions against official university documents and staff guidance.

## Attribution

This repository is based on [Cinnamon/kotaemon](https://github.com/Cinnamon/kotaemon), licensed under Apache-2.0. Original architecture and components are retained where useful, especially the `kotaemon` RAG library and `ktem` Gradio application layer. This fork contains substantial project-specific changes by the KU Digital Projects team for university-document retrieval, guest access, local model defaults, evaluation, and UI/documentation branding.

See [`NOTICE.md`](NOTICE.md) and [`LICENSE.txt`](LICENSE.txt) for attribution and license details.

## Key features

- **Gradio web UI** for document chat, administration, resources, settings, and evaluation.
- **Guest chat access** through a restricted guest button.
- **Guest users are forced to Search All** admin-indexed documents; they cannot upload files, select one file, disable document search, or access admin tabs.
- **Admin-managed document collections** for university PDFs and other supported files.
- **Local-first model setup** using Ollama/OpenAI-compatible LLM and embedding APIs by default.
- **University PDF/document handling** with custom loaders, splitters, chunk diagnostics, and indexing scripts.
- **Hybrid retrieval and reranking support** where configured through the file retrieval pipeline and TEI reranker defaults.
- **Citations/evidence panel** showing retrieved context and citation links for answers.
- **Evaluation scripts and datasets** for running curated university RAG checks and optional RAGAS/LLM-judge scoring.
- **Feedback/dislike repair workflow** for regenerating recent answers with targeted repair presets.

## Architecture overview

| Path | Purpose |
| --- | --- |
| `src/ktem/` | Gradio UI and application layer: tabs, chat page, login/guest access, resources, settings, evaluation, file index UI. |
| `src/kotaemon/` | Reusable RAG components inherited/adapted from Kotaemon: loaders, splitters, vector/doc stores, retrievers, LLMs, embeddings, rerankers, QA/citations. |
| `flowsettings.py` | Main runtime configuration: app name, model defaults, index definitions, stores, feature flags. |
| `scripts/` | University indexing, retrieval debugging, local model serving, Docker helpers, and evaluation utilities. |
| `dataset/` | Course/evaluation documents and curated question datasets. |
| `ktem_app_data/` | Runtime data: SQLite DB, uploads, vector stores, caches. **Do not commit or delete without a backup.** |

Internal Python package names remain `kotaemon` and `ktem` for compatibility with the upstream architecture. User-facing documentation and UI branding use **KURAGa**.

## Quick start

### Python 3.11+

From the repository root:

```bash
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

# Windows cmd
# .venv\Scripts\activate.bat

# Linux/macOS
# source .venv/bin/activate

pip install -r requirements_gerageragera39.txt
pip install -e .
cp .env.example .env     # Windows: copy .env.example .env
python app.py            # http://localhost:7860
```

Default user-management setup uses `admin` / `admin` unless changed in configuration or the database. Change this immediately after first login.

### Model setup

KURAGa defaults to an Ollama/OpenAI-compatible local setup:

```bash
ollama pull qwen2.5:7b
ollama pull nomic-embed-text
```

Then set `.env` values such as:

```env
KH_APP_NAME=KURAGa
LOCAL_MODEL=qwen2.5:7b
LOCAL_MODEL_EMBEDDINGS=nomic-embed-text
KH_OLLAMA_URL=http://localhost:11434/v1/
```

Cloud OpenAI-compatible providers can also be configured through `.env`, `flowsettings.py`, or the **Resources** tab.

## Docker setup

Docker support is maintained for local deployment with persistent app data in `./ktem_app_data`:

```bash
cp .env.example .env
# Windows: copy .env.example .env

docker compose up -d --build
# http://localhost:7860
```

Optional profiles:

```bash
# Bundled Ollama container
docker compose --profile ollama up -d --build

# Optional TEI reranker
docker compose --profile reranker up -d
```

Useful commands:

| Command | Effect |
| --- | --- |
| `docker compose up -d --build` | Build/start KURAGa and keep data in `./ktem_app_data`. |
| `docker compose down` | Stop containers and keep data. |
| `docker compose down -v` | Stop containers and delete named Docker volumes. |
| `docker compose logs -f kuraga` | Follow app logs after the compose service starts. |

## Guest and admin usage

### Admin flow

1. Log in as an admin.
2. Configure LLM, embedding, and optional reranking models in **Resources** or `flowsettings.py`.
3. Upload and index university/programme documents in the file collection tab.
4. Use **Evaluation** and scripts under `scripts/` to check retrieval and answer quality.

### Guest flow

1. Click **Access as Guest** on the welcome page.
2. Use **Chat** to ask questions over all admin-indexed documents.
3. Read **Project Documentation** in the app for scope, limitations, citations, and attribution.

Guest users cannot upload files, select individual files, turn document search off, or access Resources, Settings, Evaluation, or file-management pages.

## Evaluation

Current evaluation assets include:

- `rag_eval_dataset.json` and `rag_eval_dataset_kazi.json` — curated question sets.
- `dataset/testing_files/` and `dataset/documents/` — course/evaluation document folders.
- `scripts/run_rag_eval.py` — run app RAG over a dataset.
- `scripts/eval_university_retrieval.py` and `scripts/debug_university_chunks.py` — inspect retrieval/chunking.
- `scripts/evaluate_llm_judge.py` — optional LLM-judge scoring.
- `src/ktem/evaluation/ragas_eval.py` — local RAGAS integration used by the Evaluation tab.

Run scripts from the repository root and check each script's options before using it against real course data.

## Development notes

- Run commands from the repository root.
- Use Python **3.11+**.
- Install this flat-layout fork with `pip install -e .`; do **not** use upstream the upstream `libs/*` layout paths.
- Do not commit `.env`, `ktem_app_data/`, vector stores, SQLite DBs, user uploads, cache directories, or generated `__pycache__` files.
- Keep package imports as `kotaemon` and `ktem` unless a full migration plan proves it is safe to rename them.
- For architecture details, read [`PROJECT_CONTEXT.md`](PROJECT_CONTEXT.md) and [`AI_GUIDE.md`](AI_GUIDE.md).

## License

KURAGa keeps the upstream Apache-2.0 license from Cinnamon/kotaemon. See [`LICENSE.txt`](LICENSE.txt) and [`NOTICE.md`](NOTICE.md).
