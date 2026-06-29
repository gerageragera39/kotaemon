# Project context: KURAGa

## Overview

**KURAGa** (KU Retrieval-Augmented Guide Assistant) is a university-document RAG chatbot built for the KU / WFI Digital Projects course. It started from [Cinnamon/kotaemon](https://github.com/Cinnamon/kotaemon) and keeps many upstream components, but the current product focus is no longer a generic document-chat demo: it is an assistant for querying curated university and programme documents.

KURAGa is not an official KU service. It should always present answers as support for navigating documents, not as authoritative academic/legal advice.

## Layout and naming

The repository uses a flat `src/` layout:

- `src/ktem/` — Gradio UI/application layer.
- `src/kotaemon/` — reusable RAG components inherited/adapted from Kotaemon.
- `flowsettings.py` — runtime configuration and default model/index definitions.
- `ktem_app_data/` — runtime DB, uploads, vector stores, caches; never commit or edit blindly.

Do **not** rename Python import packages `kotaemon` or `ktem` unless a full migration proves it is safe. User-facing branding should say **KURAGa**.

## Current UI

Main Gradio tabs/pages:

- **Welcome** — login plus restricted guest access.
- **Chat** — conversation UI, retrieval settings, evidence/citation panel.
- **Project Documentation** — guest-visible local Markdown documentation for KURAGa.
- **File Collection / Files** — admin document upload, indexing, file/group management.
- **Evaluation** — curated RAG evaluation helpers.
- **Resources** — LLM, embedding, and reranking model configuration; admin-only when user management is enabled.
- **Settings** — application/reasoning/index settings.
- **Help** — local docs/help content for authenticated non-guest users.

Guest users see only **Chat** and **Project Documentation** plus logout. They are forced to chat against all admin-indexed documents.

## Guest restrictions

Relevant files:

- `src/ktem/pages/login.py` — reserved `guest` account creation and guest button.
- `src/ktem/main.py` — tab visibility after sign-in/sign-out.
- `src/ktem/pages/chat/__init__.py` — chat events, quick upload visibility, command handling.
- `src/ktem/index/file/ui.py` — file selector state, guest Search All resolution, file upload UI.
- `src/ktem/utils/guest_scope.py` — pure helper functions for default and guest selection scope.
- `tests/test_guest_search_scope.py` — focused guest selection tests.

Expected behavior: guest cannot upload files, select a single document, disable document search, use web-only search, or access admin/resource/settings/evaluation/file-management tabs.

## RAG flow

Simplified runtime flow:

```text
ChatPage.submit_msg
  -> ChatPage.chat_fn
  -> ChatPage.create_pipeline
  -> FileIndex.get_retriever_pipelines
  -> ktem.index.file.pipelines.DocumentRetrievalPipeline
  -> kotaemon.indices.qa citation QA pipeline
  -> answer + citations/evidence panel
```

Important files:

- `src/ktem/pages/chat/__init__.py`
- `src/ktem/reasoning/simple.py`
- `src/ktem/index/file/pipelines.py`
- `src/kotaemon/indices/vectorindex.py`
- `src/kotaemon/indices/qa/citation_qa.py`
- `src/kotaemon/indices/qa/format_context.py`
- `src/kotaemon/rerankings/tei_fast_rerank.py`

## Indexing flow

Simplified indexing flow:

```text
Admin upload / script input
  -> FileIndexPage.index_fn
  -> FileIndexIndexing pipeline
  -> loaders/docling/unstructured/PDF helpers
  -> university-aware splitting/chunking
  -> vectorstore + docstore + SQL Source/Index rows
```

Important files:

- `src/ktem/index/file/ui.py`
- `src/ktem/index/file/ingestion_v2.py`
- `src/ktem/index/file/pipelines.py`
- `src/kotaemon/indices/ingests/files.py`
- `src/kotaemon/indices/splitters/university_pdf.py`
- `src/kotaemon/loaders/docling_loader.py`
- `src/kotaemon/loaders/docling_structured_pdf_loader.py`
- `scripts/index_university_documents.py`
- `scripts/debug_university_chunks.py`

## Model defaults

`flowsettings.py` defaults to local Ollama/OpenAI-compatible endpoints:

- LLM: `LOCAL_MODEL` (default `qwen3:8b` unless overridden in `.env`).
- Embeddings: `LOCAL_MODEL_EMBEDDINGS` (default `nomic-embed-text`).
- Ollama base URL: `KH_OLLAMA_URL` (default `http://localhost:11434/v1/`).
- Optional TEI reranker: `BAAI/bge-reranker-v2-m3` at `http://localhost:8080`.

`LOCAL_MODEL` means an Ollama model name in `flowsettings.py`, but `scripts/serve_local.py` treats it as a `.gguf` file path. Check the workflow before changing it.

## Evaluation

Useful files:

- `rag_eval_dataset.json`
- `rag_eval_dataset_kazi.json`
- `dataset/testing_files/`
- `scripts/run_rag_eval.py`
- `scripts/eval_university_retrieval.py`
- `scripts/evaluate_llm_judge.py`
- `scripts/university_rag_smoke.py`
- `src/ktem/evaluation/ragas_eval.py`
- `tests/test_university_pdf_pipeline.py`
- `tests/test_vectorindex_hybrid_regression.py`

## Cleanup and legacy warnings

- Upstream OS installer scripts were removed because they reflected the original release-installer model and did not match this fork.
- Do not use the upstream `libs/*` layout or the upstream `libs/*` layout paths in new docs, CI, Docker, or scripts; this fork installs from repo root.
- Do not delete `ktem_app_data/`, `.env`, user uploads, vector stores, SQLite DBs, or dataset/evaluation files.
- Record uncertain cleanup candidates in `CLEANUP_NOTES.md` instead of deleting them blindly.

## What AI agents should inspect first

1. `README.md`, `PROJECT_CONTEXT.md`, `AI_GUIDE.md`.
2. `flowsettings.py`, `.env.example`, `pyproject.toml`.
3. For guest behavior: `src/ktem/main.py`, `src/ktem/pages/login.py`, `src/ktem/pages/chat/__init__.py`, `src/ktem/index/file/ui.py`, `src/ktem/utils/guest_scope.py`.
4. For indexing/retrieval: `src/ktem/index/file/pipelines.py`, `src/kotaemon/indices/vectorindex.py`, `src/kotaemon/indices/qa/citation_qa.py`.
5. For docs UI: `src/ktem/pages/project_docs.py`, `src/ktem/pages/help.py`, `docs/in_app_guest_docs.md`.
