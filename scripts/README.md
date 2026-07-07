# Scripts

Developer and operator scripts for indexing, evaluating, running local helpers, and managing Docker.

## Important files

- `index_university_documents.py` - deterministic preflight/indexing helper for `dataset/documents/*.pdf` with `pdf_mode=university`.
- `debug_university_chunks.py` - inspect parent/child chunks and metadata emitted by the university PDF chunker.
- `eval_university_retrieval.py` - retrieval-only regression over university test files.
- `run_rag_eval.py` - command-line wrapper for the KURAGa evaluation pipeline.
- `evaluate_llm_judge.py` - LLM-as-judge utility for answer evaluation.
- `university_rag_smoke.py` - in-memory deterministic smoke check for hybrid retrieval without external services.
- `serve_local.py` and `server_llamacpp_*` - local llama.cpp/OpenAI-compatible serving helpers.
- `docker-up.sh`, `docker-up.ps1`, `download_pdfjs.sh` - Docker and PDF.js setup helpers.
- `migrate/` - one-off migration utilities.

## How it connects

Scripts import the installed local packages from [`../src`](../src/README.md). Run them from the repository root after installing `pip install -e .` so imports resolve consistently. Evaluation scripts use [`../src/ktem/evaluation`](../src/ktem/evaluation/README.md); indexing/debug scripts exercise [`../src/kotaemon/indices/splitters/university_pdf.py`](../src/kotaemon/indices/splitters/university_pdf.py) and [`../src/ktem/index/file`](../src/ktem/index/file/README.md).

## Before changing

- Keep script defaults aligned with `.env.example`, `flowsettings.py`, and documented dataset paths.
- Avoid making scripts depend on committed runtime state under `ktem_app_data/`.
- Prefer explicit CLI flags/env vars over hard-coded local machine paths.
- If a script writes outputs, write under `ktem_app_data/evaluations/`, `dataset/.cache/`, or a user-provided path, not source folders.

## Verification

```bash
python scripts/university_rag_smoke.py
python scripts/debug_university_chunks.py dataset/documents/<file>.pdf
pytest -q tests/test_university_pdf_pipeline.py tests/test_rag_evaluation_modes.py
```
