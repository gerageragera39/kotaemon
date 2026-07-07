# `kotaemon` RAG components

This package contains the reusable RAG framework inherited from the upstream Kotaemon/Cinnamon project and adapted where KURAGa needs university-focused behavior.

## Important folders

- `base/` - component and document schema primitives.
- `loaders/` - file/web/PDF readers, including Docling readers used by university PDFs.
- `indices/` - ingestion, splitters, vector retrieval, ranking, and citation QA.
- `storages/` - docstore and vectorstore adapters.
- `embeddings/`, `llms/`, `rerankings/` - provider wrappers used by `ktem` resource managers.
- `agents/`, `chatbot/`, `contribs/`, `parsers/`, `utils/` - upstream reusable features and helpers.

## How it connects

`ktem` composes these components into app-visible pipelines. For example, file indexing uses loaders + splitters + embeddings + stores; chat retrieval uses `indices.vectorindex.VectorRetrieval`; answer generation uses `indices.qa` citation/context pipelines and configured LLM wrappers.

## KURAGa-specific adaptations

- `indices/splitters/university_pdf.py` adds structural chunking for German university PDFs.
- `indices/vectorindex.py` includes hybrid vector/text retrieval, query variants, RRF fusion, parent/sibling context expansion, and retrieval debug metadata used by tests and evaluation.
- Docling structured PDF loading is used when university PDF mode is explicitly enabled.

## Before changing

- Preserve upstream Apache-2.0 attribution and avoid unnecessary rewrites of generic Kotaemon abstractions.
- Keep fork-specific behavior behind explicit settings or metadata where practical.
- Changes here can affect both app UI and scripts; run focused tests and compile checks.

## Verification

```bash
pytest -q tests/test_university_pdf_pipeline.py tests/test_vectorindex_hybrid_regression.py tests/test_information_panel_ordering.py
python -m compileall src/kotaemon
```
