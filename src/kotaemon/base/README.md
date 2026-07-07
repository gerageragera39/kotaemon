# Base component primitives

Core schema and component abstractions inherited from Kotaemon.

## Important files

- `schema.py` - `Document`, retrieved document schemas, and shared data structures.
- `component.py` - base component/pipeline patterns used across loaders, indices, LLMs, embeddings, and rerankers.

## How it connects

Nearly every RAG layer passes `Document` objects with `text`, `doc_id`, `metadata`, and optional scores. KURAGa's university chunker adds metadata to these documents, and retrieval/QA code expects those fields to remain stable.

## Before changing

- Treat schema changes as cross-cutting API changes.
- Keep metadata flexible, but avoid changing required attributes such as `doc_id`, text/content accessors, or score handling without updating tests.

## Verification

```bash
pytest -q tests/test_university_pdf_pipeline.py tests/test_vectorindex_hybrid_regression.py
python -m compileall src/kotaemon/base
```
