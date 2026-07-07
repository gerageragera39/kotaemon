# Indexing, retrieval, and QA components

This folder contains lower-level RAG components used by the `ktem` file index and reasoning pipelines.

## Important files and folders

- `vectorindex.py` - vector/text/hybrid retrieval, reciprocal-rank fusion, query expansion, context expansion, reranker integration, and indexing helpers.
- `splitters/` - text splitting, including KURAGa's `university_pdf.py` structural splitter.
- `ingests/` - file ingestion utilities and default file extractors.
- `qa/` - context formatting, citation generation, and answer-with-context pipelines.
- `rankings/` and `retrievers/` - ranking/scoring and external web-search retrievers.
- `extractors/` - document parser helpers.

## How it connects

`src/ktem/index/file/pipelines.py` constructs `VectorRetrieval` and routes PDFs to `UniversityPDFChunker` when university mode is enabled. Reasoning code uses `qa/` to build context and answer with citations/evidence for the chat UI.

## KURAGa-specific behavior

- Hybrid retrieval queries vectorstore and docstore branches, merges query variants, then fuses candidates with RRF and metadata/lexical signals.
- Parent chunks from `UniversityPDFChunker` stay in the docstore while child chunks are embedded; retrieval can expand child hits back to parents or sibling windows.
- Retrieval debug metadata is recorded for evaluation and regression tests.

## Before changing

- Keep vector and docstore scoping compatible with parent/child chunk IDs.
- If changing retrieval defaults or fusion scoring, update tests in `tests/test_vectorindex_hybrid_regression.py`.
- If changing chunk metadata, update `tests/test_university_pdf_pipeline.py` and any docs that describe metadata fields.

## Verification

```bash
pytest -q tests/test_university_pdf_pipeline.py tests/test_vectorindex_hybrid_regression.py tests/test_information_panel_ordering.py
```
