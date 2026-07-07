# Ingestion utilities

Helpers for turning files into indexed documents.

## Important files

- `files.py` - default file extractor registry, readers, and ingestion support used by the file index.

## How it connects

`src/ktem/index/file/pipelines.py` imports default extractors/readers from here before applying KURAGa-specific routing such as university PDF mode.

## Before changing

- Keep default extractor mappings compatible with supported file types in `flowsettings.py`.
- Avoid importing optional backends unless selected.

## Verification

```bash
pytest -q tests/test_university_pdf_pipeline.py
python -m compileall src/kotaemon/indices/ingests
```
