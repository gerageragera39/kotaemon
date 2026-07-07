# Loaders

File, web, and PDF readers that turn source files into `Document` objects before splitting/indexing.

## Important files

- `pdf_loader.py`, `docx_loader.py`, `excel_loader.py`, `html_loader.py`, `txt_loader.py`, `web_loader.py` - common source readers.
- `docling_loader.py` - Docling PDF reader path.
- `docling_structured_pdf_loader.py` - structured Docling element reader used by university PDF mode.
- `unstructured_loader.py`, `ocr_loader.py`, `azureai_document_intelligence_loader.py`, `adobe_loader.py`, `mathpix_loader.py` - optional/advanced extraction backends.
- `composite_loader.py`, `base.py`, `utils/` - loader composition and support code.

## How it connects

The file index selects loaders in `src/ktem/index/file/pipelines.py`. University PDF mode routes PDFs to `DoclingStructuredPDFReader` so `UniversityPDFChunker` receives ordered elements with metadata rather than plain token chunks.

## Before changing

- Optional extraction backends may require external credentials or system packages; keep imports lazy where possible.
- Preserve metadata fields used downstream (`file_name`, `file_path`, `page_label`, ordering, table markers).
- Do not make expensive OCR/structured parsing the default for every document type without updating performance expectations.

## Verification

```bash
pytest -q tests/test_university_pdf_pipeline.py
python -m compileall src/kotaemon/loaders
```
