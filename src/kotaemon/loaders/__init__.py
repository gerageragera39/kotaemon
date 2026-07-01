"""Document readers with optional formats loaded on demand."""

from importlib import import_module
from typing import Any

_EXPORTS = {
    "AutoReader": (".base", "AutoReader"),
    "BaseReader": (".base", "BaseReader"),
    "AzureAIDocumentIntelligenceLoader": (
        ".azureai_document_intelligence_loader",
        "AzureAIDocumentIntelligenceLoader",
    ),
    "PandasExcelReader": (".excel_loader", "PandasExcelReader"),
    "ExcelReader": (".excel_loader", "ExcelReader"),
    "MathpixPDFReader": (".mathpix_loader", "MathpixPDFReader"),
    "ImageReader": (".ocr_loader", "ImageReader"),
    "OCRReader": (".ocr_loader", "OCRReader"),
    "DirectoryReader": (".composite_loader", "DirectoryReader"),
    "UnstructuredReader": (".unstructured_loader", "UnstructuredReader"),
    "DocxReader": (".docx_loader", "DocxReader"),
    "HtmlReader": (".html_loader", "HtmlReader"),
    "MhtmlReader": (".html_loader", "MhtmlReader"),
    "AdobeReader": (".adobe_loader", "AdobeReader"),
    "TxtReader": (".txt_loader", "TxtReader"),
    "PDFThumbnailReader": (".pdf_loader", "PDFThumbnailReader"),
    "WebReader": (".web_loader", "WebReader"),
    "DoclingReader": (".docling_loader", "DoclingReader"),
    "DoclingStructuredPDFReader": (
        ".docling_structured_pdf_loader",
        "DoclingStructuredPDFReader",
    ),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = _EXPORTS[name]
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value
