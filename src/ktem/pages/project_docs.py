from pathlib import Path

import gradio as gr
from theflow.settings import settings


class ProjectDocsPage:
    """Render curated local Markdown documentation inside the app."""

    DOC_FILENAMES = (
        "in_app_guest_docs.md",
        "project_overview.md",
        "guest_guide.md",
        "attribution.md",
    )

    def __init__(self, app, doc_dir: str | Path = settings.KH_DOC_DIR):
        self._app = app
        self.doc_dir = Path(doc_dir).resolve()
        self.on_building_ui()

    def _read_doc(self, filename: str) -> str:
        """Read only allowlisted docs from the configured docs directory."""
        if filename not in self.DOC_FILENAMES:
            raise ValueError(f"Unsupported documentation file: {filename}")

        doc_path = (self.doc_dir / filename).resolve()
        if self.doc_dir not in doc_path.parents and doc_path != self.doc_dir:
            raise ValueError(f"Documentation path escapes docs directory: {filename}")

        if not doc_path.exists():
            return f"## Missing documentation\n\n`{filename}` was not found."

        return doc_path.read_text(encoding="utf-8")

    def on_building_ui(self):
        with gr.Column(elem_classes=["project-docs-page"]):
            gr.Markdown(
                "\n\n---\n\n".join(self._read_doc(name) for name in self.DOC_FILENAMES)
            )
