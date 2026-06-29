from pathlib import Path

import gradio as gr
from theflow.settings import settings

KH_DEMO_MODE = getattr(settings, "KH_DEMO_MODE", False)


class HelpPage:
    def __init__(
        self,
        app,
        doc_dir: str = settings.KH_DOC_DIR,
        app_version: str | None = settings.KH_APP_VERSION,
    ):
        self._app = app
        self.doc_dir = Path(doc_dir)
        self.app_version = app_version

        self.on_building_ui()

    def _read_local_doc(self, filename: str) -> str:
        doc_path = (self.doc_dir / filename).resolve()
        doc_dir = self.doc_dir.resolve()
        if doc_dir not in doc_path.parents and doc_path != doc_dir:
            raise ValueError(f"Documentation path escapes docs directory: {filename}")
        if not doc_path.exists():
            return ""
        return doc_path.read_text(encoding="utf-8")

    def on_building_ui(self):
        about_md = self._read_local_doc("about.md")
        if about_md:
            with gr.Accordion("About KURAGa", open=True):
                version_prefix = (
                    f"Version: {self.app_version}\n\n" if self.app_version else ""
                )
                gr.Markdown(version_prefix + about_md)

        user_guide_md = self._read_local_doc("usage.md")
        if user_guide_md:
            with gr.Accordion("User Guide", open=not KH_DEMO_MODE):
                gr.Markdown(user_guide_md)

        local_model_md = self._read_local_doc("local_model.md")
        if local_model_md:
            with gr.Accordion("Local Models", open=False):
                gr.Markdown(local_model_md)

        attribution_md = self._read_local_doc("attribution.md")
        if attribution_md:
            with gr.Accordion("Attribution", open=False):
                gr.Markdown(attribution_md)
