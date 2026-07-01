# KURAG Documentation

Welcome to the documentation for **KURAGa** — the **KU Retrieval-Augmented Guide Assistant**.

KURAGa is a student-built university-document RAG chatbot for the KU / WFI Digital Projects course. It helps users ask questions over curated university and programme documents. It is based on the open-source Cinnamon/kotaemon project, but this repository has been heavily adapted for the course.

!!! warning "Not an official KU service"
KURAGa can make mistakes. Verify important decisions against official university documents and staff guidance.

## Start here

- [Project overview](project_overview.md) — architecture, scope, and key files.
- [Guest guide](guest_guide.md) — how guest users can ask questions and read evidence.
- [Admin guide](admin_guide.md) — how admins configure models and index documents.
- [Local models](local_model.md) — Ollama/OpenAI-compatible local model setup.
- [Development](development.md) — setup, tests, CI, and repository conventions.
- [Attribution](attribution.md) — upstream Kotaemon license and project attribution.

## Quick start

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1      # Windows PowerShell
# source .venv/bin/activate      # Linux/macOS

pip install -r requirements_gerageragera39.txt
pip install -e .
cp .env.example .env
python app.py
```

Open <http://localhost:7860>.

## Runtime data

KURAGa stores runtime data in `ktem_app_data/`: SQLite metadata, uploads, vector stores, chunk/markdown caches, and temporary Gradio files. Do not commit or delete this directory without a backup.
