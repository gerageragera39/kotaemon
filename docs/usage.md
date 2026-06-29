# Usage

KURAGa has two main user paths: restricted guest chat and admin document management.

## Guest usage

1. Select **Access as Guest**.
2. Ask a question in **Chat**.
3. Review the answer and evidence panel.
4. Use follow-up questions to narrow unclear answers.

Guests automatically search all admin-indexed documents and cannot change the document scope.

## Admin usage

1. Log in as an admin.
2. Configure models under **Resources** or in `flowsettings.py`.
3. Upload and index documents in the file collection tab.
4. Test answers in **Chat**.
5. Use **Evaluation** for curated checks when needed.

## Chat panel

The chat area contains:

- message history and input;
- optional conversation controls for authenticated non-guest users;
- retrieved evidence/citations in the information panel;
- chat settings such as reasoning method, language, citations, and mindmap where enabled.

## Evidence scores

When scores are shown, they can include vector similarity, reranking score, LLM relevance, or answer confidence. Treat scores as debugging aids, not guarantees.

## Feedback workflow

If the feedback/dislike repair workflow is enabled, disliking the latest answer can expose repair options and regenerate the answer with adjusted retrieval/reasoning settings. Older feedback is stored but may not trigger regeneration.
