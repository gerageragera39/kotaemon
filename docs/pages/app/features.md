## Chat

The kotaemon focuses on question and answering over a corpus of data. Below
is the gentle introduction about the chat functionality.

- Users can upload corpus of files.
- Users can converse to the chatbot to ask questions about the corpus of files.
- Users can view the reference in the files.

### Feedback-aware adaptive RAG regeneration

For university RAG chats, a thumbs-down records the feedback and asks the user
what is wrong with the answer. The selected reason chooses a temporary repair
strategy for that regeneration only: broader retrieval for incomplete answers,
query-focus or reranking/source filtering for relevance issues, stricter
grounding for hallucinations, and prompt-only cleanup for formatting problems.
The regenerated answer replaces the previous turn via the existing regen flow,
and a rich `feedback_events` entry is saved alongside the backward-compatible
`likes` list for later quality analysis.

#### User flow

1. A thumbs-up is stored as feedback only. It does not trigger regeneration.
2. A thumbs-down is stored immediately and opens a compact reason selector near
   the chat area.
3. The user selects why the answer is wrong and clicks **Regenerate**.
4. Kotaemon regenerates the latest assistant answer for the same user question.
5. The regenerated answer replaces the previous answer for that turn, and the
   retrieval/reference panel is updated through the existing regeneration path.

Adaptive regeneration currently targets the latest assistant answer. If a user
dislikes an older answer, Kotaemon keeps the feedback but does not rewrite the
middle of the conversation history.

#### Components and responsibilities

- `src/ktem/pages/chat/__init__.py`
  - Renders the feedback repair UI: reason selector, optional comment, and
    regenerate/cancel buttons.
  - Handles Gradio `Chatbot.like(...)` events.
  - Saves thumbs-up/thumbs-down events without changing the legacy `likes`
    structure.
  - Reuses the existing `state["app"]["regen"]` flow to regenerate the latest
    answer instead of introducing a separate chat execution path.
  - Stores temporary regeneration metadata under
    `state["app"]["feedback_regen"]` and clears it after persistence.

- `src/ktem/utils/feedback_repair.py`
  - Defines the feedback reason values shown in the UI.
  - Maps each reason to a one-shot repair instruction.
  - Applies temporary retrieval/prompt changes to a copied settings dictionary.
  - Creates compact before/after settings snapshots for later quality analysis.
  - Appends and updates rich `feedback_events` records.

- `src/ktem/reasoning/simple.py`
  - Defines stricter default university RAG system and QA prompts.
  - Treats the retrieved context as the single source of truth.
  - Instructs the model to avoid prior knowledge, unsupported inference, and
    silently choosing between contradictory context fragments.

#### Repair presets

- `incomplete`
  - Broadens retrieval with a higher final chunk count and candidate pool.
  - Enables sibling context expansion and a slightly larger sibling window.
  - Adds an instruction to include all relevant conditions, exceptions, dates,
    constraints, course names, module names, and requirements.

- `not_answering`
  - Enables query expansion when available.
  - Adds an instruction to focus strictly on the user's exact question.
  - Avoids blindly increasing the final top-k because extra context can add
    noise when the main issue is focus.

- `bad_sources`
  - Enables reranking and MMR where supported by the selected retriever.
  - Expands the candidate pool cautiously while keeping final top-k bounded.
  - Adds an instruction to use only the strongest relevant evidence.

- `hallucination`
  - Enables reranking but does not aggressively increase final top-k.
  - Adds a stricter grounding instruction to remove unsupported claims and say
    when the knowledge base is insufficient.

- `bad_format`
  - Leaves retrieval settings unchanged.
  - Adds a formatting-only instruction: short answer first, then details, then
    sources/citations if available.

- `other`
  - Uses a safe default repair instruction with strict grounding.

All presets are temporary. They are applied only to the copied settings used for
the active regeneration and are not written back to the user's global settings.

#### Persistence format

Kotaemon still writes the legacy `data_source["likes"]` list for backward
compatibility. It also writes `data_source["feedback_events"]` entries with:

- timestamp and event id;
- message index;
- liked/disliked value;
- selected reason and optional comment;
- original question;
- old answer and regenerated answer when available;
- selected repair preset;
- selected index/file state;
- compact settings snapshots before and after the repair preset;
- retrieval/reference panel content before and after regeneration when
  available.

This structure is intended for later offline quality analysis and does not
change normal conversation loading.
