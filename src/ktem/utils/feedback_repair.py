from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


FEEDBACK_REPAIR_REASONS: list[tuple[str, str]] = [
    ("Answer is incomplete", "incomplete"),
    ("Answer does not address the question", "not_answering"),
    ("Sources are irrelevant or weak", "bad_sources"),
    ("Answer contains unsupported claims", "hallucination"),
    ("Answer format is unclear", "bad_format"),
    ("Other", "other"),
]

FEEDBACK_REPAIR_REASON_LABELS = {
    value: label for label, value in FEEDBACK_REPAIR_REASONS
}


_REPAIR_PROMPTS: dict[str, str] = {
    "incomplete": (
        "The previous answer was incomplete. Regenerate a more complete answer "
        "using all relevant context. Do not omit conditions, exceptions, dates, "
        "constraints, course/module names, requirements, or procedural details "
        "that are explicitly present in the context."
    ),
    "not_answering": (
        "The previous answer did not focus on the user's exact question. "
        "Answer the precise question asked, using only context that directly "
        "supports the answer. Avoid tangents."
    ),
    "bad_sources": (
        "The previous answer used weak or irrelevant evidence. Use only the most "
        "relevant context fragments. Ignore weakly related context even if it is "
        "present in the retrieved evidence."
    ),
    "hallucination": (
        "The previous answer may contain unsupported claims. Remove every claim "
        "that is not explicitly supported by the context. Do not infer unstated "
        "requirements, deadlines, procedures, course rules, or policies. If the "
        "context does not answer the question, say that the knowledge base does "
        "not contain enough information."
    ),
    "bad_format": (
        "The previous answer was poorly formatted. Keep the same grounding rules, "
        "but regenerate in a clearer structure: short answer first, then concise "
        "details, then sources/citations if available."
    ),
    "other": (
        "Regenerate safely under strict grounding rules. Use only explicitly "
        "supported facts from the context and state when the knowledge base does "
        "not contain enough information."
    ),
}


_SNAPSHOT_SUFFIXES = (
    ".num_retrieval",
    ".retrieval_mode",
    ".candidate_multiplier",
    ".context_expansion_mode",
    ".sibling_window",
    ".enable_query_expansion",
    ".mmr",
    ".use_reranking",
    ".use_llm_reranking",
    ".max_context_length",
    ".system_prompt",
    ".qa_prompt",
)


def normalize_feedback_reason(reason: str | None) -> str:
    if reason in FEEDBACK_REPAIR_REASON_LABELS:
        return str(reason)
    return "other"


def build_feedback_repair_prompt(reason: str | None) -> str:
    """Return the extra one-shot repair instruction for a disliked answer."""

    return _REPAIR_PROMPTS[normalize_feedback_reason(reason)]


def _safe_int(value: Any, default: int) -> int:
    try:
        if value in (None, ""):
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _increase_int(
    settings: dict[str, Any],
    key: str,
    *,
    default: int,
    minimum: int | None = None,
    maximum: int,
    factor: float | None = None,
    add: int = 0,
) -> None:
    current = _safe_int(settings.get(key), default)
    candidate = current + add
    if factor is not None:
        candidate = max(candidate, int(round(current * factor)))
    if minimum is not None:
        candidate = max(candidate, minimum)
    settings[key] = min(candidate, maximum)


def _append_prompt(base: str | None, addition: str) -> str:
    base = (base or "").strip()
    if addition in base:
        return base
    if base:
        return f"{base}\n\nFeedback repair instruction:\n{addition}"
    return addition


def _append_reasoning_prompts(
    settings: dict[str, Any],
    reason: str,
    reasoning_id: str,
) -> None:
    prefix = f"reasoning.options.{reasoning_id}"
    repair_prompt = build_feedback_repair_prompt(reason)
    for suffix in ("system_prompt", "qa_prompt"):
        key = f"{prefix}.{suffix}"
        if key in settings:
            settings[key] = _append_prompt(settings.get(key), repair_prompt)


def apply_feedback_repair_settings(
    settings: dict[str, Any],
    reason: str | None,
    *,
    reasoning_id: str = "simple",
) -> dict[str, Any]:
    """Return settings copy with a one-shot repair preset applied.

    The function intentionally operates on a copy so disliked-answer regeneration
    does not persistently change user settings.
    """

    repaired = deepcopy(settings)
    reason = normalize_feedback_reason(reason)

    for key in list(repaired.keys()):
        if key.endswith(".num_retrieval"):
            if reason == "incomplete":
                _increase_int(
                    repaired,
                    key,
                    default=15,
                    minimum=20,
                    maximum=40,
                    factor=1.5,
                )
            elif reason == "bad_sources":
                _increase_int(
                    repaired,
                    key,
                    default=15,
                    minimum=15,
                    maximum=20,
                    add=2,
                )
            elif reason == "hallucination":
                repaired[key] = min(_safe_int(repaired.get(key), 15), 15)

        elif key.endswith(".candidate_multiplier"):
            if reason == "incomplete":
                _increase_int(
                    repaired,
                    key,
                    default=20,
                    minimum=30,
                    maximum=50,
                    factor=1.5,
                )
            elif reason in {"bad_sources", "hallucination"}:
                _increase_int(
                    repaired,
                    key,
                    default=20,
                    minimum=30,
                    maximum=40,
                    add=10,
                )

        elif key.endswith(".context_expansion_mode") and reason == "incomplete":
            repaired[key] = "siblings"

        elif key.endswith(".sibling_window") and reason == "incomplete":
            _increase_int(
                repaired,
                key,
                default=1,
                minimum=2,
                maximum=3,
                add=1,
            )

        elif key.endswith(".enable_query_expansion") and reason in {
            "not_answering",
            "incomplete",
        }:
            repaired[key] = True

        elif key.endswith(".mmr") and reason == "bad_sources":
            repaired[key] = True

        elif key.endswith(".use_reranking") and reason in {
            "bad_sources",
            "hallucination",
        }:
            repaired[key] = True

        elif key.endswith(".max_context_length") and reason == "incomplete":
            _increase_int(
                repaired,
                key,
                default=32000,
                maximum=64000,
                factor=1.25,
            )

    _append_reasoning_prompts(repaired, reason, reasoning_id)
    return repaired


def snapshot_feedback_repair_settings(settings: dict[str, Any]) -> dict[str, Any]:
    """Keep a compact quality-debug snapshot instead of storing all settings."""

    return {
        key: deepcopy(value)
        for key, value in settings.items()
        if key.endswith(_SNAPSHOT_SUFFIXES)
        or key in {"reasoning.use", "reasoning.lang", "reasoning.max_context_length"}
    }


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_feedback_event(
    data_source: dict[str, Any],
    *,
    message_index: int | None,
    liked: bool,
    reason: str | None = None,
    comment: str | None = None,
    question: str | None = None,
    old_answer: str | None = None,
    new_answer: str | None = None,
    repair_preset_name: str | None = None,
    selected: dict[str, Any] | None = None,
    settings_before: dict[str, Any] | None = None,
    settings_after: dict[str, Any] | None = None,
    retrieval_before: str | None = None,
    retrieval_after: str | None = None,
    event_id: str | None = None,
) -> str:
    """Append a backward-compatible rich feedback event and return its id."""

    event_id = event_id or uuid4().hex
    event = {
        "event_id": event_id,
        "timestamp": now_iso(),
        "message_index": message_index,
        "liked": liked,
        "reason": normalize_feedback_reason(reason) if reason else None,
        "comment": comment or None,
        "question": question,
        "old_answer": old_answer,
        "new_answer": new_answer,
        "repair_preset_name": repair_preset_name,
        "selected": deepcopy(selected) if selected is not None else None,
        "settings_before": deepcopy(settings_before) if settings_before else None,
        "settings_after": deepcopy(settings_after) if settings_after else None,
        "retrieval_before": retrieval_before,
        "retrieval_after": retrieval_after,
    }

    events = list(data_source.get("feedback_events", []))
    events.append(event)
    data_source["feedback_events"] = events
    return event_id


def update_feedback_event(
    data_source: dict[str, Any],
    event_id: str | None,
    **updates: Any,
) -> bool:
    """Update an existing rich feedback event in-place."""

    if not event_id:
        return False
    events = list(data_source.get("feedback_events", []))
    for event in reversed(events):
        if event.get("event_id") == event_id:
            event.update({key: deepcopy(value) for key, value in updates.items()})
            event["updated_at"] = now_iso()
            data_source["feedback_events"] = events
            return True
    return False
