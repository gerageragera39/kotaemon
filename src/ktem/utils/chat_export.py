import csv
import html
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from theflow.settings import settings as flowsettings

CHAT_OUTPUT_DIR = (
    Path(getattr(flowsettings, "KH_APP_DATA_DIR", Path.cwd() / "ktem_app_data"))
    / "chats"
)


def _json_csv_cell(value: Any) -> str:
    if value is None:
        return ""
    return json.dumps(value, ensure_ascii=False)


def plain_context(context_html: str) -> str:
    if not context_html:
        return ""
    text = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", str(context_html))
    text = re.sub(r"(?s)<br\s*/?>", "\n", text)
    text = re.sub(r"(?s)</(p|div|li|h[1-6]|tr|table)>", "\n", text)
    text = re.sub(r"(?s)<.*?>", " ", text)
    text = html.unescape(text)
    return re.sub(r"[ \t\r\f\v]+", " ", text).strip()


def chat_csv_path(convo_id: Any) -> Path:
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(convo_id)).strip("_")
    return CHAT_OUTPUT_DIR / f"chat_{safe_id or 'unknown'}.csv"


def export_chat_csv(
    convo_id: Any,
    messages: list | tuple | None,
    retrieval_history: list | tuple | None,
    plot_history: list | tuple | None,
    selected: dict | None,
) -> Path:
    """Persist a complete, idempotent CSV snapshot for one normal chat."""

    CHAT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    exported_at = datetime.now().isoformat(timespec="seconds")
    retrieval_items = retrieval_history or []
    plot_items = plot_history or []

    for turn_index, message in enumerate(messages or [], start=1):
        if not isinstance(message, (list, tuple)) or len(message) < 2:
            continue

        question = "" if message[0] is None else str(message[0])
        answer = "" if message[1] is None else str(message[1])
        context_html = ""
        if turn_index - 1 < len(retrieval_items):
            context_html = retrieval_items[turn_index - 1] or ""
        context_text = plain_context(context_html)
        plot_data = None
        if turn_index - 1 < len(plot_items):
            plot_data = plot_items[turn_index - 1]

        rows.append(
            {
                "conversation_id": convo_id,
                "turn_index": turn_index,
                "question": question,
                "answer": answer,
                "contexts": _json_csv_cell([context_text] if context_text else []),
                "context": context_text,
                "context_count": 1 if context_text else 0,
                "context_html": context_html,
                "plot_data": _json_csv_cell(plot_data),
                "selected_sources": _json_csv_cell(selected or {}),
                "exported_at": exported_at,
            }
        )

    output_path = chat_csv_path(convo_id)
    fieldnames = [
        "conversation_id",
        "turn_index",
        "question",
        "answer",
        "contexts",
        "context",
        "context_count",
        "context_html",
        "plot_data",
        "selected_sources",
        "exported_at",
    ]
    with output_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output_path
