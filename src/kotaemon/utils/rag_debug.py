from __future__ import annotations

import json
import logging
import sys
import time
import traceback
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

try:
    from theflow.settings import settings as flowsettings
except Exception:  # pragma: no cover - logger must not break imports
    flowsettings = None  # type: ignore[assignment]

_LOGGER_NAME = "kotaemon.rag_debug"
_JSONL_NAME = "rag_debug.jsonl"
_TEXT_NAME = "rag_debug.log"
_MAX_VALUE_CHARS = 8000
_INITIALIZED = False


def _log_dir() -> Path:
    root = getattr(flowsettings, "KH_APP_DATA_DIR", "ktem_app_data")
    path = Path(root) / "logs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _safe(value: Any, *, max_chars: int = _MAX_VALUE_CHARS) -> Any:
    if isinstance(value, dict):
        return {str(k): _safe(v, max_chars=max_chars) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v, max_chars=max_chars) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        if isinstance(value, str) and len(value) > max_chars:
            return value[:max_chars] + f"...[truncated {len(value) - max_chars} chars]"
        return value
    return repr(value)


def get_rag_debug_logger() -> logging.Logger:
    global _INITIALIZED
    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if _INITIALIZED:
        return logger

    log_dir = _log_dir()
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.INFO)
    logger.addHandler(stream_handler)

    text_handler = RotatingFileHandler(
        log_dir / _TEXT_NAME,
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    text_handler.setFormatter(formatter)
    text_handler.setLevel(logging.INFO)
    logger.addHandler(text_handler)

    _INITIALIZED = True
    return logger


def rag_log(event: str, **fields: Any) -> None:
    """Write one structured RAG debug event to console, text log, and JSONL.

    Logging must never break RAG execution. Large strings are truncated in the log;
    the live prompt remains unchanged.
    """

    record = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "event": event,
        **_safe(fields),
    }
    logger = get_rag_debug_logger()
    try:
        compact = json.dumps(record, ensure_ascii=False, default=str)
        logger.info(compact)
        with (_log_dir() / _JSONL_NAME).open("a", encoding="utf-8") as handle:
            handle.write(compact + "\n")
    except Exception:
        logger.error("rag_debug logging failed\n%s", traceback.format_exc())
