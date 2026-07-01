"""File index loaded on demand so pipeline utilities stay lightweight."""

from typing import Any

__all__ = ["FileIndex"]


def __getattr__(name: str) -> Any:
    if name != "FileIndex":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from .index import FileIndex

    globals()[name] = FileIndex
    return FileIndex
