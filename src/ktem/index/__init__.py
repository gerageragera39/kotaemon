"""Index management loaded on demand to keep utility imports lightweight."""

from typing import Any

__all__ = ["IndexManager"]


def __getattr__(name: str) -> Any:
    if name != "IndexManager":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from .manager import IndexManager

    globals()[name] = IndexManager
    return IndexManager
