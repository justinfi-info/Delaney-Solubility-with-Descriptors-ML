"""
Pipeline package public API.

Keep imports lazy to avoid brittle import-time failures (e.g., when running the
app from different working directories or under reloaders).
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = ["PredictPipeline", "CustomData"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        module = import_module(".predict_pipeline", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + __all__)
