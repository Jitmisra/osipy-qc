"""
A tiny registry so checks self-register with a decorator, mirroring osipy's
`@register_model` pattern. This keeps the runner decoupled from the checks:
the runner just iterates the registry.
"""

from __future__ import annotations

from typing import Callable

# name -> {"fn": callable, "stream": "A"|"B", "required": bool}
REGISTRY: dict[str, dict] = {}


def register_qc_check(name: str, stream: str | None = None, required: bool = True) -> Callable:
    """Decorator that registers a check function under `name`."""

    def decorator(fn: Callable) -> Callable:
        if name in REGISTRY:
            raise ValueError(f"duplicate QC check name: {name!r}")
        REGISTRY[name] = {"fn": fn, "stream": stream, "required": required}
        return fn

    return decorator


def get_check(name: str) -> Callable:
    return REGISTRY[name]["fn"]


def all_checks() -> dict[str, dict]:
    return dict(REGISTRY)
