"""
A tiny registry so checks self-register with a decorator, mirroring osipy's
`@register_model` pattern. This keeps the runner decoupled from the checks:
the runner just iterates the registry.

Organs
------
A check declares which organ(s) it grades. Until kidney and placenta arrived,
`cfg.organ` existed but nothing read it, so every organ ran the brain's checks -
which would have graded a kidney against a grey-matter prior. The organ is a
registration-time fact, not a runtime branch inside each check, because the
kidney and placenta designs give their checks their OWN ids and their OWN
methods (K6.2 is "M0 without labelling AND without background suppression",
which is not the brain's 6.3). Sharing one function across organs would mean
one of the two organs' published rules losing.

So: `organ="brain"` by default, and the runner shows a check only to the organ
that registered it. A genuinely organ-independent check can pass a tuple.
"""

from __future__ import annotations

from typing import Callable

# name -> {"fn": callable, "stream": "A"|"B", "required": bool, "organs": tuple[str, ...]}
REGISTRY: dict[str, dict] = {}

KNOWN_ORGANS: tuple[str, ...] = ("brain", "kidney", "placenta")


def register_qc_check(name: str, stream: str | None = None, required: bool = True,
                      organ: str | tuple[str, ...] = "brain") -> Callable:
    """Decorator that registers a check function under `name`.

    `organ` is the organ (or tuple of organs) this check grades. It is validated
    at import time: a typo like organ="kidneys" would otherwise register a check
    that silently never runs, and a check that never runs looks exactly like a
    check that always passes.
    """
    organs = (organ,) if isinstance(organ, str) else tuple(organ)
    unknown = [o for o in organs if o not in KNOWN_ORGANS]
    if unknown:
        raise ValueError(
            f"check {name!r} registered for unknown organ(s) {unknown}; "
            f"known: {list(KNOWN_ORGANS)}")

    def decorator(fn: Callable) -> Callable:
        if name in REGISTRY:
            raise ValueError(f"duplicate QC check name: {name!r}")
        REGISTRY[name] = {"fn": fn, "stream": stream, "required": required,
                          "organs": organs}
        return fn

    return decorator


def get_check(name: str) -> Callable:
    return REGISTRY[name]["fn"]


def all_checks(organ: str | None = None) -> dict[str, dict]:
    """Every registered check, or only those grading `organ`."""
    if organ is None:
        return dict(REGISTRY)
    return {n: e for n, e in REGISTRY.items() if organ in e["organs"]}


def checks_for_organ(organ: str) -> list[str]:
    """Names of the checks that grade `organ`, in registry order."""
    return list(all_checks(organ))


def organs_covered() -> dict[str, int]:
    """How many checks each organ has. Used by the CLI/report to say what is
    implemented, so an organ with no checks cannot masquerade as a clean pass."""
    out = {o: 0 for o in KNOWN_ORGANS}
    for entry in REGISTRY.values():
        for o in entry["organs"]:
            out[o] = out.get(o, 0) + 1
    return out
