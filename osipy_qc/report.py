"""
The runner: feed a single `inputs` dict to every registered check, collect the
results, and aggregate one overall verdict. Each check pulls the inputs it needs
and returns UNKNOWN/N/A when something is missing — so the same call works whether
the user uploaded a full CBF map or just raw NIfTIs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from .core import CheckResult, QCConfig, Verdict, aggregate, all_checks


@dataclass
class QCReport:
    overall: Verdict
    results: list[CheckResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "overall_verdict": self.overall.value,
            "n_checks": len(self.results),
            "summary": self._summary(),
            "checks": [r.to_dict() for r in self.results],
        }

    def _summary(self) -> dict:
        counts: dict[str, int] = {}
        for r in self.results:
            counts[r.verdict.value] = counts.get(r.verdict.value, 0) + 1
        return counts

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


def run_qc(inputs: dict, cfg: QCConfig | None = None,
           checks: list[str] | None = None) -> QCReport:
    """Run all registered checks (or a named subset) over `inputs`.

    `inputs` is a flat dict of everything available, e.g.:
        cbf, gm, wm, csf, voxel_mm, brain,         # Stream B
        asl_4d, motion_params, files, context,     # Stream A
        m0_type, m0_tr_s, m0_background_suppression, m0_shape, asl_shape,
        sidecar, detected, structure, background_suppression, is_raw_deltam
    Missing keys simply yield UNKNOWN/N/A for the checks that need them.
    """
    cfg = cfg or QCConfig()
    results: list[CheckResult] = []
    for name, entry in all_checks().items():
        if checks is not None and name not in checks:
            continue
        try:
            res = entry["fn"](cfg=cfg, **inputs)
        except Exception as exc:  # a check must never crash the whole run
            res = CheckResult(name, Verdict.UNKNOWN, reason=f"check error: {exc}")
        results.append(res)

    results.sort(key=lambda r: r.check)
    return QCReport(aggregate(results), results)
