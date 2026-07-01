"""
Result types and the overall-verdict aggregation rule.

Every QC check returns a `CheckResult`: a verdict, the numbers it measured
(`metric`), and a one-line human reason. The overall verdict is aggregated from
the per-check verdicts with a single, conservative rule.

Verdict meanings (must match QC_DESIGN.md):
  PASS    - the check ran and the data is fine
  WARN    - mild problem, or the report is incomplete (an UNKNOWN escalates here)
  FAIL    - a real, disqualifying problem
  UNKNOWN - a check that *should* have run *couldn't* (missing input) -> escalates to WARN
  N/A     - the check is structurally inapplicable to this data (excluded from the verdict)
  INFO    - a routing/description output, not a judgment (excluded from the verdict)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Verdict(str, Enum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"
    UNKNOWN = "UNKNOWN"
    NA = "N/A"
    INFO = "INFO"


# N/A and INFO never affect the overall verdict (a check that cannot apply, or a
# pure routing description, must not drag a clean scan down). Only UNKNOWN escalates.
EXCLUDED_FROM_AGGREGATION = frozenset({Verdict.NA, Verdict.INFO})


@dataclass
class CheckResult:
    """The uniform output of every QC check."""

    check: str                       # e.g. "1.qei" or "5.3.swap"
    verdict: Verdict
    metric: dict[str, Any] = field(default_factory=dict)
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "check": self.check,
            "verdict": self.verdict.value,
            "metric": self.metric,
            "reason": self.reason,
        }


def aggregate(results: list[CheckResult]) -> Verdict:
    """Combine per-check verdicts into one overall verdict.

    - any FAIL                         -> FAIL
    - else any WARN or UNKNOWN         -> WARN
    - else (at least one PASS)         -> PASS
    - else (nothing gradeable at all)  -> UNKNOWN

    N/A and INFO are excluded entirely.
    """
    graded = [r.verdict for r in results if r.verdict not in EXCLUDED_FROM_AGGREGATION]
    if Verdict.FAIL in graded:
        return Verdict.FAIL
    if Verdict.WARN in graded or Verdict.UNKNOWN in graded:
        return Verdict.WARN
    if Verdict.PASS in graded:
        return Verdict.PASS
    return Verdict.UNKNOWN
