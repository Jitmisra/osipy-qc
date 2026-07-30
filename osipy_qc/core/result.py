"""
Result types and the overall-verdict aggregation rule.

Every QC check returns a `CheckResult`: a verdict, the numbers it measured
(`metric`), and a one-line human reason. The overall verdict is aggregated from
the per-check verdicts with a single, conservative rule.

Verdict meanings (must match QC_DESIGN.md):
  PASS    - the check ran and the data is fine
  WARN    - a mild problem was actually measured
  FAIL    - a real, disqualifying problem
  UNKNOWN - a check had no input to look at (excluded from the verdict; see below)
  N/A     - the check is structurally inapplicable to this data (excluded from the verdict)
  INFO    - a routing/description output, not a judgment (excluded from the verdict)

Why UNKNOWN does not escalate
-----------------------------
It used to. An UNKNOWN became a WARN on the reasoning that an incomplete report
deserves caution. In practice that conflated two different statements:

    "I measured something and it concerns me"        (a finding)
    "I was given nothing to measure"                 (an absence)

and the absence won. A reviewer who uploaded only a CBF map - a legitimate and
common thing to do - got WARN on a flawless map, because the ten checks needing
raw acquisition files or tissue maps all returned UNKNOWN. No file they could
supply would fix it short of supplying every file. A verdict that cannot be
earned is not a verdict, so UNKNOWN is now excluded from aggregation exactly as
N/A and INFO are.

The cost is real and must be paid elsewhere: the overall verdict no longer
carries the fact that the report was partial. A bare PASS on a lone CBF map
would be *more* misleading than the old WARN. So anything presenting a verdict
MUST also present the coverage - how many checks graded, and what was missing.
`coverage()` below returns that, and the JSON, HTML and web reports all show it.
Never render `overall` without it.
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


# Verdicts that state an absence rather than a finding, and so never move the
# overall verdict: a check that cannot apply (N/A), a routing description (INFO),
# and a check that had no input to look at (UNKNOWN). See the module docstring -
# excluding UNKNOWN is a deliberate change, and it obliges every caller that
# shows `overall` to show `coverage()` alongside it.
EXCLUDED_FROM_AGGREGATION = frozenset({Verdict.NA, Verdict.INFO, Verdict.UNKNOWN})


@dataclass
class CheckResult:
    """The uniform output of every QC check."""

    check: str                       # e.g. "1.qei" or "5.3.swap"
    verdict: Verdict
    metric: dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    #: True when this verdict was decided by an UNCALIBRATED threshold (an
    #: engineering default with no published derivation). Such a verdict is
    #: "provisional": under strict grading it can be a FAIL, but it would soften
    #: to a WARN with strict off, and it must never be presented as a hard,
    #: evidence-backed failure. The report renders these distinctly.
    provisional: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "check": self.check,
            "verdict": self.verdict.value,
            "metric": self.metric,
            "reason": self.reason,
            "provisional": self.provisional,
        }


def aggregate(results: list[CheckResult]) -> Verdict:
    """Combine per-check verdicts into one overall verdict.

    - any FAIL                         -> FAIL
    - else any WARN                    -> WARN
    - else (at least one PASS)         -> PASS
    - else (nothing gradeable at all)  -> UNKNOWN

    UNKNOWN, N/A and INFO are excluded: they report an absence, not a finding.
    A verdict from this function says only "of what could be measured, this is
    the worst of it" - it says nothing about how much could be measured. Pair it
    with `coverage()`.
    """
    graded = [r.verdict for r in results if r.verdict not in EXCLUDED_FROM_AGGREGATION]
    if Verdict.FAIL in graded:
        return Verdict.FAIL
    if Verdict.WARN in graded:
        return Verdict.WARN
    if Verdict.PASS in graded:
        return Verdict.PASS
    # Nothing was gradeable at all - every check was an absence. This is the one
    # case where UNKNOWN is the honest overall answer, and it is why a lone
    # unreadable file cannot come back PASS on a technicality.
    return Verdict.UNKNOWN


def coverage(results: list[CheckResult]) -> dict[str, Any]:
    """How much of the report was actually decided, and what was not.

    The companion to `aggregate()`. Since UNKNOWN no longer escalates, this is
    the only thing carrying "the report is partial" - so a caller that shows a
    verdict without showing this is misrepresenting it.

    Returns:
        graded:      how many checks reached a real verdict (PASS/WARN/FAIL)
        total:       how many checks ran, excluding the structurally inapplicable
        unknown:     how many had no input to look at
        complete:    True when nothing was missing
        missing:     the check ids that had no input, in registry order
    """
    graded = [r for r in results if r.verdict in (Verdict.PASS, Verdict.WARN, Verdict.FAIL)]
    unknown = [r for r in results if r.verdict is Verdict.UNKNOWN]
    return {
        "graded": len(graded),
        # N/A and INFO are not part of the denominator: a check that cannot apply
        # to this data was never owed an answer, so counting it as un-covered
        # would understate coverage.
        "total": len(graded) + len(unknown),
        "unknown": len(unknown),
        "complete": not unknown,
        "missing": [r.check for r in unknown],
    }
