"""
osipy_qc — Quality Control ToolBox for ASL MRI.

Grades an ASL-derived CBF map (and the raw data) and returns a PASS / WARN / FAIL
verdict per check, with reasons. A QC *layer* — it does not reprocess data.

Pure NumPy (+ nibabel for I/O). No scipy.
"""

from __future__ import annotations

__version__ = "0.1.0"

from . import checks  # noqa: F401  (registers all checks on import)
from .core import CheckResult, QCConfig, Verdict, aggregate, all_checks
from .io import grade_cbf, load_cbf_inputs
from .report import QCReport, run_qc

__all__ = [
    "__version__",
    "CheckResult",
    "QCConfig",
    "Verdict",
    "aggregate",
    "all_checks",
    "QCReport",
    "run_qc",
    "grade_cbf",
    "load_cbf_inputs",
]
