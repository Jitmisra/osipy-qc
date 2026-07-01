"""Importing this package registers every check via its decorator.

Add new check modules here as they are built so the runner can discover them.
"""

from . import cbf_level  # noqa: F401  (Module 3)
from . import coreg      # noqa: F401  (Module 4)
from . import m0         # noqa: F401  (Module 6)
from . import motion     # noqa: F401  (Module 7)
from . import noise      # noqa: F401  (Module 2)
from . import qei        # noqa: F401  (Module 1)
from . import schema     # noqa: F401  (Module 5 + 8.2)

__all__ = ["qei", "noise", "cbf_level", "coreg", "schema", "m0", "motion"]
