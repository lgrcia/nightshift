# type: ignore[attr-defined]
"""A Python package to compute and visualize an observation phase coverage"""

import sys

if sys.version_info >= (3, 8):
    from importlib import metadata as importlib_metadata
else:
    import importlib_metadata


def get_version() -> str:
    try:
        return importlib_metadata.version(__name__)
    except importlib_metadata.PackageNotFoundError:  # pragma: no cover
        return "unknown"


version: str = get_version()

from .main import coverage, period_match, coverage_multiple
