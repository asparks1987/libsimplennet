"""Checkout compatibility shim for the historical ``libSimpleNNet`` module."""

from __future__ import annotations

import sys
from pathlib import Path

_src = Path(__file__).resolve().parent / "src"
if _src.exists() and str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from simplennet.legacy import *  # noqa: F401,F403,E402
