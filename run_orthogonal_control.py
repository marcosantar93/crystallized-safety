#!/usr/bin/env python3
"""
Convenience entrypoint.

Delegates to `scripts/run_orthogonal_control.py`.
"""

from __future__ import annotations

import runpy
from pathlib import Path


def main() -> None:
    script = Path(__file__).parent / "scripts" / "run_orthogonal_control.py"
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()

