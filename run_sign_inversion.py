#!/usr/bin/env python3
"""
Convenience entrypoint.

This repo's runnable scripts live in `scripts/`. The README references
`run_sign_inversion.py` at the repo root, so this thin wrapper keeps that
command working while delegating to `scripts/run_sign_inversion.py`.
"""

from __future__ import annotations

import runpy
from pathlib import Path


def main() -> None:
    script = Path(__file__).parent / "scripts" / "run_sign_inversion.py"
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()

