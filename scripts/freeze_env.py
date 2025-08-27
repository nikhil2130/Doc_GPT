# -*- coding: utf-8 -*-
"""
Freeze the current venv into requirements-freeze.txt (exact versions).
Run this *after* a successful install in your .venv.

Usage (PowerShell):
  cd D:\projects\Doc_GPT
  .\.venv\Scripts\Activate.ps1
  py -3.12 .\scripts\freeze_env.py
"""
from __future__ import annotations

import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "requirements-freeze.txt"

def die(msg: str) -> None:
    print(f"[fatal] {msg}", flush=True)
    sys.exit(1)

def main() -> None:
    # Ensure we are in a venv
    if "VIRTUAL_ENV" not in os.environ:
        die("No active virtual environment detected. Activate .venv first.")

    # Ensure pip exists in this interpreter
    try:
        subprocess.run([sys.executable, "-m", "pip", "--version"], check=True, capture_output=True)
    except Exception as e:
        die(f"pip not available in this Python: {e!r}")

    # Run pip freeze
    print("[freeze] Collecting installed packages...")
    proc = subprocess.run([sys.executable, "-m", "pip", "freeze"], check=True, capture_output=True, text=True)
    lines = proc.stdout.strip().splitlines()

    # Write with a header (timestamp + Python version)
    header = [
        "# Locked environment for Doc_GPT",
        f"# Generated at: {datetime.now(timezone.utc).isoformat()}",
        f"# Python: {sys.version.split()[0]}",
        "# NOTE: This file is generated from the current .venv. Do not edit manually.",
        "",
    ]
    OUT.write_text("\n".join(header + lines) + "\n", encoding="utf-8")
    print(f"[done] Wrote {OUT}")

if __name__ == "__main__":
    main()
