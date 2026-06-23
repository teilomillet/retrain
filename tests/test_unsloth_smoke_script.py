from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_unsloth_smoke_script_help_is_available_without_training_stack_imports():
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "smoke_unsloth_backend.py"

    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert result.returncode == 0
    assert "--model" in result.stdout
    assert "--max-seq-length" in result.stdout
    assert "--require-cuda" in result.stdout
