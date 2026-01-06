#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


NOTEBOOK_NAME = "_5. isotonic_calibrated_betting_engine_daily_driver.ipynb"


def resolve_source_root() -> Path:
    source_root = os.environ.get("SOURCE_ROOT")
    if source_root:
        return Path(source_root)
    workspace = os.environ.get("GITHUB_WORKSPACE")
    if workspace:
        return Path(workspace) / "2026"
    return Path("2026")


def resolve_notebook_path() -> Path:
    notebook_path = Path(NOTEBOOK_NAME)
    if notebook_path.exists():
        return notebook_path
    repo_root = Path(__file__).resolve().parents[1]
    fallback = repo_root / NOTEBOOK_NAME
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"Notebook not found: {NOTEBOOK_NAME}")


def run_notebook(notebook_path: Path) -> None:
    cmd = [
        sys.executable,
        "-m",
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        "--inplace",
        "--ExecutePreprocessor.timeout=1800",
        str(notebook_path),
    ]
    subprocess.run(cmd, check=True)


def assert_outputs(source_root: Path) -> None:
    output_dir = source_root / "output" / "LightGBM"
    required_files = [
        output_dir / "metrics_snapshot.json",
        output_dir / "strategy_params.txt",
    ]
    missing = [path for path in required_files if not path.exists()]
    if missing:
        missing_list = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing required outputs: {missing_list}")

    local_matched = sorted(output_dir.glob("local_matched_games_*.csv"))
    if not local_matched:
        raise FileNotFoundError(
            f"Missing required outputs: {output_dir}/local_matched_games_*.csv"
        )


def main() -> int:
    source_root = resolve_source_root()
    os.environ.setdefault("SOURCE_ROOT", str(source_root))
    os.environ.setdefault("N_WINDOW", "200")

    notebook_path = resolve_notebook_path()
    run_notebook(notebook_path)
    assert_outputs(source_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
