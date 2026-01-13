#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


SCRIPT5_NAME = "5_Isotonic_based_betting_strategy_2026.py"


def resolve_script5_path() -> Path:
    explicit = os.getenv("SCRIPT5_PATH")
    if explicit:
        path = Path(explicit)
        if not path.is_absolute():
            path = Path.cwd() / path
        if path.exists():
            return path
        raise FileNotFoundError(f"SCRIPT5_PATH gesetzt aber nicht gefunden: {path}")

    locations = [
        Path.cwd() / SCRIPT5_NAME,
        Path.cwd() / "src" / SCRIPT5_NAME,
        Path.cwd() / "scripts" / SCRIPT5_NAME,
        Path.cwd() / "2026" / "src" / SCRIPT5_NAME,
    ]
    for path in locations:
        if path.exists():
            return path

    matches = sorted(Path.cwd().rglob("*.py"))
    raise FileNotFoundError(
        f"Script 5 nicht gefunden. cwd={Path.cwd()} Kandidaten={matches[:20]}"
    )


def resolve_source_root() -> Path:
    source_root = os.environ.get("SOURCE_ROOT")
    if source_root:
        return Path(source_root)
    workspace = os.environ.get("GITHUB_WORKSPACE")
    if workspace:
        return Path(workspace) / "2026"
    return Path("2026")


def assert_outputs(source_root: Path) -> None:
    lgbm_dir = os.environ.get("LGBM_DIR")
    output_dir = Path(lgbm_dir) if lgbm_dir else source_root / "LightGBM"
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

    script_path = resolve_script5_path()
    print(f"Using Script 5: {script_path}")
    subprocess.run([sys.executable, str(script_path)], check=True)
    assert_outputs(source_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
