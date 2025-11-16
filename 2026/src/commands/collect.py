"""
Data collection commands

Wraps the data collection scripts for the CLI interface.
"""

import subprocess
import sys
from pathlib import Path
from typing import Optional

from logger import get_logger

logger = get_logger(__name__)

# Get the src directory
SRC_DIR = Path(__file__).parent.parent


def run_script(script_name: str, args: list = None) -> bool:
    """
    Run a Python script and return success status

    Args:
        script_name: Name of the script file (e.g., "1_get_data_previous_game_day_2026.py")
        args: Optional list of command-line arguments

    Returns:
        True if script succeeded, False otherwise
    """
    script_path = SRC_DIR / script_name
    cmd = [sys.executable, str(script_path)]

    if args:
        cmd.extend(args)

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,
            text=True
        )
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        logger.error(f"Script {script_name} failed with exit code {e.returncode}")
        return False
    except Exception as e:
        logger.error(f"Error running {script_name}: {e}")
        return False


def run_historical_collection(
    date: Optional[str] = None,
    collect_date: Optional[str] = None
) -> bool:
    """
    Collect historical game data

    Args:
        date: Anchor date in YYYY-MM-DD; collects games from the day before
        collect_date: Exact game date to collect in YYYY-MM-DD (overrides --date)

    Returns:
        True if successful, False otherwise
    """
    args = []
    if collect_date:
        args.extend(["--collect-date", collect_date])
    elif date:
        args.extend(["--date", date])

    logger.info("Starting historical data collection...")
    return run_script("collect_historical_games.py", args)


def run_upcoming_collection(date: Optional[str] = None) -> bool:
    """
    Collect upcoming game schedule and odds

    Args:
        date: Date for which to collect upcoming games (YYYY-MM-DD)

    Returns:
        True if successful, False otherwise
    """
    args = []
    if date:
        args.extend(["--date", date])

    logger.info("Starting upcoming games collection...")
    return run_script("collect_upcoming_games.py", args)
