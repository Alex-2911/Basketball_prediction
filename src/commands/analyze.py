"""
Analysis commands

Wraps the analysis scripts for the CLI interface.
"""

import subprocess
import sys
from pathlib import Path

from utils.logger import get_logger

logger = get_logger(__name__)

# Get the project root directory (where scripts/ is located)
PROJECT_ROOT = Path(__file__).parent.parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


def run_script(script_name: str) -> bool:
    """
    Run a Python script and return success status

    Args:
        script_name: Name of the script file

    Returns:
        True if script succeeded, False otherwise
    """
    script_path = SCRIPTS_DIR / script_name
    cmd = [sys.executable, str(script_path)]

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


def run_statistics() -> bool:
    """
    Calculate betting statistics from historical predictions

    Returns:
        True if successful, False otherwise
    """
    logger.info("Calculating betting statistics...")
    return run_script("calculate_betting_statistics.py")


def run_kelly() -> bool:
    """
    Calculate Kelly Criterion betting parameters

    Returns:
        True if successful, False otherwise
    """
    logger.info("Calculating Kelly Criterion parameters...")
    return run_script("calculate_kelly_parameters.py")


def run_recommendations() -> bool:
    """
    Generate betting recommendations with optimal stakes

    Returns:
        True if successful, False otherwise
    """
    logger.info("Generating betting recommendations...")
    return run_script("show_bet_recommendations.py")


def run_all_analysis() -> bool:
    """
    Run all analysis steps in sequence

    Returns:
        True if all steps succeeded, False otherwise
    """
    logger.info("Starting complete betting analysis...")

    # Run statistics
    if not run_statistics():
        logger.error("Statistics calculation failed")
        return False

    # Run Kelly Criterion
    if not run_kelly():
        logger.error("Kelly Criterion calculation failed")
        return False

    # Generate recommendations
    if not run_recommendations():
        logger.error("Recommendation generation failed")
        return False

    logger.info("Complete analysis finished successfully")
    return True
