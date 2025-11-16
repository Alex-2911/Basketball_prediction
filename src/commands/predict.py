"""
Prediction command

Wraps the prediction script for the CLI interface.
"""

import subprocess
import sys
from pathlib import Path
from typing import Optional

from utils.logger import get_logger

logger = get_logger(__name__)

# Get the project root directory (where scripts/ is located)
PROJECT_ROOT = Path(__file__).parent.parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


def run_prediction(
    model_path: Optional[str] = None,
    output_dir: Optional[str] = None
) -> bool:
    """
    Generate predictions for upcoming games

    Args:
        model_path: Path to saved LightGBM model (optional)
        output_dir: Directory for prediction output files

    Returns:
        True if successful, False otherwise
    """
    script_path = SCRIPTS_DIR / "generate_predictions.py"
    cmd = [sys.executable, str(script_path)]

    # Note: Current script doesn't have these CLI args, but we're prepared for future
    # if model_path:
    #     cmd.extend(["--model-path", model_path])
    # if output_dir:
    #     cmd.extend(["--output-dir", output_dir])

    try:
        logger.info("Starting prediction generation...")
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,
            text=True
        )
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        logger.error(f"Prediction script failed with exit code {e.returncode}")
        return False
    except Exception as e:
        logger.error(f"Error running prediction script: {e}")
        return False
