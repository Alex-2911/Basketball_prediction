"""
Tests for the CLI interface

These tests verify that the CLI commands are properly configured and can be invoked.
"""

import subprocess
import sys
from pathlib import Path

import pytest

# Path to the CLI module
CLI_MODULE = Path(__file__).parent.parent / "2026" / "src" / "cli.py"


class TestCLIStructure:
    """Test CLI command structure and help text"""

    def test_cli_help(self):
        """Test that the main CLI help works"""
        result = subprocess.run(
            [sys.executable, str(CLI_MODULE), "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "NBA Prediction System" in result.stdout
        assert "collect" in result.stdout
        assert "predict" in result.stdout
        assert "analyze" in result.stdout
        assert "pipeline" in result.stdout
        assert "dashboard" in result.stdout

    def test_cli_version(self):
        """Test that the version flag works"""
        result = subprocess.run(
            [sys.executable, str(CLI_MODULE), "--version"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "2026.1.0" in result.stdout

    def test_collect_help(self):
        """Test that collect subcommand help works"""
        result = subprocess.run(
            [sys.executable, str(CLI_MODULE), "collect", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "historical" in result.stdout
        assert "upcoming" in result.stdout

    def test_collect_historical_help(self):
        """Test that collect historical command help works"""
        result = subprocess.run(
            [sys.executable, str(CLI_MODULE), "collect", "historical", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "Scrape historical game data" in result.stdout
        assert "--date" in result.stdout
        assert "--collect-date" in result.stdout

    def test_collect_upcoming_help(self):
        """Test that collect upcoming command help works"""
        result = subprocess.run(
            [sys.executable, str(CLI_MODULE), "collect", "upcoming", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "Get upcoming game schedule" in result.stdout

    def test_predict_help(self):
        """Test that predict command help works"""
        result = subprocess.run(
            [sys.executable, str(CLI_MODULE), "predict", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "Generate predictions" in result.stdout

    def test_analyze_help(self):
        """Test that analyze subcommand help works"""
        result = subprocess.run(
            [sys.executable, str(CLI_MODULE), "analyze", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "stats" in result.stdout
        assert "kelly" in result.stdout
        assert "recommend" in result.stdout
        assert "all" in result.stdout

    def test_analyze_stats_help(self):
        """Test that analyze stats command help works"""
        result = subprocess.run(
            [sys.executable, str(CLI_MODULE), "analyze", "stats", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "Calculate betting statistics" in result.stdout

    def test_analyze_kelly_help(self):
        """Test that analyze kelly command help works"""
        result = subprocess.run(
            [sys.executable, str(CLI_MODULE), "analyze", "kelly", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "Kelly Criterion" in result.stdout

    def test_analyze_recommend_help(self):
        """Test that analyze recommend command help works"""
        result = subprocess.run(
            [sys.executable, str(CLI_MODULE), "analyze", "recommend", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "betting recommendations" in result.stdout

    def test_analyze_all_help(self):
        """Test that analyze all command help works"""
        result = subprocess.run(
            [sys.executable, str(CLI_MODULE), "analyze", "all", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "all analysis steps" in result.stdout

    def test_pipeline_help(self):
        """Test that pipeline command help works"""
        result = subprocess.run(
            [sys.executable, str(CLI_MODULE), "pipeline", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "complete NBA prediction pipeline" in result.stdout
        assert "--skip-collection" in result.stdout
        assert "--skip-analysis" in result.stdout

    def test_dashboard_help(self):
        """Test that dashboard command help works"""
        result = subprocess.run(
            [sys.executable, str(CLI_MODULE), "dashboard", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "Streamlit dashboard" in result.stdout
        assert "--port" in result.stdout


class TestCLICommandModules:
    """Test that command modules can be imported"""

    def test_import_collect_module(self):
        """Test that collect command module can be imported"""
        import sys
        from pathlib import Path

        src_dir = Path(__file__).parent.parent / "2026" / "src"
        sys.path.insert(0, str(src_dir))

        from commands import collect

        assert hasattr(collect, "run_historical_collection")
        assert hasattr(collect, "run_upcoming_collection")

    def test_import_predict_module(self):
        """Test that predict command module can be imported"""
        import sys
        from pathlib import Path

        src_dir = Path(__file__).parent.parent / "2026" / "src"
        sys.path.insert(0, str(src_dir))

        from commands import predict

        assert hasattr(predict, "run_prediction")

    def test_import_analyze_module(self):
        """Test that analyze command module can be imported"""
        import sys
        from pathlib import Path

        src_dir = Path(__file__).parent.parent / "2026" / "src"
        sys.path.insert(0, str(src_dir))

        from commands import analyze

        assert hasattr(analyze, "run_statistics")
        assert hasattr(analyze, "run_kelly")
        assert hasattr(analyze, "run_recommendations")
        assert hasattr(analyze, "run_all_analysis")

    def test_import_pipeline_module(self):
        """Test that pipeline command module can be imported"""
        import sys
        from pathlib import Path

        src_dir = Path(__file__).parent.parent / "2026" / "src"
        sys.path.insert(0, str(src_dir))

        from commands import pipeline

        assert hasattr(pipeline, "run_full_pipeline")
