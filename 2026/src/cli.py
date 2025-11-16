#!/usr/bin/env python
"""
NBA Prediction System - Command Line Interface

A modern CLI for running NBA game predictions, data collection, and betting analysis.

Usage:
    nba-predict collect historical      # Scrape previous game data
    nba-predict collect upcoming        # Get upcoming game schedule
    nba-predict predict                 # Generate predictions
    nba-predict analyze stats           # Calculate betting statistics
    nba-predict analyze kelly           # Calculate Kelly Criterion parameters
    nba-predict analyze recommend       # Show betting recommendations
    nba-predict analyze all             # Run all analysis steps
    nba-predict pipeline                # Run complete pipeline (all steps)
    nba-predict dashboard               # Launch interactive dashboard

Examples:
    # Run full pipeline
    nba-predict pipeline

    # Run only data collection
    nba-predict collect historical
    nba-predict collect upcoming

    # Generate predictions only
    nba-predict predict

    # Run analysis steps
    nba-predict analyze all
"""

import sys
from pathlib import Path

import click

# Add src directory to Python path
SRC_DIR = Path(__file__).parent
sys.path.insert(0, str(SRC_DIR))


@click.group()
@click.version_option(version="2026.1.0", prog_name="nba-predict")
def cli():
    """
    NBA Prediction System - Advanced sports betting analytics

    Built with machine learning (LightGBM), web scraping, and Kelly Criterion
    for optimal bet sizing on NBA games.
    """
    pass


@cli.group()
def collect():
    """Collect NBA game data (historical and upcoming)"""
    pass


@collect.command(name="historical")
@click.option(
    "--date",
    type=str,
    help="Anchor date in YYYY-MM-DD; collects games from the day before",
)
@click.option(
    "--collect-date",
    type=str,
    help="Exact game date to collect in YYYY-MM-DD (overrides --date)",
)
def collect_historical(date, collect_date):
    """Scrape historical game data from Basketball Reference"""
    from commands.collect import run_historical_collection

    click.echo("📊 Collecting historical game data...")
    success = run_historical_collection(date=date, collect_date=collect_date)

    if success:
        click.secho("✓ Historical data collection completed", fg="green")
    else:
        click.secho("✗ Historical data collection failed", fg="red")
        sys.exit(1)


@collect.command(name="upcoming")
@click.option(
    "--date",
    type=str,
    help="Date for which to collect upcoming games (YYYY-MM-DD)",
)
def collect_upcoming(date):
    """Get upcoming game schedule and betting odds"""
    from commands.collect import run_upcoming_collection

    click.echo("📅 Collecting upcoming game schedule...")
    success = run_upcoming_collection(date=date)

    if success:
        click.secho("✓ Upcoming games collection completed", fg="green")
    else:
        click.secho("✗ Upcoming games collection failed", fg="red")
        sys.exit(1)


@cli.command()
@click.option(
    "--model-path",
    type=click.Path(exists=True),
    help="Path to saved LightGBM model (optional)",
)
@click.option(
    "--output-dir",
    type=click.Path(),
    help="Directory for prediction output files",
)
def predict(model_path, output_dir):
    """Generate predictions for upcoming games using LightGBM"""
    from commands.predict import run_prediction

    click.echo("🔮 Generating predictions...")
    success = run_prediction(model_path=model_path, output_dir=output_dir)

    if success:
        click.secho("✓ Predictions generated successfully", fg="green")
    else:
        click.secho("✗ Prediction generation failed", fg="red")
        sys.exit(1)


@cli.group()
def analyze():
    """Run betting analysis (statistics, Kelly Criterion, recommendations)"""
    pass


@analyze.command(name="stats")
def analyze_stats():
    """Calculate betting statistics from historical predictions"""
    from commands.analyze import run_statistics

    click.echo("📈 Calculating betting statistics...")
    success = run_statistics()

    if success:
        click.secho("✓ Statistics calculated successfully", fg="green")
    else:
        click.secho("✗ Statistics calculation failed", fg="red")
        sys.exit(1)


@analyze.command(name="kelly")
def analyze_kelly():
    """Calculate Kelly Criterion betting parameters"""
    from commands.analyze import run_kelly

    click.echo("💰 Calculating Kelly Criterion parameters...")
    success = run_kelly()

    if success:
        click.secho("✓ Kelly parameters calculated successfully", fg="green")
    else:
        click.secho("✗ Kelly calculation failed", fg="red")
        sys.exit(1)


@analyze.command(name="recommend")
def analyze_recommend():
    """Show betting recommendations with optimal stakes"""
    from commands.analyze import run_recommendations

    click.echo("🎯 Generating betting recommendations...")
    success = run_recommendations()

    if success:
        click.secho("✓ Recommendations generated successfully", fg="green")
    else:
        click.secho("✗ Recommendation generation failed", fg="red")
        sys.exit(1)


@analyze.command(name="all")
def analyze_all():
    """Run all analysis steps (stats → kelly → recommend)"""
    from commands.analyze import run_all_analysis

    click.echo("🔍 Running complete betting analysis...")
    success = run_all_analysis()

    if success:
        click.secho("✓ Complete analysis finished successfully", fg="green")
    else:
        click.secho("✗ Analysis failed", fg="red")
        sys.exit(1)


@cli.command()
@click.option(
    "--skip-collection",
    is_flag=True,
    help="Skip data collection steps (use existing data)",
)
@click.option(
    "--skip-analysis",
    is_flag=True,
    help="Skip analysis steps (only collect and predict)",
)
def pipeline(skip_collection, skip_analysis):
    """
    Run the complete NBA prediction pipeline

    Pipeline steps:
    1. Collect historical game data
    2. Collect upcoming game schedule
    3. Generate predictions
    4. Calculate statistics
    5. Calculate Kelly parameters
    6. Generate recommendations
    """
    from commands.pipeline import run_full_pipeline

    click.echo("🚀 Starting NBA Prediction Pipeline...")
    click.echo("=" * 60)

    success = run_full_pipeline(
        skip_collection=skip_collection,
        skip_analysis=skip_analysis
    )

    click.echo("=" * 60)
    if success:
        click.secho("✓ Pipeline completed successfully!", fg="green", bold=True)
    else:
        click.secho("✗ Pipeline failed", fg="red", bold=True)
        sys.exit(1)


@cli.command()
@click.option(
    "--port",
    default=8501,
    type=int,
    help="Port for Streamlit dashboard (default: 8501)",
)
def dashboard(port):
    """Launch the interactive Streamlit dashboard"""
    import subprocess

    dashboard_path = SRC_DIR.parent.parent / "dashboard" / "app.py"

    if not dashboard_path.exists():
        click.secho("✗ Dashboard not found at: " + str(dashboard_path), fg="red")
        sys.exit(1)

    click.echo(f"🎨 Launching dashboard on http://localhost:{port}")
    click.echo("Press Ctrl+C to stop the dashboard")

    try:
        subprocess.run([
            "streamlit",
            "run",
            str(dashboard_path),
            "--server.port",
            str(port),
            "--server.address",
            "localhost"
        ])
    except KeyboardInterrupt:
        click.echo("\n👋 Dashboard stopped")
    except FileNotFoundError:
        click.secho(
            "✗ Streamlit not found. Install with: pip install streamlit",
            fg="red"
        )
        sys.exit(1)


def main():
    """Entry point for the CLI"""
    cli()


if __name__ == "__main__":
    main()
