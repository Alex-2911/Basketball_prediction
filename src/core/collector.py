#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NBA Game Data Collectors

This module provides high-level classes for collecting NBA game data from
Basketball-Reference.com. It consolidates data collection logic for both
historical game statistics and upcoming game schedules.

Classes:
    HistoricalGameCollector: Scrapes and processes historical game box scores
    UpcomingGameCollector: Scrapes upcoming game schedules

Example:
    >>> from src.core.collector import HistoricalGameCollector
    >>> from datetime import date
    >>>
    >>> collector = HistoricalGameCollector(
    ...     season=2026,
    ...     standings_dir="data/2026_standings",
    ...     scores_dir="data/2026_scores"
    ... )
    >>> games_df = collector.collect_games_for_date(date(2025, 10, 22))
"""

import calendar
import os
import re
from datetime import date, datetime, timedelta
from io import StringIO
from typing import Dict, List, Optional, Tuple

import pandas as pd
from bs4 import BeautifulSoup

from src.utils.db_utils import DatabaseOperations, db_config
from src.utils.error_handlers import (
    DataValidationError,
    ErrorContext,
    NetworkError,
    ScrapingError,
    get_requests_session_with_retries,
    handle_missing_data,
    log_dataframe_info,
    retry_on_network_error,
    validate_dataframe,
    validate_file_exists,
)
from src.utils.logger import get_logger
from src.utils.nba_utils import (
    CURRENT_SEASON,
    get_html,
    parse_html,
    rename_duplicated_columns,
)

# Initialize logger
logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────
# UTILITY FUNCTIONS
# ─────────────────────────────────────────────────────────


def parse_ymd(date_str: str) -> date:
    """
    Parse date string in YYYY-MM-DD format.

    Args:
        date_str: Date string (e.g., "2025-10-22")

    Returns:
        date object

    Example:
        >>> parse_ymd("2025-10-22")
        date(2025, 10, 22)
    """
    return datetime.strptime(date_str, "%Y-%m-%d").date()


def month_name_lower(d: date) -> str:
    """
    Get lowercase month name from date.

    Args:
        d: Date object

    Returns:
        Lowercase month name (e.g., "october")

    Example:
        >>> month_name_lower(date(2025, 10, 22))
        'october'
    """
    return calendar.month_name[d.month].lower()


# ─────────────────────────────────────────────────────────
# HISTORICAL GAME COLLECTOR
# ─────────────────────────────────────────────────────────


class HistoricalGameCollector:
    """
    Collector for historical NBA game box scores.

    This class handles scraping historical game data from Basketball-Reference,
    including downloading monthly schedules, retrieving box score HTMLs, and
    parsing them into structured DataFrames.

    Attributes:
        season (int): NBA season year (e.g., 2026 for 2025-26 season)
        standings_dir (str): Directory for monthly schedule HTML files
        scores_dir (str): Directory for box score HTML files
        db_ops (DatabaseOperations): Database operations handler

    Example:
        >>> collector = HistoricalGameCollector(
        ...     season=2026,
        ...     standings_dir="data/2026_standings",
        ...     scores_dir="data/2026_scores"
        ... )
        >>>
        >>> # Collect games for a specific date
        >>> games_df = collector.collect_games_for_date(date(2025, 10, 22))
        >>>
        >>> # Save to database
        >>> collector.save_to_database(games_df)
    """

    def __init__(
        self,
        season: int = CURRENT_SEASON,
        standings_dir: Optional[str] = None,
        scores_dir: Optional[str] = None,
    ):
        """
        Initialize the Historical Game Collector.

        Args:
            season: NBA season year (default: CURRENT_SEASON from nba_utils)
            standings_dir: Path to store monthly schedule files
            scores_dir: Path to store box score HTML files
        """
        self.season = season
        self.standings_dir = standings_dir or f"data/{season}_standings"
        self.scores_dir = scores_dir or f"data/{season}_scores"
        self.db_ops = DatabaseOperations() if db_config.enabled else None

        # Ensure directories exist
        os.makedirs(self.standings_dir, exist_ok=True)
        os.makedirs(self.scores_dir, exist_ok=True)

        logger.info(f"HistoricalGameCollector initialized for season {season}")
        logger.debug(f"  - Standings directory: {self.standings_dir}")
        logger.debug(f"  - Scores directory: {self.scores_dir}")

    # ─────────────────────────────────────────────────────
    # HTML PARSING HELPERS
    # ─────────────────────────────────────────────────────

    @staticmethod
    def read_line_score(soup: BeautifulSoup) -> pd.DataFrame:
        """
        Extract line score table from box score HTML.

        Args:
            soup: BeautifulSoup object of box score page

        Returns:
            DataFrame with columns ['team', 'total']

        Example:
            >>> soup = parse_html("data/202510220LAL.html")
            >>> line_score = HistoricalGameCollector.read_line_score(soup)
            >>> print(line_score)
                        team  total
            0  Los Angeles Lakers    110
            1  Golden State Warriors    105
        """
        line_score = pd.read_html(StringIO(str(soup)), attrs={"id": "line_score"})[0]
        cols = list(line_score.columns)
        cols[0] = "team"
        cols[-1] = "total"
        line_score.columns = cols
        line_score = line_score[["team", "total"]]
        return line_score

    @staticmethod
    def read_stats(soup: BeautifulSoup, team: str, stat: str) -> pd.DataFrame:
        """
        Extract team statistics table from box score HTML.

        Args:
            soup: BeautifulSoup object of box score page
            team: Team abbreviation (e.g., "LAL")
            stat: Stat type ("basic" or "advanced")

        Returns:
            DataFrame with numeric statistics

        Example:
            >>> soup = parse_html("data/202510220LAL.html")
            >>> basic_stats = HistoricalGameCollector.read_stats(soup, "LAL", "basic")
        """
        df = pd.read_html(
            StringIO(str(soup)), attrs={"id": f"box-{team}-game-{stat}"}, index_col=0
        )[0]
        return df.apply(pd.to_numeric, errors="coerce")

    @staticmethod
    def read_season_info(soup: BeautifulSoup) -> str:
        """
        Extract season year from box score HTML.

        Args:
            soup: BeautifulSoup object of box score page

        Returns:
            Season year as string (e.g., "2026")

        Example:
            >>> soup = parse_html("data/202510220LAL.html")
            >>> season = HistoricalGameCollector.read_season_info(soup)
            >>> print(season)
            '2026'
        """
        nav = soup.select("#bottom_nav_container")[0]
        hrefs = [a["href"] for a in nav.find_all("a")]
        return os.path.basename(hrefs[1]).split("_")[0]

    # ─────────────────────────────────────────────────────
    # SCRAPING METHODS
    # ─────────────────────────────────────────────────────

    def scrape_monthly_schedule(self, month_name: str, force_refresh: bool = True) -> Optional[str]:
        """
        Download monthly schedule HTML from Basketball-Reference.

        This method always fetches fresh data by default, deleting any existing
        monthly file before downloading.

        Args:
            month_name: Lowercase month name (e.g., "october")
            force_refresh: If True, delete existing file and re-download

        Returns:
            Path to saved monthly schedule file, or None if scraping failed

        Raises:
            ScrapingError: If unable to fetch or parse schedule

        Example:
            >>> collector = HistoricalGameCollector(season=2026)
            >>> path = collector.scrape_monthly_schedule("october")
            >>> print(path)
            'data/2026_standings/NBA_2026_games-october.html'
        """
        with ErrorContext(f"Scraping {month_name} {self.season} schedule", logger=logger):
            monthly_filename = f"NBA_{self.season}_games-{month_name}.html"
            monthly_path = os.path.join(self.standings_dir, monthly_filename)

            # Delete existing file to ensure fresh data
            if force_refresh and os.path.exists(monthly_path):
                try:
                    os.remove(monthly_path)
                    logger.info(f"Deleted outdated monthly file: {monthly_path}")
                except Exception as e:
                    logger.warning(f"Could not delete {monthly_path}: {e}")

            # Fetch season landing page to discover month URLs
            season_url = f"https://www.basketball-reference.com/leagues/NBA_{self.season}_games.html"
            selector = "#content .filter"

            logger.info(f"Fetching season page: {season_url}")
            html_content = get_html(season_url, selector)
            if not html_content:
                logger.error(f"Failed to retrieve {season_url}")
                raise ScrapingError(f"Could not fetch season page: {season_url}")

            # Parse and find monthly link
            soup = BeautifulSoup(html_content, "html.parser")
            links = soup.find_all("a", href=re.compile(r"/leagues/NBA_[0-9]{4}_games-[a-z]+\.html"))

            wanted_url = None
            for link in links:
                href = link.get("href", "")
                if f"NBA_{self.season}_games-{month_name}" in href:
                    wanted_url = "https://www.basketball-reference.com" + href
                    break

            if not wanted_url:
                logger.warning(f"No monthly URL found for '{month_name}' in season {self.season}")
                return None

            # Fetch monthly schedule
            logger.info(f"Fetching fresh month page: {wanted_url}")
            month_html = get_html(wanted_url, "#all_schedule")
            if not month_html:
                logger.warning(f"Could not fetch monthly page: {wanted_url}")
                return None

            # Save to file
            try:
                with open(monthly_path, "w", encoding="utf-8") as f:
                    f.write(month_html)
                logger.info(f"Saved fresh monthly file → {monthly_path}")
            except Exception as e:
                logger.error(f"Error saving {monthly_path}: {e}")
                return None

            return monthly_path

    def scrape_boxscores_for_date(
        self, monthly_schedule_file: str, target_date: date
    ) -> int:
        """
        Download box score HTMLs for a specific game date.

        Parses the monthly schedule HTML to find box score links for the target
        date, then downloads any missing box score files.

        Args:
            monthly_schedule_file: Path to monthly schedule HTML
            target_date: Date to collect games for

        Returns:
            Number of new box score files downloaded

        Example:
            >>> collector = HistoricalGameCollector(season=2026)
            >>> schedule_file = "data/2026_standings/NBA_2026_games-october.html"
            >>> count = collector.scrape_boxscores_for_date(
            ...     schedule_file,
            ...     date(2025, 10, 22)
            ... )
            >>> print(f"Downloaded {count} new box scores")
        """
        with ErrorContext(f"Scraping box scores for {target_date}", logger=logger):
            # Read monthly schedule
            with open(monthly_schedule_file, "r", encoding="utf-8") as f:
                html = f.read()
            soup = BeautifulSoup(html, "html.parser")
            hrefs = [a.get("href") for a in soup.find_all("a")]

            # Filter for target date box scores
            wanted_tag = target_date.strftime("%Y%m%d")
            box_score_urls = [
                "https://www.basketball-reference.com" + h
                for h in hrefs
                if h and "boxscores" in h and h.endswith(".html") and wanted_tag in h
            ]

            logger.info(f"Found {len(box_score_urls)} box score URLs for {target_date}")

            # Download missing box scores
            saved = 0
            for url in box_score_urls:
                save_path = os.path.join(self.scores_dir, os.path.basename(url))
                if os.path.exists(save_path):
                    logger.debug(f"Box score already exists: {save_path}")
                    continue

                page_html = get_html(url, "#content")
                if not page_html:
                    logger.warning(f"Failed to fetch box score: {url}")
                    continue

                with open(save_path, "wb") as f:
                    f.write(page_html.encode("utf-8"))
                saved += 1
                logger.info(f"Saved box score → {save_path}")

            return saved

    def process_boxscores_for_date(
        self, target_date: date, existing_statistics: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Parse all box score HTMLs for a specific date into DataFrame.

        Reads downloaded box score files, extracts team statistics, and builds
        game-level rows with home/away splits and opponent stats.

        Args:
            target_date: Date to process games for
            existing_statistics: Reference DataFrame for column alignment

        Returns:
            DataFrame with processed game statistics

        Raises:
            DataValidationError: If data processing fails

        Example:
            >>> collector = HistoricalGameCollector(season=2026)
            >>> games_df = collector.process_boxscores_for_date(date(2025, 10, 22))
            >>> print(games_df.shape)
            (6, 247)  # 3 games × 2 teams = 6 rows
        """
        with ErrorContext(f"Processing box scores for {target_date}", logger=logger):
            box_files = [
                os.path.join(self.scores_dir, f)
                for f in os.listdir(self.scores_dir)
                if f.endswith(".html")
            ]

            if not box_files:
                logger.warning("No box score files found.")
                return pd.DataFrame()

            games = []
            base_cols = None

            for path in box_files:
                try:
                    # Parse date from filename
                    file_date = pd.Timestamp(os.path.basename(path)[:8]).date()
                    if file_date != target_date:
                        continue

                    # Parse HTML
                    soup = parse_html(path)
                    if soup is None:
                        logger.warning(f"Failed to parse HTML: {path}")
                        continue

                    # Extract line score
                    line_score = self.read_line_score(soup)
                    teams = list(line_score["team"])

                    # Process each team's statistics
                    summaries = []
                    for team in teams:
                        basic = self.read_stats(soup, team, "basic")
                        advanced = self.read_stats(soup, team, "advanced")

                        # Team totals (last row)
                        totals = pd.concat([basic.iloc[-1], advanced.iloc[-1]])
                        totals.index = totals.index.str.lower()

                        # Player maximums
                        maxes = pd.concat([basic.iloc[:-1].max(), advanced.iloc[:-1].max()])
                        maxes.index = maxes.index.str.lower() + "_max"

                        # Combine totals and maxes
                        summary = pd.concat([totals, maxes])

                        # Establish base columns on first iteration
                        if base_cols is None:
                            base_cols = [
                                b
                                for b in summary.index.drop_duplicates(keep="first")
                                if "bpm" not in b
                            ]
                        summary = summary[base_cols]
                        summaries.append(summary)

                    # Combine team summaries
                    summary = pd.concat(summaries, axis=1).T

                    # Add line score and home indicator
                    game = pd.concat([summary, line_score], axis=1)
                    game["home"] = [0, 1]

                    # Create opponent columns
                    game_opp = game.iloc[::-1].reset_index()
                    game_opp.columns += "_opp"

                    # Combine home/away with opponent stats
                    full_game = pd.concat([game, game_opp], axis=1)
                    full_game["season"] = self.read_season_info(soup)
                    full_game["date"] = pd.Timestamp(os.path.basename(path)[:8])
                    full_game["won"] = full_game["total"] > full_game["total_opp"]

                    games.append(full_game)

                except Exception as e:
                    logger.error(f"Error processing {path}: {e}", exc_info=True)

            if not games:
                logger.warning(f"No games parsed for {target_date}")
                return pd.DataFrame()

            # Combine all games
            games_df = pd.concat(games, ignore_index=True)
            games_df = rename_duplicated_columns(games_df)

            # Align columns with existing statistics
            if existing_statistics is not None and not existing_statistics.empty:
                games_df = games_df.reindex(columns=existing_statistics.columns)

            logger.info(f"Processed {len(games_df)} game rows for {target_date}")
            return games_df

    # ─────────────────────────────────────────────────────
    # HIGH-LEVEL COLLECTION METHODS
    # ─────────────────────────────────────────────────────

    def collect_games_for_date(
        self,
        target_date: date,
        existing_statistics: Optional[pd.DataFrame] = None,
        force_refresh: bool = True,
    ) -> pd.DataFrame:
        """
        Complete workflow to collect and process games for a date.

        This is the main entry point that orchestrates:
        1. Download monthly schedule (with fresh data)
        2. Download box score HTMLs
        3. Parse and process into DataFrame

        Args:
            target_date: Date to collect games for
            existing_statistics: Reference DataFrame for column structure
            force_refresh: Delete and re-download monthly schedule

        Returns:
            DataFrame with game statistics

        Example:
            >>> collector = HistoricalGameCollector(season=2026)
            >>> games_df = collector.collect_games_for_date(date(2025, 10, 22))
            >>> print(f"Collected {len(games_df)} games")
        """
        with ErrorContext(f"Collecting games for {target_date}", logger=logger):
            # Step 1: Get monthly schedule
            month_name = month_name_lower(target_date)
            monthly_file = self.scrape_monthly_schedule(month_name, force_refresh=force_refresh)

            if monthly_file is None:
                # Fall back to existing file if available
                monthly_file = os.path.join(
                    self.standings_dir, f"NBA_{self.season}_games-{month_name}.html"
                )
                if not os.path.exists(monthly_file):
                    logger.error(f"No monthly schedule available for {month_name}")
                    return pd.DataFrame()

            # Step 2: Download box scores
            saved_count = self.scrape_boxscores_for_date(monthly_file, target_date)
            logger.info(f"Downloaded {saved_count} new box score(s) for {target_date}")

            # Step 3: Process box scores
            games_df = self.process_boxscores_for_date(target_date, existing_statistics)

            # Validate results
            if not games_df.empty:
                log_dataframe_info(games_df, name=f"Games for {target_date}", logger=logger)

            return games_df

    # ─────────────────────────────────────────────────────
    # DATABASE INTEGRATION
    # ─────────────────────────────────────────────────────

    def save_to_database(self, df: pd.DataFrame) -> int:
        """
        Save game statistics to database.

        Args:
            df: DataFrame with game statistics

        Returns:
            Number of rows saved to database

        Example:
            >>> games_df = collector.collect_games_for_date(date(2025, 10, 22))
            >>> rows_saved = collector.save_to_database(games_df)
            >>> print(f"Saved {rows_saved} rows to database")
        """
        if not db_config.enabled:
            logger.info("Database not enabled, skipping save")
            return 0

        if df.empty:
            logger.warning("Empty DataFrame, nothing to save")
            return 0

        try:
            rows_saved = self.db_ops.save_game_statistics(df)
            logger.info(f"Saved {rows_saved} game statistics to database")
            return rows_saved
        except Exception as e:
            logger.error(f"Failed to save to database: {e}", exc_info=True)
            return 0


# ─────────────────────────────────────────────────────────
# UPCOMING GAME COLLECTOR
# ─────────────────────────────────────────────────────────


class UpcomingGameCollector:
    """
    Collector for upcoming NBA game schedules.

    This class handles scraping upcoming game schedules from Basketball-Reference,
    finding the next game day after a specified date, and formatting the results
    as a structured DataFrame.

    Attributes:
        season (int): NBA season year (e.g., 2026 for 2025-26 season)
        standings_dir (str): Directory for monthly schedule HTML files
        db_ops (DatabaseOperations): Database operations handler

    Example:
        >>> collector = UpcomingGameCollector(
        ...     season=2026,
        ...     standings_dir="data/2026_standings"
        ... )
        >>>
        >>> # Find next games after a specific date
        >>> games_df = collector.find_next_games(date(2025, 10, 21))
        >>> print(games_df)
           home_team away_team  game_date
        0        OKC       HOU 2025-10-22
        1        LAL       GSW 2025-10-22
        >>>
        >>> # Save to database
        >>> collector.save_to_database(games_df)
    """

    def __init__(
        self,
        season: Optional[int] = None,
        standings_dir: Optional[str] = None,
    ):
        """
        Initialize the Upcoming Game Collector.

        Args:
            season: NBA season year (defaults to CURRENT_SEASON or env SEASON_YEAR)
            standings_dir: Path to store monthly schedule files
        """
        # Determine season (env var → explicit param → CURRENT_SEASON)
        self.season = self.determine_season_year(season or CURRENT_SEASON)
        self.standings_dir = standings_dir or f"data/{self.season}_standings"
        self.db_ops = DatabaseOperations() if db_config.enabled else None

        # Ensure directory exists
        os.makedirs(self.standings_dir, exist_ok=True)

        logger.info(f"UpcomingGameCollector initialized for season {self.season}")
        logger.debug(f"  - Standings directory: {self.standings_dir}")

    # ─────────────────────────────────────────────────────
    # SEASON YEAR DETERMINATION
    # ─────────────────────────────────────────────────────

    @staticmethod
    def determine_season_year(fallback_season: int) -> int:
        """
        Determine NBA season year, with optional environment override.

        Basketball-Reference labels seasons by the year they conclude.
        For example, 2025-26 season is labeled as 2026.

        Args:
            fallback_season: Default season year if no override

        Returns:
            Season year to use for scraping

        Example:
            >>> # With SEASON_YEAR=2027 environment variable
            >>> season = UpcomingGameCollector.determine_season_year(2026)
            >>> print(season)
            2027
            >>>
            >>> # Without environment variable
            >>> season = UpcomingGameCollector.determine_season_year(2026)
            >>> print(season)
            2026
        """
        env_season = os.getenv("SEASON_YEAR")
        if env_season is not None:
            try:
                season = int(env_season)
                logger.info(f"Using season year from SEASON_YEAR env: {season}")
                return season
            except ValueError:
                logger.warning(
                    f"Invalid SEASON_YEAR '{env_season}', falling back to {fallback_season}"
                )
        return fallback_season

    # ─────────────────────────────────────────────────────
    # SCRAPING METHODS
    # ─────────────────────────────────────────────────────

    @retry_on_network_error(max_retries=4, backoff_factor=2.0)
    def scrape_monthly_schedule(self, month: int, month_name: str) -> None:
        """
        Scrape monthly NBA schedule from Basketball-Reference.

        Downloads the monthly games page if it doesn't exist locally.
        Uses retry decorator for resilient network operations.

        Args:
            month: Month number (1-12)
            month_name: Lowercase month name (e.g., "october")

        Raises:
            NetworkError: If scraping fails after retries
            ScrapingError: If HTML parsing fails

        Example:
            >>> collector = UpcomingGameCollector(season=2026)
            >>> collector.scrape_monthly_schedule(10, "october")
            # Saves to: data/2026_standings/NBA_2026_games-october.html
        """
        with ErrorContext(f"Scraping {month_name.title()} {self.season} schedule", logger=logger):
            season_url = f"https://www.basketball-reference.com/leagues/NBA_{self.season}_games.html"

            # Use session with automatic retries
            session = get_requests_session_with_retries()

            logger.info(f"Fetching season page: {season_url}")
            response = session.get(season_url, timeout=30)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")
            links = soup.find_all("a", href=True)

            # Find monthly schedule link
            month_link: Optional[str] = None
            for link in links:
                if f"NBA_{self.season}_games-{month_name}" in link["href"]:
                    month_link = "https://www.basketball-reference.com" + link["href"]
                    break

            if not month_link:
                logger.warning(f"No link found for {month_name.title()} {self.season}")
                raise ScrapingError(f"Could not find monthly schedule link for {month_name} {self.season}")

            # Fetch monthly schedule
            logger.info(f"Fetching monthly page: {month_link}")
            month_response = session.get(month_link, timeout=30)
            month_response.raise_for_status()

            # Save HTML
            output_path = os.path.join(
                self.standings_dir, f"NBA_{self.season}_games-{month_name}.html"
            )
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(month_response.text)

            logger.info(f"Saved schedule data → {output_path}")

    def find_next_game_day(
        self, target_date: datetime, file_paths: List[str]
    ) -> List[Dict[str, any]]:
        """
        Find games scheduled on the next game day after target date.

        Parses monthly schedule HTML files to find the first game day on or
        after the target date, then returns all games scheduled that day.

        Args:
            target_date: Starting point for search (inclusive)
            file_paths: List of monthly schedule HTML files to search

        Returns:
            List of game dictionaries with keys: date, home_team, visitor_team

        Raises:
            ScrapingError: If HTML parsing fails critically

        Example:
            >>> collector = UpcomingGameCollector(season=2026)
            >>> from datetime import datetime
            >>> target = datetime(2025, 10, 21)
            >>> files = ["data/2026_standings/NBA_2026_games-october.html"]
            >>> games = collector.find_next_game_day(target, files)
            >>> print(games)
            [
                {
                    'date': datetime(2025, 10, 22),
                    'home_team': 'Oklahoma City Thunder',
                    'visitor_team': 'Houston Rockets'
                },
                ...
            ]
        """
        next_game_info: List[Dict[str, any]] = []
        next_game_date: Optional[datetime] = None

        logger.info(f"Searching for next game day after {target_date.strftime('%Y-%m-%d')}")

        for path in file_paths:
            if not os.path.exists(path):
                logger.debug(f"Schedule file not found: {path}")
                continue

            try:
                with ErrorContext(
                    f"Parsing schedule file: {path}", logger=logger, raise_on_error=False
                ):
                    with open(path, "r", encoding="utf-8") as f:
                        soup = BeautifulSoup(f.read(), "html.parser")

                    table = soup.find("table", {"id": "schedule"})
                    if not table:
                        logger.warning(f"No schedule table found in {path}")
                        continue

                    rows = table.find_all("tr")
                    logger.debug(f"Found {len(rows)} rows in schedule table")

                    # Iterate over schedule rows (skip header)
                    for row in rows[1:]:
                        date_cell = row.find("th", {"data-stat": "date_game"})
                        if not date_cell:
                            continue

                        date_str = date_cell.text.strip()
                        try:
                            game_date = datetime.strptime(date_str, "%a, %b %d, %Y")
                        except ValueError as e:
                            logger.debug(f"Could not parse date '{date_str}': {e}")
                            continue

                        # Identify next game day on or after target
                        if game_date >= target_date and next_game_date is None:
                            next_game_date = game_date
                            logger.info(f"Found next game day: {game_date.strftime('%Y-%m-%d')}")

                        # Collect all games on that date
                        if next_game_date is not None and game_date == next_game_date:
                            cols = row.find_all("td")
                            if len(cols) >= 4:
                                next_game_info.append(
                                    {
                                        "date": game_date,
                                        "home_team": cols[3].text.strip(),
                                        "visitor_team": cols[1].text.strip(),
                                    }
                                )

                        # Stop once we've moved past the target game day
                        if next_game_date is not None and game_date > next_game_date:
                            logger.info(
                                f"Found {len(next_game_info)} games on "
                                f"{next_game_date.strftime('%Y-%m-%d')}"
                            )
                            return next_game_info

            except Exception as e:
                logger.error(f"Error reading schedule from {path}: {e}", exc_info=True)
                continue

        if next_game_info:
            logger.info(
                f"Found {len(next_game_info)} games on {next_game_date.strftime('%Y-%m-%d')}"
            )
        else:
            logger.warning(f"No games found after {target_date.strftime('%Y-%m-%d')}")

        return next_game_info

    # ─────────────────────────────────────────────────────
    # HIGH-LEVEL COLLECTION METHODS
    # ─────────────────────────────────────────────────────

    def find_next_games(
        self,
        target_date: datetime,
        team_codes: Optional[Dict[str, str]] = None,
    ) -> pd.DataFrame:
        """
        Find all games scheduled on the next game day.

        This is the main entry point that:
        1. Ensures monthly schedules are downloaded
        2. Searches for next game day
        3. Maps team names to codes
        4. Returns formatted DataFrame

        Args:
            target_date: Date to start searching from
            team_codes: Optional dict mapping full names to abbreviations

        Returns:
            DataFrame with columns: home_team, away_team, game_date

        Example:
            >>> from datetime import datetime
            >>> collector = UpcomingGameCollector(season=2026)
            >>> games_df = collector.find_next_games(datetime(2025, 10, 21))
            >>> print(games_df)
               home_team away_team  game_date
            0        OKC       HOU 2025-10-22
            1        LAL       GSW 2025-10-22
        """
        with ErrorContext(f"Finding next games after {target_date}", logger=logger):
            # Get team codes for mapping
            if team_codes is None:
                from src.utils.nba_utils import get_team_codes
                team_codes = get_team_codes()

            # Determine target month
            month_num = target_date.month
            month_name = calendar.month_name[month_num].lower()
            logger.info(f"Target month: {month_name}")

            # Ensure monthly schedule exists
            html_path = os.path.join(
                self.standings_dir, f"NBA_{self.season}_games-{month_name}.html"
            )

            if not os.path.exists(html_path):
                logger.info(f"Scraping {month_name} schedule...")
                try:
                    self.scrape_monthly_schedule(month_num, month_name)
                except (NetworkError, ScrapingError) as e:
                    logger.error(f"Failed to scrape {month_name}: {e}")
                    return pd.DataFrame(columns=["home_team", "away_team", "game_date"])

            # Find next game day
            games_info = self.find_next_game_day(target_date, [html_path])

            # If no games in current month, check subsequent months
            if not games_info:
                logger.info("No games in current month, checking subsequent months...")
                next_month = (month_num % 12) + 1
                months_checked = 0

                while not games_info and months_checked < 12:
                    next_month_name = calendar.month_name[next_month].lower()
                    next_html = os.path.join(
                        self.standings_dir, f"NBA_{self.season}_games-{next_month_name}.html"
                    )

                    if not os.path.exists(next_html):
                        logger.info(f"Scraping {next_month_name} schedule...")
                        try:
                            self.scrape_monthly_schedule(next_month, next_month_name)
                        except (NetworkError, ScrapingError) as e:
                            logger.warning(f"Could not scrape {next_month_name}: {e}")
                            next_month = (next_month % 12) + 1
                            months_checked += 1
                            continue

                    games_info = self.find_next_game_day(target_date, [next_html])
                    next_month = (next_month % 12) + 1
                    months_checked += 1

            # Convert to DataFrame
            games_data = []
            if games_info:
                for game in games_info:
                    home_code = team_codes.get(game["home_team"], game["home_team"])
                    away_code = team_codes.get(game["visitor_team"], game["visitor_team"])
                    games_data.append(
                        (home_code, away_code, game["date"].strftime("%Y-%m-%d"))
                    )
                    logger.info(
                        f"Scheduled: {game['visitor_team']} @ {game['home_team']} "
                        f"on {game['date'].strftime('%Y-%m-%d')}"
                    )

            df = pd.DataFrame(games_data, columns=["home_team", "away_team", "game_date"])

            # Validate DataFrame
            try:
                validate_dataframe(
                    df,
                    required_columns=["home_team", "away_team", "game_date"],
                    allow_empty=True,
                )
                log_dataframe_info(df, name="Next games", logger=logger)
            except Exception as e:
                logger.warning(f"DataFrame validation warning: {e}")

            return df

    # ─────────────────────────────────────────────────────
    # DATABASE INTEGRATION
    # ─────────────────────────────────────────────────────

    def save_to_database(self, df: pd.DataFrame) -> int:
        """
        Save game schedule to database.

        Args:
            df: DataFrame with upcoming games

        Returns:
            Number of rows saved to database

        Example:
            >>> games_df = collector.find_next_games(datetime(2025, 10, 21))
            >>> rows_saved = collector.save_to_database(games_df)
            >>> print(f"Saved {rows_saved} rows to database")
        """
        if not db_config.enabled:
            logger.info("Database not enabled, skipping save")
            return 0

        if df.empty:
            logger.warning("Empty DataFrame, nothing to save")
            return 0

        try:
            rows_saved = self.db_ops.save_game_schedule(df)
            logger.info(f"Saved {rows_saved} game schedules to database")
            return rows_saved
        except Exception as e:
            logger.error(f"Failed to save to database: {e}", exc_info=True)
            return 0


# ─────────────────────────────────────────────────────────
# MODULE TEST / EXAMPLE USAGE
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    from datetime import datetime

    # Example 1: Historical Game Collector
    print("=" * 60)
    print("HISTORICAL GAME COLLECTOR EXAMPLE")
    print("=" * 60)

    historical = HistoricalGameCollector(
        season=2026,
        standings_dir="data/2026_standings",
        scores_dir="data/2026_scores",
    )

    # Collect games for October 22, 2025
    target = date(2025, 10, 22)
    games = historical.collect_games_for_date(target)
    print(f"\nCollected {len(games)} game rows for {target}")
    if not games.empty:
        print(f"Columns: {list(games.columns[:10])}...")

    # Example 2: Upcoming Game Collector
    print("\n" + "=" * 60)
    print("UPCOMING GAME COLLECTOR EXAMPLE")
    print("=" * 60)

    upcoming = UpcomingGameCollector(
        season=2026,
        standings_dir="data/2026_standings",
    )

    # Find next games after October 21, 2025
    target_dt = datetime(2025, 10, 21)
    next_games = upcoming.find_next_games(target_dt)
    print(f"\nFound {len(next_games)} upcoming games")
    if not next_games.empty:
        print(next_games.to_string())

    print("\n" + "=" * 60)
    print("Examples completed successfully!")
    print("=" * 60)
