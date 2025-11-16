#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Database utilities for Basketball Prediction project.

Provides connection management and CRUD operations for Supabase/PostgreSQL.
Falls back to CSV operations if database is not configured.
"""

import os
from typing import Optional, Dict, List, Any, Tuple
from datetime import datetime, date
import pandas as pd
import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor
from contextlib import contextmanager

from logger import get_logger
from error_handlers import (
    ErrorContext,
    validate_dataframe,
    ConfigurationError,
    DataValidationError,
)

logger = get_logger(__name__)

# ─────────────────────────────────────────────────────────
# DATABASE CONFIGURATION
# ─────────────────────────────────────────────────────────

class DatabaseConfig:
    """Database configuration from environment variables."""

    def __init__(self):
        self.enabled = os.getenv("USE_DATABASE", "false").lower() == "true"

        if self.enabled:
            # Supabase connection string format:
            # postgresql://[user]:[password]@[host]:[port]/[database]
            self.connection_string = os.getenv("DATABASE_URL")

            # Or individual components
            self.host = os.getenv("DB_HOST", "localhost")
            self.port = int(os.getenv("DB_PORT", "5432"))
            self.database = os.getenv("DB_NAME", "basketball_predictions")
            self.user = os.getenv("DB_USER", "postgres")
            self.password = os.getenv("DB_PASSWORD", "")

            # Connection pool settings
            self.min_connections = int(os.getenv("DB_MIN_CONNECTIONS", "1"))
            self.max_connections = int(os.getenv("DB_MAX_CONNECTIONS", "10"))

    def get_connection_params(self) -> Dict[str, Any]:
        """Get connection parameters as dictionary."""
        if self.connection_string:
            return {"dsn": self.connection_string}
        else:
            return {
                "host": self.host,
                "port": self.port,
                "database": self.database,
                "user": self.user,
                "password": self.password,
            }

    def validate(self):
        """Validate database configuration."""
        if not self.enabled:
            return

        if not self.connection_string:
            if not all([self.host, self.database, self.user, self.password]):
                raise ConfigurationError(
                    "Database configuration incomplete. Provide either DATABASE_URL "
                    "or DB_HOST, DB_NAME, DB_USER, and DB_PASSWORD environment variables."
                )


# Global configuration instance
db_config = DatabaseConfig()

# ─────────────────────────────────────────────────────────
# CONNECTION POOL
# ─────────────────────────────────────────────────────────

class DatabasePool:
    """Manages PostgreSQL connection pool."""

    _instance = None
    _pool = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def initialize(self):
        """Initialize connection pool."""
        if not db_config.enabled:
            logger.info("Database disabled, using CSV fallback mode")
            return

        if self._pool is not None:
            logger.warning("Connection pool already initialized")
            return

        with ErrorContext("Initializing database connection pool", logger=logger):
            try:
                db_config.validate()
                conn_params = db_config.get_connection_params()

                self._pool = psycopg2.pool.ThreadedConnectionPool(
                    db_config.min_connections,
                    db_config.max_connections,
                    **conn_params
                )

                logger.info(
                    f"Database connection pool initialized "
                    f"({db_config.min_connections}-{db_config.max_connections} connections)"
                )

                # Test connection
                with self.get_connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute("SELECT version();")
                        version = cur.fetchone()[0]
                        logger.info(f"Connected to PostgreSQL: {version}")

            except Exception as e:
                logger.error(f"Failed to initialize database pool: {e}")
                raise

    @contextmanager
    def get_connection(self):
        """Get a connection from the pool (context manager)."""
        if not db_config.enabled or self._pool is None:
            raise ConfigurationError("Database not initialized")

        conn = self._pool.getconn()
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Database error, rolling back: {e}")
            raise
        finally:
            self._pool.putconn(conn)

    def close_all(self):
        """Close all connections in the pool."""
        if self._pool is not None:
            self._pool.closeall()
            logger.info("Database connection pool closed")
            self._pool = None


# Global pool instance
db_pool = DatabasePool()

# ─────────────────────────────────────────────────────────
# DATABASE OPERATIONS
# ─────────────────────────────────────────────────────────

class DatabaseOperations:
    """High-level database operations."""

    def __init__(self):
        self.pool = db_pool

    @staticmethod
    def is_enabled() -> bool:
        """Check if database is enabled."""
        return db_config.enabled

    # ─────────────────────────────────────────────────────
    # GAME STATISTICS
    # ─────────────────────────────────────────────────────

    def save_game_statistics(self, df: pd.DataFrame) -> int:
        """
        Save game statistics to database.

        Args:
            df: DataFrame with game statistics

        Returns:
            Number of rows inserted
        """
        with ErrorContext("Saving game statistics to database", logger=logger):
            validate_dataframe(df, min_rows=1)

            # Prepare data
            df = df.copy()
            df['created_at'] = datetime.now()
            df['updated_at'] = datetime.now()

            inserted = 0

            with self.pool.get_connection() as conn:
                with conn.cursor() as cur:
                    for _, row in df.iterrows():
                        try:
                            cur.execute("""
                                INSERT INTO game_statistics (
                                    season, date, team, team_opp, home, won,
                                    total, total_opp, created_at, updated_at
                                    -- Add all other columns as needed
                                )
                                VALUES (
                                    %(season)s, %(date)s, %(team)s, %(team_opp)s,
                                    %(home)s, %(won)s, %(total)s, %(total_opp)s,
                                    %(created_at)s, %(updated_at)s
                                )
                                ON CONFLICT (season, date, team, team_opp)
                                DO UPDATE SET
                                    won = EXCLUDED.won,
                                    total = EXCLUDED.total,
                                    total_opp = EXCLUDED.total_opp,
                                    updated_at = EXCLUDED.updated_at
                            """, row.to_dict())
                            inserted += 1
                        except Exception as e:
                            logger.warning(f"Failed to insert row: {e}")
                            continue

            logger.info(f"Saved {inserted} game statistics to database")
            return inserted

    def get_latest_game_statistics(
        self,
        team: Optional[str] = None,
        limit: int = 100
    ) -> pd.DataFrame:
        """
        Retrieve latest game statistics.

        Args:
            team: Optional team code filter
            limit: Maximum number of records

        Returns:
            DataFrame with game statistics
        """
        with ErrorContext("Retrieving game statistics from database", logger=logger):
            with self.pool.get_connection() as conn:
                query = """
                    SELECT * FROM game_statistics
                    WHERE 1=1
                    {team_filter}
                    ORDER BY date DESC
                    LIMIT %s
                """

                team_filter = "AND team = %s" if team else ""
                query = query.format(team_filter=team_filter)

                params = [team, limit] if team else [limit]

                df = pd.read_sql_query(query, conn, params=params)
                logger.info(f"Retrieved {len(df)} game statistics")
                return df

    # ─────────────────────────────────────────────────────
    # GAME SCHEDULE
    # ─────────────────────────────────────────────────────

    def save_game_schedule(self, df: pd.DataFrame) -> int:
        """
        Save game schedule to database.

        Args:
            df: DataFrame with columns [home_team, away_team, game_date]

        Returns:
            Number of rows inserted
        """
        with ErrorContext("Saving game schedule to database", logger=logger):
            validate_dataframe(
                df,
                required_columns=['home_team', 'away_team', 'game_date'],
                min_rows=1
            )

            inserted = 0

            with self.pool.get_connection() as conn:
                with conn.cursor() as cur:
                    for _, row in df.iterrows():
                        try:
                            cur.execute("""
                                INSERT INTO game_schedule (home_team, away_team, game_date)
                                VALUES (%(home_team)s, %(away_team)s, %(game_date)s)
                                ON CONFLICT (home_team, away_team, game_date)
                                DO NOTHING
                            """, row.to_dict())
                            inserted += cur.rowcount
                        except Exception as e:
                            logger.warning(f"Failed to insert schedule row: {e}")
                            continue

            logger.info(f"Saved {inserted} game schedules to database")
            return inserted

    def get_upcoming_games(self, days_ahead: int = 7) -> pd.DataFrame:
        """
        Get upcoming game schedule.

        Args:
            days_ahead: Number of days to look ahead

        Returns:
            DataFrame with upcoming games
        """
        with ErrorContext("Retrieving upcoming games from database", logger=logger):
            with self.pool.get_connection() as conn:
                query = """
                    SELECT *
                    FROM game_schedule
                    WHERE game_date >= CURRENT_DATE
                      AND game_date <= CURRENT_DATE + INTERVAL '%s days'
                    ORDER BY game_date
                """

                df = pd.read_sql_query(query, conn, params=[days_ahead])
                logger.info(f"Retrieved {len(df)} upcoming games")
                return df

    # ─────────────────────────────────────────────────────
    # PREDICTIONS
    # ─────────────────────────────────────────────────────

    def save_predictions(self, df: pd.DataFrame, model_version: str = None) -> int:
        """
        Save predictions to database.

        Args:
            df: DataFrame with predictions
            model_version: Optional model version identifier

        Returns:
            Number of rows inserted
        """
        with ErrorContext("Saving predictions to database", logger=logger):
            validate_dataframe(
                df,
                required_columns=['home_team', 'away_team', 'date', 'home_team_prob'],
                min_rows=1
            )

            inserted = 0
            prediction_date = datetime.now().date()

            with self.pool.get_connection() as conn:
                with conn.cursor() as cur:
                    for _, row in df.iterrows():
                        try:
                            cur.execute("""
                                INSERT INTO predictions (
                                    home_team, away_team, date, home_team_prob,
                                    odds_1, odds_2, result, model_version,
                                    prediction_date
                                )
                                VALUES (
                                    %(home_team)s, %(away_team)s, %(date)s,
                                    %(home_team_prob)s, %(odds_1)s, %(odds_2)s,
                                    %(result)s, %s, %s
                                )
                                ON CONFLICT (home_team, away_team, date, prediction_date)
                                DO UPDATE SET
                                    home_team_prob = EXCLUDED.home_team_prob,
                                    odds_1 = EXCLUDED.odds_1,
                                    odds_2 = EXCLUDED.odds_2,
                                    result = EXCLUDED.result,
                                    model_version = EXCLUDED.model_version,
                                    updated_at = NOW()
                            """, {**row.to_dict(), 'model_version': model_version, 'prediction_date': prediction_date})
                            inserted += 1
                        except Exception as e:
                            logger.warning(f"Failed to insert prediction row: {e}")
                            continue

            logger.info(f"Saved {inserted} predictions to database")
            return inserted

    def get_latest_predictions(self, limit: int = 100) -> pd.DataFrame:
        """
        Get latest predictions.

        Args:
            limit: Maximum number of records

        Returns:
            DataFrame with predictions
        """
        with ErrorContext("Retrieving predictions from database", logger=logger):
            with self.pool.get_connection() as conn:
                query = """
                    SELECT *
                    FROM v_latest_predictions
                    LIMIT %s
                """

                df = pd.read_sql_query(query, conn, params=[limit])
                logger.info(f"Retrieved {len(df)} predictions")
                return df

    # ─────────────────────────────────────────────────────
    # BETTING STATISTICS
    # ─────────────────────────────────────────────────────

    def save_enriched_predictions(self, df: pd.DataFrame) -> int:
        """
        Save enriched predictions with Kelly stakes.

        Args:
            df: DataFrame with enriched predictions

        Returns:
            Number of rows inserted
        """
        with ErrorContext("Saving enriched predictions to database", logger=logger):
            validate_dataframe(df, min_rows=1)

            inserted = 0
            enrichment_date = datetime.now().date()

            with self.pool.get_connection() as conn:
                with conn.cursor() as cur:
                    for _, row in df.iterrows():
                        try:
                            # First, find the prediction_id
                            cur.execute("""
                                SELECT id FROM predictions
                                WHERE home_team = %s
                                  AND away_team = %s
                                  AND date = %s
                                ORDER BY prediction_date DESC
                                LIMIT 1
                            """, (row['home_team'], row['away_team'], row['date']))

                            result = cur.fetchone()
                            if not result:
                                logger.warning(
                                    f"No prediction found for {row['home_team']} vs "
                                    f"{row['away_team']} on {row['date']}"
                                )
                                continue

                            prediction_id = result[0]

                            # Insert enriched prediction
                            cur.execute("""
                                INSERT INTO enriched_predictions (
                                    prediction_id, home_team, away_team, date,
                                    home_team_prob, raw_prob, odds_1, odds_2,
                                    prob_platt, prob_iso,
                                    stake_raw, stake_platt, stake_iso,
                                    win, pnl_raw, pnl_platt, pnl_iso,
                                    home_win_rate, enrichment_date
                                )
                                VALUES (
                                    %s, %(home_team)s, %(away_team)s, %(date)s,
                                    %(home_team_prob)s, %(raw_prob)s, %(odds_1)s, %(odds_2)s,
                                    %(prob_platt)s, %(prob_iso)s,
                                    %(stake_raw)s, %(stake_platt)s, %(stake_iso)s,
                                    %(win)s, %(pnl_raw)s, %(pnl_platt)s, %(pnl_iso)s,
                                    %(home_win_rate)s, %s
                                )
                                ON CONFLICT (prediction_id, enrichment_date)
                                DO UPDATE SET
                                    prob_platt = EXCLUDED.prob_platt,
                                    prob_iso = EXCLUDED.prob_iso,
                                    stake_raw = EXCLUDED.stake_raw,
                                    stake_platt = EXCLUDED.stake_platt,
                                    stake_iso = EXCLUDED.stake_iso,
                                    win = EXCLUDED.win,
                                    pnl_raw = EXCLUDED.pnl_raw,
                                    pnl_platt = EXCLUDED.pnl_platt,
                                    pnl_iso = EXCLUDED.pnl_iso,
                                    updated_at = NOW()
                            """, {**row.to_dict(), 'prediction_id': prediction_id, 'enrichment_date': enrichment_date})
                            inserted += 1
                        except Exception as e:
                            logger.warning(f"Failed to insert enriched prediction row: {e}")
                            continue

            logger.info(f"Saved {inserted} enriched predictions to database")
            return inserted

    def get_betting_performance(self) -> Dict[str, Any]:
        """
        Get betting performance summary.

        Returns:
            Dictionary with betting performance metrics
        """
        with ErrorContext("Retrieving betting performance from database", logger=logger):
            with self.pool.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("SELECT * FROM v_betting_performance")
                    result = cur.fetchone()

                    if result:
                        performance = dict(result)
                        logger.info(
                            f"Betting performance: {performance['total_bets']} bets, "
                            f"{performance['win_rate']}% win rate"
                        )
                        return performance
                    else:
                        return {}

    # ─────────────────────────────────────────────────────
    # UTILITY METHODS
    # ─────────────────────────────────────────────────────

    def execute_query(self, query: str, params: Optional[Tuple] = None) -> List[Dict]:
        """
        Execute a raw SQL query.

        Args:
            query: SQL query string
            params: Optional query parameters

        Returns:
            List of result dictionaries
        """
        with ErrorContext("Executing custom query", logger=logger):
            with self.pool.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(query, params)
                    return [dict(row) for row in cur.fetchall()]


# Global database operations instance
db = DatabaseOperations()

# ─────────────────────────────────────────────────────────
# INITIALIZATION
# ─────────────────────────────────────────────────────────

def initialize_database():
    """Initialize database connection pool."""
    if db_config.enabled:
        db_pool.initialize()
    else:
        logger.info("Database disabled - using CSV storage")


def close_database():
    """Close database connections."""
    db_pool.close_all()


# ─────────────────────────────────────────────────────────
# EXAMPLE USAGE
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Example: Initialize and test database connection
    try:
        initialize_database()

        if db_config.enabled:
            # Test query
            result = db.execute_query("SELECT COUNT(*) as count FROM teams")
            print(f"Teams in database: {result[0]['count']}")

            # Get betting performance
            performance = db.get_betting_performance()
            print(f"Betting performance: {performance}")

    except Exception as e:
        logger.error(f"Database test failed: {e}")
    finally:
        close_database()
