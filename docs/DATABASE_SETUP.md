# Database Setup Guide - Supabase/PostgreSQL Migration

This guide explains how to migrate your Basketball Prediction system from CSV storage to a production-grade PostgreSQL database using Supabase.

## Table of Contents

1. [Why Migrate to PostgreSQL?](#why-migrate-to-postgresql)
2. [Quick Start](#quick-start)
3. [Supabase Setup](#supabase-setup)
4. [Environment Configuration](#environment-configuration)
5. [Database Schema](#database-schema)
6. [Running Migrations](#running-migrations)
7. [Integration with Existing Scripts](#integration-with-existing-scripts)
8. [Troubleshooting](#troubleshooting)
9. [Rollback Strategy](#rollback-strategy)

---

## Why Migrate to PostgreSQL?

### Benefits

✅ **Better Performance**: SQL queries are faster than scanning CSV files
✅ **Data Integrity**: Foreign keys and constraints prevent invalid data
✅ **Concurrent Access**: Multiple processes can read/write simultaneously
✅ **Advanced Queries**: Complex analysis with SQL joins, aggregations, window functions
✅ **Audit Trail**: Automatic change tracking with triggers
✅ **Scalability**: Handles millions of rows efficiently
✅ **Backups**: Supabase provides automatic backups and point-in-time recovery

### CSV vs PostgreSQL Comparison

| Feature | CSV Files | PostgreSQL |
|---------|-----------|------------|
| Query Speed | O(n) - full scan | O(log n) - indexed |
| Concurrent Writes | ❌ File locking issues | ✅ MVCC transactions |
| Data Validation | ❌ None | ✅ Constraints & types |
| Relationships | ❌ Manual joins | ✅ Foreign keys |
| Audit Trail | ❌ None | ✅ Automatic triggers |
| Backup/Recovery | ❌ Manual | ✅ Automated |

---

## Quick Start

### 1. Enable Database (Optional)

The system works with **both CSV and database** simultaneously. To enable database:

```bash
# In your .env file
USE_DATABASE=true
```

To keep using CSV only (default):

```bash
USE_DATABASE=false
```

### 2. Three Ways to Connect

Choose ONE of these methods:

**Option A: Full Connection String (Recommended for Supabase)**
```bash
DATABASE_URL=postgresql://postgres:[password]@db.[project-id].supabase.co:5432/postgres
```

**Option B: Individual Components**
```bash
DB_HOST=db.yourproject.supabase.co
DB_PORT=5432
DB_NAME=postgres
DB_USER=postgres
DB_PASSWORD=your_password
```

**Option C: Local PostgreSQL (Development)**
```bash
DATABASE_URL=postgresql://localhost/basketball_predictions
```

---

## Supabase Setup

### Step 1: Create Supabase Project

1. Go to [supabase.com](https://supabase.com)
2. Click **"New Project"**
3. Choose:
   - **Organization**: Your organization or create new
   - **Name**: `basketball-predictions` (or your choice)
   - **Database Password**: Strong password (save this!)
   - **Region**: Choose closest to you
4. Click **"Create new project"** (takes ~2 minutes)

### Step 2: Get Connection Details

1. In Supabase dashboard, go to **Settings** → **Database**
2. Scroll to **Connection String** section
3. Copy the **URI** format string:
   ```
   postgresql://postgres:[YOUR-PASSWORD]@db.xyz.supabase.co:5432/postgres
   ```
4. Replace `[YOUR-PASSWORD]` with your actual password

### Step 3: Apply Schema

Two ways to apply the schema:

#### Option A: Supabase SQL Editor (Recommended)

1. In Supabase dashboard, go to **SQL Editor**
2. Click **"New Query"**
3. Copy entire contents of `database/schemas/001_initial_schema.sql`
4. Paste into editor
5. Click **"Run"**
6. Verify: Should see "Success. No rows returned" (this is correct!)

#### Option B: Command Line (psql)

```bash
# Install PostgreSQL client if needed
sudo apt-get install postgresql-client  # Linux
brew install postgresql                  # macOS

# Apply schema
psql "postgresql://postgres:[password]@db.xyz.supabase.co:5432/postgres" \
  -f database/schemas/001_initial_schema.sql
```

### Step 4: Verify Installation

In Supabase **Table Editor**, you should see these tables:

- `game_statistics`
- `game_schedule`
- `predictions`
- `betting_statistics`
- `enriched_predictions`
- `teams` (should have 30 rows - NBA teams)
- `model_versions`
- `audit_log`

---

## Environment Configuration

### Required Variables

Add these to your `.env` file:

```bash
# =============================================================================
# Database Configuration (Optional)
# =============================================================================

# Enable database (set to 'false' to use CSV only)
USE_DATABASE=true

# Connection String (RECOMMENDED - Get from Supabase dashboard)
DATABASE_URL=postgresql://postgres:your_password@db.yourproject.supabase.co:5432/postgres

# OR use individual components:
# DB_HOST=db.yourproject.supabase.co
# DB_PORT=5432
# DB_NAME=postgres
# DB_USER=postgres
# DB_PASSWORD=your_password

# Connection Pool Settings (Optional - defaults shown)
DB_POOL_MIN_CONNECTIONS=2
DB_POOL_MAX_CONNECTIONS=10
DB_CONNECT_TIMEOUT=30
```

### .env.example Template

Your `.env.example` has been updated with database variables. Copy to `.env`:

```bash
cp .env.example .env
# Then edit .env with your actual values
```

---

## Database Schema

### Tables Overview

#### 1. `game_statistics`
Stores historical game data from Basketball-Reference.

**Key Columns:**
- `season`, `date`, `team`, `team_opp`
- `home`, `won`, `total`, `total_opp`
- Advanced stats: `fg`, `fga`, `fg_pct`, `3p`, `3pa`, `ts_pct`, `orb`, `drb`, etc.
- Opponent stats: All stats with `_opp` suffix

**Indexes:**
- `(season, date, team, team_opp)` - UNIQUE
- `date` - for date range queries
- `team` - for team-specific queries

**Sample Query:**
```sql
-- Get last 10 games for Lakers
SELECT * FROM game_statistics
WHERE team = 'LAL'
ORDER BY date DESC
LIMIT 10;
```

#### 2. `predictions`
Stores model predictions for upcoming games.

**Key Columns:**
- `home_team`, `away_team`, `date`
- `home_team_prob` - probability home team wins (0-1)
- `odds_1`, `odds_2` - betting odds
- `result` - actual outcome (updated after game)
- `prediction_date` - when prediction was made

**Unique Constraint:** `(home_team, away_team, date, prediction_date)`

**Sample Query:**
```sql
-- Today's predictions
SELECT * FROM predictions
WHERE prediction_date = CURRENT_DATE;
```

#### 3. `enriched_predictions`
Kelly criterion stakes and profit/loss tracking.

**Key Columns:**
- `prediction_id` - foreign key to predictions
- `prob_platt`, `prob_iso` - calibrated probabilities
- `stake_raw`, `stake_platt`, `stake_iso` - Kelly stakes
- `pnl_raw`, `pnl_platt`, `pnl_iso` - profit/loss
- `edge_raw`, `edge_platt`, `edge_iso` - expected value

**Sample Query:**
```sql
-- Best bets (highest expected value)
SELECT p.home_team, p.away_team, e.edge_platt, e.stake_platt
FROM predictions p
JOIN enriched_predictions e ON p.id = e.prediction_id
WHERE e.stake_platt > 0
ORDER BY e.edge_platt DESC
LIMIT 10;
```

#### 4. `betting_statistics`
Historical betting performance by team and strategy.

**Key Columns:**
- `team`, `strategy` (e.g., 'raw', 'platt', 'isotonic')
- `games_played`, `win_rate`
- `total_stake`, `total_pnl`, `roi`
- `sharpe_ratio`, `max_drawdown`

**Sample Query:**
```sql
-- Best performing teams with Platt calibration
SELECT team, win_rate, roi, sharpe_ratio
FROM betting_statistics
WHERE strategy = 'platt'
  AND games_played >= 10
ORDER BY roi DESC;
```

#### 5. `teams`
NBA team reference data.

**Key Columns:**
- `code` - 3-letter code (e.g., 'LAL')
- `full_name` - "Los Angeles Lakers"
- `alternate_codes` - array of codes (handles PHO→PHX, BKN→BRK)
- `conference`, `division`

**Sample Query:**
```sql
-- Find team by any code
SELECT * FROM teams
WHERE 'PHO' = ANY(alternate_codes);  -- Returns Phoenix Suns
```

### Useful Views

#### `v_latest_predictions`
Most recent predictions with enriched data:

```sql
SELECT * FROM v_latest_predictions;
```

#### `v_betting_summary`
Betting performance summary across all strategies:

```sql
SELECT * FROM v_betting_summary;
```

---

## Running Migrations

### Migration Scripts Location

All migration scripts are in `database/scripts/`:

1. `migrate_game_statistics.py` - Historical game data
2. `migrate_predictions.py` - Prediction history
3. `migrate_enriched_predictions.py` - Kelly stakes and PnL

### Prerequisites

1. **Database must be set up** (schema applied)
2. **Environment variables configured** (`.env` file)
3. **USE_DATABASE=true** in `.env`

### Migration Order

⚠️ **IMPORTANT**: Run migrations in this order:

```bash
# 1. Migrate game statistics (largest dataset)
python database/scripts/migrate_game_statistics.py

# 2. Migrate predictions
python database/scripts/migrate_predictions.py

# 3. Migrate enriched predictions (requires predictions to exist)
python database/scripts/migrate_enriched_predictions.py
```

### Migration Options

Each script supports:

#### Dry Run (Recommended First)

Test migration without inserting data:

```bash
python database/scripts/migrate_game_statistics.py --dry-run
```

Output shows:
- Files that would be processed
- Row counts
- Sample data preview
- No data is inserted

#### Batch Size

Control memory usage for large datasets:

```bash
# Smaller batches for slow connections
python database/scripts/migrate_game_statistics.py --batch-size 500

# Larger batches for fast local connections
python database/scripts/migrate_game_statistics.py --batch-size 5000
```

#### Latest Only

Migrate only the most recent file:

```bash
python database/scripts/migrate_predictions.py --latest-only
```

Useful for:
- Testing migrations
- Incremental updates
- Quick verification

### Full Migration Example

```bash
# Step 1: Dry run to verify everything
python database/scripts/migrate_game_statistics.py --dry-run
python database/scripts/migrate_predictions.py --dry-run
python database/scripts/migrate_enriched_predictions.py --dry-run

# Step 2: Migrate game statistics (takes longest)
python database/scripts/migrate_game_statistics.py --batch-size 1000

# Step 3: Migrate predictions
python database/scripts/migrate_predictions.py --batch-size 500

# Step 4: Migrate enriched predictions
python database/scripts/migrate_enriched_predictions.py --batch-size 500
```

### Expected Output

Successful migration output:

```
============================================================
Game Statistics Migration to PostgreSQL
============================================================
Source directory: /path/to/2026/data/statistics
Dry run: False
Batch size: 1000
============================================================
Found 150 CSV files
Database connection established
Reading nba_games_2024-10-22.csv
Progress: 1000/15234 rows (6%)
Progress: 2000/15234 rows (13%)
...
Successfully migrated 15234 rows from nba_games_2024-10-22.csv
...
============================================================
Migration Summary
============================================================
Files processed: 150
Successful: 150
Failed: 0
Total rows: 2,284,500
============================================================
```

### Incremental Migrations

After initial migration, you can run migrations periodically to sync new data:

```bash
# Migrate only today's new data
python database/scripts/migrate_game_statistics.py --latest-only
python database/scripts/migrate_predictions.py --latest-only
python database/scripts/migrate_enriched_predictions.py --latest-only
```

---

## Integration with Existing Scripts

### Dual Mode Operation

All scripts work with **both CSV and database** simultaneously:

```python
# In your scripts
if db_config.enabled:
    # Save to database
    db_ops.save_game_statistics(games_df)
    logger.info("Saved to database")

# Always save to CSV (backwards compatibility)
games_df.to_csv(output_path, index=False)
logger.info("Saved to CSV")
```

### Reading Data

Scripts automatically check database first:

```python
# Automatically tries database first, falls back to CSV
if db_config.enabled:
    games_df = db_ops.get_latest_game_statistics(limit=1000)
    if not games_df.empty:
        logger.info("Loaded from database")
    else:
        games_df = pd.read_csv(csv_path)
        logger.info("Loaded from CSV")
else:
    games_df = pd.read_csv(csv_path)
```

### Updating Scripts to Use Database

Example for Script 1 (Get Previous Game Day):

```python
# At the end of main(), after saving CSV:
if db_config.enabled:
    try:
        db_ops.save_game_statistics(combined)
        logger.info(f"Saved {len(combined)} rows to database")
    except Exception as e:
        logger.warning(f"Failed to save to database: {e}")
        logger.info("Data still saved to CSV")
```

No changes needed if `USE_DATABASE=false` - scripts continue working with CSV only.

---

## Troubleshooting

### Connection Errors

**Error**: `FATAL: password authentication failed`

**Solution**:
- Verify password in `.env` is correct
- Check DATABASE_URL has no typos
- Ensure password doesn't contain special characters that need URL encoding

**Error**: `could not connect to server: Connection refused`

**Solution**:
- Check DB_HOST is correct
- Verify Supabase project is active (not paused)
- Check firewall allows port 5432

### Schema Errors

**Error**: `relation "game_statistics" does not exist`

**Solution**:
- Schema not applied - run `001_initial_schema.sql` in Supabase SQL Editor
- Verify with: `SELECT * FROM teams;` (should return 30 NBA teams)

**Error**: `column "xyz" does not exist`

**Solution**:
- Schema version mismatch
- Drop all tables and re-apply schema:
  ```sql
  DROP SCHEMA public CASCADE;
  CREATE SCHEMA public;
  -- Then run 001_initial_schema.sql
  ```

### Migration Errors

**Error**: `No existing predictions in database`

**Solution**:
- Run `migrate_predictions.py` **before** `migrate_enriched_predictions.py`
- Enriched predictions need prediction IDs to reference

**Error**: `IntegrityError: duplicate key value violates unique constraint`

**Solution**:
- Data already exists in database
- Either:
  - Skip migration (data already there)
  - Clear table: `TRUNCATE TABLE game_statistics CASCADE;`
  - Run with `--latest-only` to update only recent data

### Performance Issues

**Slow queries?**

1. Check indexes exist:
   ```sql
   SELECT tablename, indexname FROM pg_indexes
   WHERE schemaname = 'public'
   ORDER BY tablename, indexname;
   ```

2. Analyze tables (updates statistics):
   ```sql
   ANALYZE game_statistics;
   ANALYZE predictions;
   ANALYZE enriched_predictions;
   ```

3. Add custom indexes for your queries:
   ```sql
   CREATE INDEX idx_games_team_date ON game_statistics(team, date DESC);
   ```

**Too many connections?**

- Increase pool size in `.env`:
  ```bash
  DB_POOL_MAX_CONNECTIONS=20
  ```

- Or reduce if running out of connections:
  ```bash
  DB_POOL_MAX_CONNECTIONS=5
  ```

---

## Rollback Strategy

### Keep CSV Files

**Important**: Don't delete CSV files after migration!

The system is designed to work with both:
- Database provides fast queries and advanced features
- CSV files provide backup and portability

### Disable Database

To revert to CSV-only mode:

```bash
# In .env
USE_DATABASE=false
```

All scripts immediately fall back to CSV without code changes.

### Export Database to CSV

To create CSV backups from database:

```python
import pandas as pd
from db_utils import DatabaseOperations, db_pool

db_pool.initialize()
db_ops = DatabaseOperations()

# Export game statistics
games_df = db_ops.get_all_game_statistics()
games_df.to_csv('backup_game_statistics.csv', index=False)

# Export predictions
preds_df = db_ops.get_all_predictions()
preds_df.to_csv('backup_predictions.csv', index=False)
```

### Supabase Backups

Supabase provides automatic backups:

1. Go to **Database** → **Backups**
2. See daily automatic backups (retained based on plan)
3. Click **Restore** to revert to a specific backup

---

## Advanced Topics

### Custom Queries

Create custom analysis queries:

```python
from db_utils import db_pool

with db_pool.get_connection() as conn:
    # Custom SQL query
    query = """
    SELECT
        team,
        AVG(total) as avg_points,
        AVG(total_opp) as avg_points_allowed,
        SUM(CASE WHEN won THEN 1 ELSE 0 END)::float / COUNT(*) as win_rate
    FROM game_statistics
    WHERE season = '2026'
      AND date >= CURRENT_DATE - INTERVAL '30 days'
    GROUP BY team
    ORDER BY win_rate DESC;
    """

    df = pd.read_sql_query(query, conn)
    print(df)
```

### Adding Indexes

For frequently-run queries, add custom indexes:

```sql
-- Index for team performance queries
CREATE INDEX idx_games_team_season_date
ON game_statistics(team, season, date DESC);

-- Index for prediction lookups by date
CREATE INDEX idx_predictions_date
ON predictions(date DESC, prediction_date DESC);
```

### Monitoring

Check database size:

```sql
SELECT
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
```

Check query performance:

```sql
-- Enable query timing
\timing on

-- Run your query
SELECT * FROM game_statistics WHERE team = 'LAL';

-- See execution time
```

---

## Next Steps

After successful migration:

1. ✅ **Verify data** - Run sample queries to check data integrity
2. ✅ **Update scripts** - Add database save operations to Scripts 1-6
3. ✅ **Test dual mode** - Ensure CSV and database both work
4. ✅ **Monitor performance** - Check query speeds and connection pool
5. ✅ **Set up backups** - Configure Supabase backup retention
6. ✅ **Create dashboards** - Build visualizations using database queries

---

## Support

- **Supabase Docs**: https://supabase.com/docs
- **PostgreSQL Docs**: https://www.postgresql.org/docs/
- **Project Issues**: Check logs in `logs/` directory
- **Migration Issues**: Run with `--dry-run` first to identify problems

---

## Summary

| Task | Command |
|------|---------|
| **Setup Supabase** | Create project at supabase.com |
| **Apply Schema** | Run `001_initial_schema.sql` in SQL Editor |
| **Configure .env** | Add DATABASE_URL |
| **Test Connection** | `python -c "from db_utils import db_pool; db_pool.initialize()"` |
| **Dry Run Migration** | `python database/scripts/migrate_*.py --dry-run` |
| **Full Migration** | Run all 3 migration scripts in order |
| **Verify** | Check Supabase Table Editor |
| **Enable in Scripts** | `USE_DATABASE=true` in `.env` |

---

**Happy migrating! 🚀**
