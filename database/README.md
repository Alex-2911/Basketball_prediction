# Database Migration Infrastructure

This directory contains the complete infrastructure for migrating your Basketball Prediction system from CSV storage to PostgreSQL/Supabase.

## 📁 Directory Structure

```
database/
├── schemas/
│   └── 001_initial_schema.sql    # Complete PostgreSQL schema (8 tables, indexes, triggers)
├── scripts/
│   ├── migrate_game_statistics.py       # Migrate historical game data
│   ├── migrate_predictions.py           # Migrate prediction history
│   └── migrate_enriched_predictions.py  # Migrate Kelly stakes & PnL
└── README.md                      # This file
```

## 🚀 Quick Start

### 1. Set Up Supabase (5 minutes)

1. Go to [supabase.com](https://supabase.com) and create a new project
2. In Supabase SQL Editor, run the contents of `schemas/001_initial_schema.sql`
3. Get your connection string from Settings → Database → Connection String

### 2. Configure Environment

Add to your `.env` file:

```bash
USE_DATABASE=true
DATABASE_URL=postgresql://postgres:[password]@db.yourproject.supabase.co:5432/postgres
```

### 3. Run Migrations

```bash
# Test first (no data inserted)
python database/scripts/migrate_game_statistics.py --dry-run

# Run actual migrations (in order!)
python database/scripts/migrate_game_statistics.py
python database/scripts/migrate_predictions.py
python database/scripts/migrate_enriched_predictions.py
```

## 📊 Schema Overview

### Tables Created

| Table | Purpose | Records |
|-------|---------|---------|
| `game_statistics` | Historical game data from Basketball-Reference | Millions |
| `predictions` | Model predictions for upcoming games | Thousands |
| `enriched_predictions` | Kelly criterion stakes and profit/loss | Thousands |
| `betting_statistics` | Performance metrics by team/strategy | Hundreds |
| `game_schedule` | Upcoming game schedule | Dozens |
| `teams` | NBA team reference (30 teams) | 30 |
| `model_versions` | Model training history | Dozens |
| `audit_log` | Change tracking | All changes |

### Key Features

✅ **Foreign Keys** - Data integrity enforced
✅ **Indexes** - Fast queries on date, team, season
✅ **Triggers** - Automatic timestamp updates
✅ **Views** - Pre-built queries for common operations
✅ **Constraints** - Prevent invalid data

## 🛠️ Migration Scripts

### migrate_game_statistics.py

Migrates all `nba_games_*.csv` files to the database.

**Options:**
- `--dry-run` - Preview without inserting data
- `--batch-size 1000` - Control memory usage
- `--latest-only` - Migrate only the newest file

**Example:**
```bash
# Preview migration
python database/scripts/migrate_game_statistics.py --dry-run

# Migrate all historical data
python database/scripts/migrate_game_statistics.py --batch-size 1000
```

### migrate_predictions.py

Migrates `predictions_*.csv` files to the database.

**Example:**
```bash
# Migrate all predictions
python database/scripts/migrate_predictions.py

# Update only today's predictions
python database/scripts/migrate_predictions.py --latest-only
```

### migrate_enriched_predictions.py

Migrates `enriched_predictions_*.csv` (Kelly stakes, PnL) to the database.

**⚠️ Important:** Run `migrate_predictions.py` first! This script needs prediction IDs.

**Example:**
```bash
# Migrate enriched data
python database/scripts/migrate_enriched_predictions.py
```

## 📖 Full Documentation

For complete setup instructions, troubleshooting, and advanced usage, see:

**[docs/DATABASE_SETUP.md](../docs/DATABASE_SETUP.md)**

This comprehensive guide includes:
- Step-by-step Supabase setup with screenshots
- Environment configuration options
- Complete schema documentation with sample queries
- Migration walkthrough
- Integration with existing scripts
- Troubleshooting common issues
- Rollback strategy

## 🔄 Dual Mode Operation

The system works with **both CSV and database** simultaneously:

- **Database OFF** (`USE_DATABASE=false`) - Uses CSV files (default)
- **Database ON** (`USE_DATABASE=true`) - Saves to both database AND CSV

This means:
- ✅ No breaking changes - CSV workflow still works
- ✅ Gradual migration - Enable when ready
- ✅ Backup strategy - CSV files are preserved
- ✅ Easy rollback - Just set `USE_DATABASE=false`

## 🎯 Migration Checklist

- [ ] Create Supabase project
- [ ] Apply schema (`001_initial_schema.sql`)
- [ ] Configure `.env` with `DATABASE_URL`
- [ ] Set `USE_DATABASE=true`
- [ ] Run `migrate_game_statistics.py --dry-run`
- [ ] Run `migrate_game_statistics.py`
- [ ] Run `migrate_predictions.py`
- [ ] Run `migrate_enriched_predictions.py`
- [ ] Verify data in Supabase Table Editor
- [ ] Update Scripts 1-6 to use database (optional)

## 💡 Tips

### Test Connection

```python
python -c "from db_utils import db_pool; db_pool.initialize(); print('Connected!')"
```

### Check Row Counts

```sql
SELECT
    'game_statistics' as table_name,
    COUNT(*) as rows
FROM game_statistics
UNION ALL
SELECT 'predictions', COUNT(*) FROM predictions
UNION ALL
SELECT 'enriched_predictions', COUNT(*) FROM enriched_predictions;
```

### Incremental Updates

After initial migration, sync new data daily:

```bash
python database/scripts/migrate_game_statistics.py --latest-only
python database/scripts/migrate_predictions.py --latest-only
python database/scripts/migrate_enriched_predictions.py --latest-only
```

## 🆘 Troubleshooting

**"Could not connect to database"**
- Verify `DATABASE_URL` is correct in `.env`
- Check Supabase project is active (not paused)

**"Relation does not exist"**
- Schema not applied - run `001_initial_schema.sql` in Supabase SQL Editor

**"No existing predictions in database"**
- Run migrations in order: game_statistics → predictions → enriched_predictions

**More issues?** See [DATABASE_SETUP.md](../docs/DATABASE_SETUP.md) troubleshooting section.

## 📦 Dependencies

Required packages (already in `requirements.txt`):
- `psycopg2-binary==2.9.9` - PostgreSQL driver
- `pandas>=2.2.3` - Data manipulation
- `python-dotenv>=1.0.0` - Environment variables

## 🔗 Related Files

- `2026/src/db_utils.py` - Database connection and CRUD operations
- `2026/src/error_handlers.py` - Error handling utilities
- `2026/src/logger.py` - Logging configuration
- `.env.example` - Environment variable template
- `docs/DATABASE_SETUP.md` - Complete documentation

---

**Questions?** See the full documentation in [docs/DATABASE_SETUP.md](../docs/DATABASE_SETUP.md)
