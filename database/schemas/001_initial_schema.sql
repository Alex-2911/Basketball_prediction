-- =====================================================
-- Basketball Prediction Database Schema
-- Supabase/PostgreSQL Migration
-- =====================================================

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- =====================================================
-- 1. GAME STATISTICS TABLE
-- Stores historical game data scraped from Basketball-Reference
-- =====================================================

CREATE TABLE IF NOT EXISTS game_statistics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Game identification
    season VARCHAR(10) NOT NULL,
    date DATE NOT NULL,
    team VARCHAR(3) NOT NULL,
    team_opp VARCHAR(3) NOT NULL,
    home INTEGER NOT NULL CHECK (home IN (0, 1)),
    won BOOLEAN NOT NULL,

    -- Scores
    total INTEGER,
    total_opp INTEGER,

    -- Team statistics (basic)
    mp NUMERIC,
    fg NUMERIC,
    fga NUMERIC,
    fg_pct NUMERIC,
    fg3 NUMERIC,
    fg3a NUMERIC,
    fg3_pct NUMERIC,
    ft NUMERIC,
    fta NUMERIC,
    ft_pct NUMERIC,
    orb NUMERIC,
    drb NUMERIC,
    trb NUMERIC,
    ast NUMERIC,
    stl NUMERIC,
    blk NUMERIC,
    tov NUMERIC,
    pf NUMERIC,

    -- Advanced statistics
    ts_pct NUMERIC,
    efg_pct NUMERIC,
    fg3a_rate NUMERIC,
    fta_rate NUMERIC,
    orb_pct NUMERIC,
    drb_pct NUMERIC,
    trb_pct NUMERIC,
    ast_pct NUMERIC,
    stl_pct NUMERIC,
    blk_pct NUMERIC,
    tov_pct NUMERIC,
    usg_pct NUMERIC,
    ortg NUMERIC,
    drtg NUMERIC,

    -- Opponent statistics (mirrored structure)
    mp_opp NUMERIC,
    fg_opp NUMERIC,
    fga_opp NUMERIC,
    fg_pct_opp NUMERIC,
    fg3_opp NUMERIC,
    fg3a_opp NUMERIC,
    fg3_pct_opp NUMERIC,
    ft_opp NUMERIC,
    fta_opp NUMERIC,
    ft_pct_opp NUMERIC,
    orb_opp NUMERIC,
    drb_opp NUMERIC,
    trb_opp NUMERIC,
    ast_opp NUMERIC,
    stl_opp NUMERIC,
    blk_opp NUMERIC,
    tov_opp NUMERIC,
    pf_opp NUMERIC,

    -- Metadata
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    -- Constraints
    UNIQUE (season, date, team, team_opp)
);

-- Indexes for performance
CREATE INDEX idx_game_statistics_date ON game_statistics(date DESC);
CREATE INDEX idx_game_statistics_team ON game_statistics(team);
CREATE INDEX idx_game_statistics_season ON game_statistics(season);
CREATE INDEX idx_game_statistics_team_date ON game_statistics(team, date DESC);

-- =====================================================
-- 2. GAME SCHEDULE TABLE
-- Stores upcoming game schedules
-- =====================================================

CREATE TABLE IF NOT EXISTS game_schedule (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Game details
    home_team VARCHAR(3) NOT NULL,
    away_team VARCHAR(3) NOT NULL,
    game_date DATE NOT NULL,

    -- Metadata
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    -- Constraints
    UNIQUE (home_team, away_team, game_date)
);

-- Indexes
CREATE INDEX idx_game_schedule_date ON game_schedule(game_date DESC);
CREATE INDEX idx_game_schedule_home ON game_schedule(home_team);
CREATE INDEX idx_game_schedule_away ON game_schedule(away_team);

-- =====================================================
-- 3. PREDICTIONS TABLE
-- Stores ML model predictions for upcoming games
-- =====================================================

CREATE TABLE IF NOT EXISTS predictions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Game identification
    home_team VARCHAR(3) NOT NULL,
    away_team VARCHAR(3) NOT NULL,
    date DATE NOT NULL,

    -- Prediction outputs
    home_team_prob NUMERIC NOT NULL CHECK (home_team_prob >= 0 AND home_team_prob <= 1),
    result VARCHAR(10),  -- 'home', 'away', or NULL if not played yet

    -- Odds data
    odds_1 NUMERIC,  -- Home team odds (decimal format)
    odds_2 NUMERIC,  -- Away team odds (decimal format)

    -- Model metadata
    model_version VARCHAR(50),
    prediction_date DATE NOT NULL,

    -- Metadata
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    -- Constraints
    UNIQUE (home_team, away_team, date, prediction_date)
);

-- Indexes
CREATE INDEX idx_predictions_date ON predictions(date DESC);
CREATE INDEX idx_predictions_home ON predictions(home_team);
CREATE INDEX idx_predictions_prediction_date ON predictions(prediction_date DESC);

-- =====================================================
-- 4. BETTING STATISTICS TABLE
-- Stores betting performance tracking
-- =====================================================

CREATE TABLE IF NOT EXISTS betting_statistics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Links to prediction
    prediction_id UUID REFERENCES predictions(id) ON DELETE CASCADE,

    -- Game details
    home_team VARCHAR(3) NOT NULL,
    away_team VARCHAR(3) NOT NULL,
    date DATE NOT NULL,

    -- Prediction data
    home_team_prob NUMERIC,
    odds_1 NUMERIC,
    odds_2 NUMERIC,

    -- Actual outcome
    win BOOLEAN,  -- Did home team win?
    accuracy NUMERIC,

    -- Metadata
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_betting_statistics_date ON betting_statistics(date DESC);
CREATE INDEX idx_betting_statistics_prediction ON betting_statistics(prediction_id);

-- =====================================================
-- 5. ENRICHED PREDICTIONS TABLE
-- Stores calibrated predictions with Kelly Criterion stakes
-- =====================================================

CREATE TABLE IF NOT EXISTS enriched_predictions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Links to prediction
    prediction_id UUID REFERENCES predictions(id) ON DELETE CASCADE,

    -- Game details
    home_team VARCHAR(3) NOT NULL,
    away_team VARCHAR(3) NOT NULL,
    date DATE NOT NULL,

    -- Original prediction
    home_team_prob NUMERIC,
    raw_prob NUMERIC,
    odds_1 NUMERIC,
    odds_2 NUMERIC,

    -- Calibrated probabilities
    prob_platt NUMERIC,  -- Platt scaling calibrated probability
    prob_iso NUMERIC,    -- Isotonic regression calibrated probability

    -- Kelly Criterion stakes (as fraction of bankroll)
    stake_raw NUMERIC DEFAULT 0,
    stake_platt NUMERIC DEFAULT 0,
    stake_iso NUMERIC DEFAULT 0,

    -- Profit/Loss tracking
    win BOOLEAN,
    pnl_raw NUMERIC DEFAULT 0,
    pnl_platt NUMERIC DEFAULT 0,
    pnl_iso NUMERIC DEFAULT 0,

    -- Home team performance metrics
    home_win_rate NUMERIC,

    -- Metadata
    enrichment_date DATE NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    -- Constraints
    UNIQUE (prediction_id, enrichment_date)
);

-- Indexes
CREATE INDEX idx_enriched_predictions_date ON enriched_predictions(date DESC);
CREATE INDEX idx_enriched_predictions_stake_raw ON enriched_predictions(stake_raw) WHERE stake_raw > 0;
CREATE INDEX idx_enriched_predictions_stake_platt ON enriched_predictions(stake_platt) WHERE stake_platt > 0;
CREATE INDEX idx_enriched_predictions_stake_iso ON enriched_predictions(stake_iso) WHERE stake_iso > 0;

-- =====================================================
-- 6. TEAM METADATA TABLE
-- Stores team information and mappings
-- =====================================================

CREATE TABLE IF NOT EXISTS teams (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Team identification
    code VARCHAR(3) UNIQUE NOT NULL,
    full_name VARCHAR(100) NOT NULL,

    -- Alternative codes/names for normalization
    alternate_codes TEXT[],  -- Array of alternative codes (PHO, PHX, etc.)

    -- Team metadata
    conference VARCHAR(10),  -- 'East' or 'West'
    division VARCHAR(20),

    -- Metadata
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index
CREATE INDEX idx_teams_code ON teams(code);

-- =====================================================
-- 7. MODEL VERSIONS TABLE
-- Tracks ML model versions and performance
-- =====================================================

CREATE TABLE IF NOT EXISTS model_versions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Model identification
    version VARCHAR(50) UNIQUE NOT NULL,
    model_type VARCHAR(50) NOT NULL,  -- 'LightGBM', etc.

    -- Training metadata
    trained_at TIMESTAMPTZ NOT NULL,
    training_samples INTEGER,
    test_accuracy NUMERIC,

    -- Model parameters (stored as JSONB)
    parameters JSONB,
    feature_importance JSONB,

    -- Model file location
    file_path TEXT,

    -- Status
    is_active BOOLEAN DEFAULT FALSE,

    -- Metadata
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index
CREATE INDEX idx_model_versions_version ON model_versions(version);
CREATE INDEX idx_model_versions_active ON model_versions(is_active) WHERE is_active = TRUE;

-- =====================================================
-- 8. AUDIT LOG TABLE
-- Tracks all data changes for debugging
-- =====================================================

CREATE TABLE IF NOT EXISTS audit_log (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Audit details
    table_name VARCHAR(50) NOT NULL,
    operation VARCHAR(10) NOT NULL,  -- 'INSERT', 'UPDATE', 'DELETE'
    record_id UUID,

    -- Changes (stored as JSONB)
    old_values JSONB,
    new_values JSONB,

    -- Context
    script_name VARCHAR(100),
    user_context TEXT,

    -- Timestamp
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index
CREATE INDEX idx_audit_log_table ON audit_log(table_name);
CREATE INDEX idx_audit_log_created ON audit_log(created_at DESC);
CREATE INDEX idx_audit_log_operation ON audit_log(operation);

-- =====================================================
-- FUNCTIONS AND TRIGGERS
-- =====================================================

-- Function to update 'updated_at' timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply updated_at trigger to all tables
CREATE TRIGGER update_game_statistics_updated_at BEFORE UPDATE ON game_statistics
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_game_schedule_updated_at BEFORE UPDATE ON game_schedule
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_predictions_updated_at BEFORE UPDATE ON predictions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_betting_statistics_updated_at BEFORE UPDATE ON betting_statistics
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_enriched_predictions_updated_at BEFORE UPDATE ON enriched_predictions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_teams_updated_at BEFORE UPDATE ON teams
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_model_versions_updated_at BEFORE UPDATE ON model_versions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- =====================================================
-- VIEWS FOR COMMON QUERIES
-- =====================================================

-- View: Latest predictions with enriched data
CREATE OR REPLACE VIEW v_latest_predictions AS
SELECT
    p.id,
    p.home_team,
    p.away_team,
    p.date,
    p.home_team_prob,
    p.odds_1,
    p.odds_2,
    p.result,
    e.prob_platt,
    e.prob_iso,
    e.stake_raw,
    e.stake_platt,
    e.stake_iso,
    e.win,
    e.pnl_raw,
    e.pnl_platt,
    e.pnl_iso,
    e.home_win_rate,
    p.prediction_date,
    e.enrichment_date
FROM predictions p
LEFT JOIN enriched_predictions e ON p.id = e.prediction_id
WHERE p.prediction_date = (SELECT MAX(prediction_date) FROM predictions)
ORDER BY p.date DESC;

-- View: Betting performance summary
CREATE OR REPLACE VIEW v_betting_performance AS
SELECT
    COUNT(*) as total_bets,
    SUM(CASE WHEN win = TRUE THEN 1 ELSE 0 END) as wins,
    SUM(CASE WHEN win = FALSE THEN 1 ELSE 0 END) as losses,
    ROUND(SUM(CASE WHEN win = TRUE THEN 1 ELSE 0 END)::NUMERIC / NULLIF(COUNT(*), 0) * 100, 2) as win_rate,
    ROUND(SUM(pnl_raw), 2) as total_pnl_raw,
    ROUND(SUM(pnl_platt), 2) as total_pnl_platt,
    ROUND(SUM(pnl_iso), 2) as total_pnl_iso,
    ROUND(AVG(stake_raw), 4) as avg_stake_raw,
    ROUND(AVG(stake_platt), 4) as avg_stake_platt,
    ROUND(AVG(stake_iso), 4) as avg_stake_iso
FROM enriched_predictions
WHERE stake_raw > 0 OR stake_platt > 0 OR stake_iso > 0;

-- View: Team performance summary
CREATE OR REPLACE VIEW v_team_performance AS
SELECT
    team,
    season,
    COUNT(*) as games_played,
    SUM(CASE WHEN won = TRUE THEN 1 ELSE 0 END) as wins,
    SUM(CASE WHEN won = FALSE THEN 1 ELSE 0 END) as losses,
    ROUND(SUM(CASE WHEN won = TRUE THEN 1 ELSE 0 END)::NUMERIC / NULLIF(COUNT(*), 0) * 100, 2) as win_rate,
    SUM(CASE WHEN home = 1 AND won = TRUE THEN 1 ELSE 0 END) as home_wins,
    SUM(CASE WHEN home = 1 THEN 1 ELSE 0 END) as home_games,
    ROUND(SUM(CASE WHEN home = 1 AND won = TRUE THEN 1 ELSE 0 END)::NUMERIC /
          NULLIF(SUM(CASE WHEN home = 1 THEN 1 ELSE 0 END), 0) * 100, 2) as home_win_rate,
    ROUND(AVG(total), 2) as avg_points,
    ROUND(AVG(total_opp), 2) as avg_points_allowed
FROM game_statistics
GROUP BY team, season
ORDER BY season DESC, win_rate DESC;

-- =====================================================
-- ROW LEVEL SECURITY (Optional - for Supabase)
-- =====================================================

-- Enable RLS on all tables (optional, can be configured later)
-- ALTER TABLE game_statistics ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE game_schedule ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE predictions ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE betting_statistics ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE enriched_predictions ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE teams ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE model_versions ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE audit_log ENABLE ROW LEVEL SECURITY;

-- =====================================================
-- INITIAL DATA: TEAMS
-- =====================================================

INSERT INTO teams (code, full_name, alternate_codes, conference, division) VALUES
('ATL', 'Atlanta Hawks', ARRAY['ATL'], 'East', 'Southeast'),
('BOS', 'Boston Celtics', ARRAY['BOS'], 'East', 'Atlantic'),
('BRK', 'Brooklyn Nets', ARRAY['BRK', 'BKN'], 'East', 'Atlantic'),
('CHO', 'Charlotte Hornets', ARRAY['CHO', 'CHA'], 'East', 'Southeast'),
('CHI', 'Chicago Bulls', ARRAY['CHI'], 'East', 'Central'),
('CLE', 'Cleveland Cavaliers', ARRAY['CLE'], 'East', 'Central'),
('DAL', 'Dallas Mavericks', ARRAY['DAL'], 'West', 'Southwest'),
('DEN', 'Denver Nuggets', ARRAY['DEN'], 'West', 'Northwest'),
('DET', 'Detroit Pistons', ARRAY['DET'], 'East', 'Central'),
('GSW', 'Golden State Warriors', ARRAY['GSW', 'GS'], 'West', 'Pacific'),
('HOU', 'Houston Rockets', ARRAY['HOU'], 'West', 'Southwest'),
('IND', 'Indiana Pacers', ARRAY['IND'], 'East', 'Central'),
('LAC', 'Los Angeles Clippers', ARRAY['LAC'], 'West', 'Pacific'),
('LAL', 'Los Angeles Lakers', ARRAY['LAL'], 'West', 'Pacific'),
('MEM', 'Memphis Grizzlies', ARRAY['MEM'], 'West', 'Southwest'),
('MIA', 'Miami Heat', ARRAY['MIA'], 'East', 'Southeast'),
('MIL', 'Milwaukee Bucks', ARRAY['MIL'], 'East', 'Central'),
('MIN', 'Minnesota Timberwolves', ARRAY['MIN'], 'West', 'Northwest'),
('NOP', 'New Orleans Pelicans', ARRAY['NOP', 'NO'], 'West', 'Southwest'),
('NYK', 'New York Knicks', ARRAY['NYK', 'NY'], 'East', 'Atlantic'),
('OKC', 'Oklahoma City Thunder', ARRAY['OKC', 'OKL'], 'West', 'Northwest'),
('ORL', 'Orlando Magic', ARRAY['ORL'], 'East', 'Southeast'),
('PHI', 'Philadelphia 76ers', ARRAY['PHI'], 'East', 'Atlantic'),
('PHX', 'Phoenix Suns', ARRAY['PHX', 'PHO'], 'West', 'Pacific'),
('POR', 'Portland Trail Blazers', ARRAY['POR'], 'West', 'Northwest'),
('SAC', 'Sacramento Kings', ARRAY['SAC'], 'West', 'Pacific'),
('SAS', 'San Antonio Spurs', ARRAY['SAS', 'SA'], 'West', 'Southwest'),
('TOR', 'Toronto Raptors', ARRAY['TOR'], 'East', 'Atlantic'),
('UTA', 'Utah Jazz', ARRAY['UTA', 'UTAH'], 'West', 'Northwest'),
('WAS', 'Washington Wizards', ARRAY['WAS'], 'East', 'Southeast')
ON CONFLICT (code) DO NOTHING;

-- =====================================================
-- COMMENTS FOR DOCUMENTATION
-- =====================================================

COMMENT ON TABLE game_statistics IS 'Historical NBA game statistics scraped from Basketball-Reference';
COMMENT ON TABLE game_schedule IS 'Upcoming NBA game schedules';
COMMENT ON TABLE predictions IS 'ML model predictions for upcoming games';
COMMENT ON TABLE betting_statistics IS 'Betting performance tracking and accuracy';
COMMENT ON TABLE enriched_predictions IS 'Calibrated predictions with Kelly Criterion stakes';
COMMENT ON TABLE teams IS 'NBA team metadata and code mappings';
COMMENT ON TABLE model_versions IS 'ML model version tracking and performance';
COMMENT ON TABLE audit_log IS 'Audit trail of all data changes';

-- =====================================================
-- GRANT PERMISSIONS (adjust as needed)
-- =====================================================

-- Grant read access to authenticated users (Supabase specific)
-- GRANT SELECT ON ALL TABLES IN SCHEMA public TO authenticated;
-- GRANT INSERT, UPDATE ON predictions, betting_statistics, enriched_predictions TO authenticated;
