#!/usr/bin/env node
'use strict';

const fs = require('fs');
const path = require('path');

const repoRoot = path.resolve(__dirname, '..');

// Basketball_prediction outputs live here (default):
//   <repo>/2026/output/LightGBM
// You can override with env.LGBM_DIR.
const outputDir = path.resolve(
  process.env.LGBM_DIR || path.join(repoRoot, '2026', 'output', 'LightGBM')
);

// Dashboard expects files here:
const webDataDir = path.join(repoRoot, 'web', 'public', 'data');

const REQUIRED_WINDOW_SIZE = 200;

function ensureDir(dirPath) {
  fs.mkdirSync(dirPath, { recursive: true });
}

function parseStrategyParams(txt) {
  const params = {};
  txt.split(/\r?\n/).forEach((line) => {
    if (!line.trim()) return;
    const [key, value] = line.split('=');
    if (!key) return;
    const trimmedValue = value?.trim();
    if (trimmedValue === undefined) return;
    const num = Number(trimmedValue);
    params[key.trim()] = Number.isNaN(num) ? trimmedValue : num;
  });
  return params;
}

function parseCsv(csvText) {
  const lines = csvText.trim().split(/\r?\n/);
  if (!lines.length) return [];
  const headers = splitCsvLine(lines[0]);
  return lines.slice(1).map((line) => {
    const cells = splitCsvLine(line);
    const row = {};
    headers.forEach((header, idx) => {
      row[header] = cells[idx] ?? '';
    });
    return row;
  });
}

function splitCsvLine(line) {
  const cells = [];
  let current = '';
  let inQuotes = false;
  for (let i = 0; i < line.length; i += 1) {
    const char = line[i];
    if (char === '"') {
      if (inQuotes && line[i + 1] === '"') {
        current += '"';
        i += 1;
      } else {
        inQuotes = !inQuotes;
      }
    } else if (char === ',' && !inQuotes) {
      cells.push(current);
      current = '';
    } else {
      current += char;
    }
  }
  cells.push(current);
  return cells.map((cell) => cell.trim());
}

function findLatestFile(dir, prefix, suffix = '.csv') {
  if (!fs.existsSync(dir)) return null;
  const entries = fs.readdirSync(dir);
  const candidates = entries
    .filter((name) => name.startsWith(prefix) && name.endsWith(suffix))
    .map((name) => {
      const match = name.match(/(\d{4}-\d{2}-\d{2})/);
      return match ? { name, date: match[1] } : null;
    })
    .filter(Boolean)
    .sort((a, b) => (a.date > b.date ? 1 : -1));

  if (!candidates.length) return null;
  return candidates[candidates.length - 1].name;
}

function formatNumber(value, decimals = 2) {
  if (value === null || value === undefined || Number.isNaN(value)) return '—';
  return Number(value).toFixed(decimals);
}

function formatMinEv(value) {
  if (value === null || value === undefined || Number.isNaN(value)) return '—';
  const num = Number(value);
  if (num < 0) {
    return `\u2212${Math.abs(num)}`;
  }
  return `${num}`;
}

function formatDateUTC(date) {
  return date.toISOString().slice(0, 10);
}

// NOTE: keeps your previous behavior: last N days in UTC ending "today" (runner time).
function determineWindowDates(windowSize) {
  const windowEndDate = new Date();
  const windowStartDate = new Date(windowEndDate);
  windowStartDate.setUTCDate(windowEndDate.getUTCDate() - (windowSize - 1));
  return {
    windowStart: formatDateUTC(windowStartDate),
    windowEnd: formatDateUTC(windowEndDate),
  };
}

function copyFile(source, target) {
  ensureDir(path.dirname(target));
  fs.copyFileSync(source, target);
}

function safeReadJson(filePath, label) {
  if (!fs.existsSync(filePath)) {
    throw new Error(`Missing required file: ${label} (${filePath})`);
  }
  const raw = fs.readFileSync(filePath, 'utf8');
  try {
    return JSON.parse(raw);
  } catch (e) {
    throw new Error(`Failed to parse JSON: ${label} (${filePath})`);
  }
}

function main() {
  ensureDir(webDataDir);

  // ----------------------------
  // Required: metrics snapshot
  // ----------------------------
  const metricsPath = path.join(outputDir, 'metrics_snapshot.json');
  const metricsSnapshot = safeReadJson(metricsPath, 'metrics_snapshot.json');

  // ----------------------------
  // Prefer ISO combined (Kelly/combined_nba_predictions_iso_*)
  // else fallback to ACC combined (combined_nba_predictions_acc_*)
  // ----------------------------
  const kellyDir = path.join(outputDir, 'Kelly');
  const combinedIsoName = fs.existsSync(kellyDir)
    ? findLatestFile(kellyDir, 'combined_nba_predictions_iso_')
    : null;

  const combinedAccName = findLatestFile(outputDir, 'combined_nba_predictions_acc_');

  const combinedLatestPath = combinedIsoName
    ? path.join(kellyDir, combinedIsoName)
    : (combinedAccName ? path.join(outputDir, combinedAccName) : null);

  if (!combinedLatestPath) {
    throw new Error(
      `No combined predictions found. Looked for:
- ${path.join(kellyDir, 'combined_nba_predictions_iso_*.csv')}
- ${path.join(outputDir, 'combined_nba_predictions_acc_*.csv')}`
    );
  }

  console.log(
    combinedIsoName
      ? `Using ISO combined for dashboard: ${path.join('Kelly', combinedIsoName)}`
      : `Using ACC combined for dashboard: ${combinedAccName}`
  );

  // ----------------------------
  // Window dates
  // ----------------------------
  const { windowStart, windowEnd } = determineWindowDates(REQUIRED_WINDOW_SIZE);

  // ----------------------------
  // local matched games
  // ----------------------------
  const localMatchedLatestName = findLatestFile(outputDir, 'local_matched_games_');
  if (!localMatchedLatestName) {
    throw new Error('No local_matched_games_*.csv found');
  }
  const localMatchedPath = path.join(outputDir, localMatchedLatestName);
  const localMatchedText = fs.readFileSync(localMatchedPath, 'utf8');
  const localMatchedRows = parseCsv(localMatchedText);
  const localMatchedDateKey = localMatchedRows[0]?.date
    ? 'date'
    : (localMatchedRows[0]?.game_date ? 'game_date' : 'date');

  // ----------------------------
  // bet log (optional)
  // ----------------------------
  const betLogPath = path.join(outputDir, 'bet_log_flat_live.csv');
  const betLogExists = fs.existsSync(betLogPath);

  // ----------------------------
  // strategy_params.txt -> strategy_params.json (optional)
  // ----------------------------
  const strategyParamsTxtPath = path.join(outputDir, 'strategy_params.txt');
  let strategyParams = {};
  if (fs.existsSync(strategyParamsTxtPath)) {
    const raw = fs.readFileSync(strategyParamsTxtPath, 'utf8');
    strategyParams = parseStrategyParams(raw);
  }

  const asOfDate =
    strategyParams.as_of_date ||
    metricsSnapshot?.meta?.eval_base_date_max ||
    windowEnd;

  // ----------------------------
  // Active filters text
  // ----------------------------
  const filterParams = metricsSnapshot.filter_params || {};
  const activeFiltersText = [
    `HW \u2265 ${formatNumber(filterParams.home_win_rate_threshold, 2)}`,
    `odds ${formatNumber(filterParams.odds_min, 2)}\u2013${formatNumber(filterParams.odds_max, 2)}`,
    `p \u2265 ${formatNumber(filterParams.prob_threshold, 2)}`,
    `EV > ${formatMinEv(filterParams.min_EV)}`,
    `window ${REQUIRED_WINDOW_SIZE} days (${windowStart || '—'} \u2192 ${windowEnd || '—'})`,
  ].join(' | ');

  // ----------------------------
  // Windowed local matches count + as_of_date from matches
  // ----------------------------
  const inWindowLocalMatches = localMatchedRows.filter((row) => {
    const dateValue = row[localMatchedDateKey];
    if (!dateValue || !windowStart || !windowEnd) return false;
    return dateValue >= windowStart && dateValue <= windowEnd;
  });

  const strategyAsOfDate = inWindowLocalMatches.length
    ? inWindowLocalMatches.map((row) => row[localMatchedDateKey]).sort().slice(-1)[0]
    : (localMatchedRows.map((row) => row[localMatchedDateKey]).sort().slice(-1)[0] || null);

  // ----------------------------
  // dashboard_state.json
  // ----------------------------
  const dashboardState = {
    as_of_date: asOfDate,
    window_size: REQUIRED_WINDOW_SIZE,
    window_start: windowStart,
    window_end: windowEnd,
    active_filters_text: activeFiltersText,
    params_used_label: 'Historical',
    params_source_label: metricsSnapshot.params_used_type || 'Unknown',
    strategy_as_of_date: strategyAsOfDate,
    last_update_utc: new Date().toISOString(),
    source_files: {
      combined: 'combined_latest.csv',
      combined_source: combinedIsoName
        ? `Kelly/${combinedIsoName}`
        : (combinedAccName || null),
      local_matched: 'local_matched_games_latest.csv',
      local_matched_source: localMatchedLatestName,
      bet_log: betLogExists ? 'bet_log_flat_live.csv' : null,
    },
    strategy_matches_window: inWindowLocalMatches.length,
  };

  // ----------------------------
  // Copy artifacts into web/public/data
  // ----------------------------
  copyFile(metricsPath, path.join(webDataDir, 'metrics_snapshot.json'));
  copyFile(combinedLatestPath, path.join(webDataDir, 'combined_latest.csv'));
  copyFile(localMatchedPath, path.join(webDataDir, 'local_matched_games_latest.csv'));

  if (betLogExists) {
    copyFile(betLogPath, path.join(webDataDir, 'bet_log_flat_live.csv'));
  }

  if (Object.keys(strategyParams).length) {
    fs.writeFileSync(
      path.join(webDataDir, 'strategy_params.json'),
      JSON.stringify(strategyParams, null, 2),
      'utf8'
    );
  }

  fs.writeFileSync(
    path.join(webDataDir, 'dashboard_state.json'),
    JSON.stringify(dashboardState, null, 2),
    'utf8'
  );

  // Small sanity output (helps debugging in Actions logs)
  const combinedHeader = fs.readFileSync(path.join(webDataDir, 'combined_latest.csv'), 'utf8')
    .split(/\r?\n/)[0];
  console.log('combined_latest.csv header:', combinedHeader);

  console.log('Dashboard assets prepared in web/public/data');
}

main();
