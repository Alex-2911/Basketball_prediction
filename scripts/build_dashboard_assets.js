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

const REQUIRED_WINDOW_SIZE = Number(process.env.N_WINDOW || 200);

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

function detectDelimiter(headerLine) {
  // Prefer tab if tabs exist and commas don't.
  const hasTab = headerLine.includes('\t');
  const hasComma = headerLine.includes(',');
  if (hasTab && !hasComma) return '\t';
  return ','; // default
}

function splitLine(line, delimiter) {
  // Minimal CSV/TSV splitter with quote support (works for comma or tab)
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
    } else if (char === delimiter && !inQuotes) {
      cells.push(current);
      current = '';
    } else {
      current += char;
    }
  }
  cells.push(current);
  return cells.map((c) => c.trim());
}

function parseDelimited(text) {
  const raw = text.trim();
  if (!raw) return [];
  const lines = raw.split(/\r?\n/);
  if (!lines.length) return [];
  const delimiter = detectDelimiter(lines[0]);
  const headers = splitLine(lines[0], delimiter);

  return lines.slice(1).map((line) => {
    const cells = splitLine(line, delimiter);
    const row = {};
    headers.forEach((h, idx) => {
      row[h] = cells[idx] ?? '';
    });
    return row;
  });
}

function parseDelimitedWithDelimiter(text) {
  const raw = text.trim();
  if (!raw) return { headers: [], rows: [], delimiter: ',' };
  const lines = raw.split(/\r?\n/);
  if (!lines.length) return { headers: [], rows: [], delimiter: ',' };
  const delimiter = detectDelimiter(lines[0]);
  const headers = splitLine(lines[0], delimiter);
  const rows = lines.slice(1).map((line) => {
    const cells = splitLine(line, delimiter);
    const row = {};
    headers.forEach((h, idx) => {
      row[h] = cells[idx] ?? '';
    });
    return row;
  });
  return { headers, rows, delimiter };
}

function escapeDelimitedValue(value, delimiter) {
  const stringValue = value === null || value === undefined ? '' : String(value);
  const needsQuotes =
    stringValue.includes('"') ||
    stringValue.includes('\n') ||
    stringValue.includes('\r') ||
    stringValue.includes(delimiter);
  if (!needsQuotes) return stringValue;
  return `"${stringValue.replace(/"/g, '""')}"`;
}

function serializeDelimited(headers, rows, delimiter) {
  if (!headers.length) return '';
  const lines = [];
  lines.push(headers.map((header) => escapeDelimitedValue(header, delimiter)).join(delimiter));
  rows.forEach((row) => {
    const line = headers.map((header) => escapeDelimitedValue(row[header] ?? '', delimiter)).join(delimiter);
    lines.push(line);
  });
  return `${lines.join('\n')}\n`;
}

function isMissing(value) {
  if (value === null || value === undefined) return true;
  const normalized = String(value).trim().toLowerCase();
  if (!normalized) return true;
  return ['nan', 'none', 'null', 'undefined'].includes(normalized);
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
  if (num < 0) return `\u2212${Math.abs(num)}`;
  return `${num}`;
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

function copyFile(source, target) {
  ensureDir(path.dirname(target));
  fs.copyFileSync(source, target);
}

function addDaysISO(isoDate, deltaDays) {
  const d = new Date(`${isoDate}T00:00:00Z`);
  d.setUTCDate(d.getUTCDate() + deltaDays);
  return d.toISOString().slice(0, 10);
}

// Window is last N days ending at as_of_date (NOT runner "today")
function determineWindowDatesFromAsOf(asOfDate, windowSize) {
  const windowEnd = asOfDate;
  const windowStart = addDaysISO(windowEnd, -(windowSize - 1));
  return { windowStart, windowEnd };
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
  // strategy_params.txt -> strategy_params.json (optional)
  // ----------------------------
  const strategyParamsTxtPath = path.join(outputDir, 'strategy_params.txt');
  let strategyParams = {};
  if (fs.existsSync(strategyParamsTxtPath)) {
    const raw = fs.readFileSync(strategyParamsTxtPath, 'utf8');
    strategyParams = parseStrategyParams(raw);
  }

  // ----------------------------
  // as_of_date: prefer strategy_params.txt, else metrics_snapshot meta
  // ----------------------------
  const asOfDate =
    strategyParams.as_of_date ||
    metricsSnapshot?.meta?.eval_base_date_max ||
    new Date().toISOString().slice(0, 10);

  // ----------------------------
  // Window dates based on as_of_date
  // ----------------------------
  const { windowStart, windowEnd } =
    determineWindowDatesFromAsOf(asOfDate, REQUIRED_WINDOW_SIZE);

  // ----------------------------
  // local matched games: use the DEPLOYED file if it exists,
  // otherwise fall back to latest local_matched_games_*.csv
  // ----------------------------
  const deployedLocalMatched = path.join(webDataDir, 'local_matched_games_latest.csv');

  let localMatchedSourcePath = null;

  if (fs.existsSync(deployedLocalMatched)) {
    localMatchedSourcePath = deployedLocalMatched;
    console.log('Using existing web/public/data/local_matched_games_latest.csv as source.');
  } else {
    const localMatchedLatestName = findLatestFile(outputDir, 'local_matched_games_');
    if (!localMatchedLatestName) {
      throw new Error('No local_matched_games_*.csv found and no deployed local_matched_games_latest.csv exists.');
    }
    localMatchedSourcePath = path.join(outputDir, localMatchedLatestName);
    console.log(`Using output local matched source: ${localMatchedLatestName}`);
  }

  const localMatchedText = fs.readFileSync(localMatchedSourcePath, 'utf8');
  const localMatchedRows = parseDelimited(localMatchedText);

  // detect date column key
  const firstRow = localMatchedRows[0] || {};
  const localMatchedDateKey =
    Object.prototype.hasOwnProperty.call(firstRow, 'date') ? 'date'
      : (Object.prototype.hasOwnProperty.call(firstRow, 'game_date') ? 'game_date' : 'date');

  // ----------------------------
  // bet log (optional)
  // ----------------------------
  const betLogPath = path.join(outputDir, 'bet_log_flat_live.csv');
  const betLogExists = fs.existsSync(betLogPath);

  // ----------------------------
  // Active filters text
  // Prefer your new Script 5 structure:
  // metricsSnapshot.params_used + strategyParams.min_ev
  // ----------------------------
  const paramsUsed = metricsSnapshot.params_used || metricsSnapshot.filter_params || {};
  const minEV = (strategyParams.min_ev !== undefined) ? strategyParams.min_ev : paramsUsed.min_EV;

  const activeFiltersText = [
    `HW \u2265 ${formatNumber(paramsUsed.home_win_rate_threshold, 2)}`,
    `odds ${formatNumber(paramsUsed.odds_min, 2)}\u2013${formatNumber(paramsUsed.odds_max, 2)}`,
    `p \u2265 ${formatNumber(paramsUsed.prob_threshold, 2)}`,
    `EV > ${formatMinEv(minEV)}`,
    `window ${REQUIRED_WINDOW_SIZE} days (${windowStart} \u2192 ${windowEnd})`,
  ].join(' | ');

  // ----------------------------
  // Windowed local matches count
  // ----------------------------
  const inWindowLocalMatches = localMatchedRows.filter((row) => {
    const dateValue = row[localMatchedDateKey];
    if (!dateValue) return false;
    return dateValue >= windowStart && dateValue <= windowEnd;
  });

  const strategyAsOfDate = localMatchedRows.length
    ? localMatchedRows.map((r) => r[localMatchedDateKey]).filter(Boolean).sort().slice(-1)[0]
    : null;

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
      combined_source: combinedIsoName ? `Kelly/${combinedIsoName}` : (combinedAccName || null),
      local_matched: 'local_matched_games_latest.csv',
      local_matched_source: path.relative(repoRoot, localMatchedSourcePath),
      bet_log: betLogExists ? 'bet_log_flat_live.csv' : null,
    },
    strategy_matches_window: inWindowLocalMatches.length,
  };

  // ----------------------------
  // Copy artifacts into web/public/data
  // ----------------------------
  copyFile(metricsPath, path.join(webDataDir, 'metrics_snapshot.json'));
  const combinedRawText = fs.readFileSync(combinedLatestPath, 'utf8');
  const combinedParsed = parseDelimitedWithDelimiter(combinedRawText);
  const combinedHeaders = combinedParsed.headers;
  const combinedDelimiter = combinedParsed.delimiter;
  const combinedRows = combinedParsed.rows;
  const filteredCombinedRows = combinedRows.filter((row) => {
    const resultRawValue = row.result_raw ?? '';
    const resultValue = row.result ?? '';
    const hasPlaceholderResult =
      ['0', '1'].includes(String(resultRawValue).trim()) ||
      ['0', '1'].includes(String(resultValue).trim());
    return !(hasPlaceholderResult && isMissing(row.home_team_won));
  });
  const filteredCount = combinedRows.length - filteredCombinedRows.length;
  if (filteredCount > 0) {
    console.log(`Filtered placeholder outcomes: removed ${filteredCount}/${combinedRows.length} rows`);
  } else {
    console.log(`No placeholder outcomes found (checked ${combinedRows.length} rows).`);
  }
  fs.writeFileSync(
    path.join(webDataDir, 'combined_latest.csv'),
    serializeDelimited(combinedHeaders, filteredCombinedRows, combinedDelimiter),
    'utf8'
  );

  // Always ensure deployed local_matched_games_latest.csv matches our chosen source
  copyFile(localMatchedSourcePath, path.join(webDataDir, 'local_matched_games_latest.csv'));

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

  // Sanity output
  const combinedHeader = fs.readFileSync(path.join(webDataDir, 'combined_latest.csv'), 'utf8')
    .split(/\r?\n/)[0];
  console.log('combined_latest.csv header:', combinedHeader);

  console.log('local_matched_games_latest.csv rows:', localMatchedRows.length);
  console.log('strategy_matches_window:', inWindowLocalMatches.length);

  console.log('Dashboard assets prepared in web/public/data');
}

main();
