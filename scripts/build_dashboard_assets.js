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
const HARD_DEFAULT_PARAMS = {
  home_win_rate_threshold: 0.5,
  odds_min: 2.3,
  odds_max: 3.2,
  prob_threshold: 0.45,
  min_ev: 0,
};

function ensureDir(dirPath) {
  fs.mkdirSync(dirPath, { recursive: true });
}

function parseStrategyParams(txt) {
  const params = {};
  txt.split(/\r?\n/).forEach((line) => {
    if (!line.trim()) return;
    const separator = line.includes(':') ? ':' : '=';
    const [key, ...valueParts] = line.split(separator);
    if (!key) return;
    const trimmedValue = valueParts.join(separator).trim();
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

function resolveHeader(headers, candidates) {
  const headerSet = new Set(headers);
  for (const candidate of candidates) {
    if (headerSet.has(candidate)) return candidate;
  }
  const lowered = new Map(headers.map((header) => [header.toLowerCase(), header]));
  for (const candidate of candidates) {
    const resolved = lowered.get(candidate.toLowerCase());
    if (resolved) return resolved;
  }
  return null;
}

function validateLocalMatchedSchema(headers, delimiter) {
  const dateKey = resolveHeader(headers, ['date', 'game_date']);
  const homeKey = resolveHeader(headers, ['home_team', 'home', 'team_home']);
  const awayKey = resolveHeader(headers, ['away_team', 'away', 'team_away']);

  if (!dateKey) {
    throw new Error(
      `local_matched_games schema missing date column. Expected "date" or "game_date". ` +
        `Detected headers: [${headers.join(', ')}] (delimiter: ${delimiter === '\t' ? 'tab' : ','})`
    );
  }

  if (!homeKey || !awayKey) {
    throw new Error(
      `local_matched_games schema missing team identifiers. Expected home/away columns ` +
        `like "home_team"/"away_team" or "home"/"away" or "team_home"/"team_away". ` +
        `Detected headers: [${headers.join(', ')}] (delimiter: ${delimiter === '\t' ? 'tab' : ','})`
    );
  }

  return { dateKey, homeKey, awayKey };
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


function copyOptionalAgentArtifact(sourcePath, targetName, label, sources) {
  if (!fs.existsSync(sourcePath)) return null;
  const targetPath = path.join(webDataDir, targetName);
  copyFile(sourcePath, targetPath);
  const stat = fs.statSync(sourcePath);
  const entry = {
    label,
    path: targetName,
    source_path: path.relative(repoRoot, sourcePath),
    bytes: stat.size,
  };
  sources.push(entry);
  return entry;
}

function writeAgentManifest(sources) {
  const manifest = {
    version: 1,
    mode: 'read_only_mvp',
    generated_at_utc: new Date().toISOString(),
    read_only_sources: sources,
    frontend_contract: {
      endpoint_config: 'window.HOOPS_AGENT_API_URL or <meta name="hoops-agent-api" content="...">',
      request_shape: {
        question: 'string',
        capability: 'read_only',
        context: {
          dashboard_state_url: 'public/data/dashboard_state.json',
          metrics_url: 'public/data/metrics_snapshot.json',
          agent_manifest_url: 'public/data/agent_manifest.json',
        },
      },
    },
    guardrails: {
      browser_secrets_allowed: false,
      allowed: [
        'read latest CSV and JSON outputs',
        'summarize today board and no-bet reasons',
        'compare canonical, setup-profitability, near-miss, and vibe candidates',
        'summarize settled manual bet rows',
        'prepare draft Steadivus log entries',
      ],
      disallowed: [
        'place bets',
        'access betting accounts',
        'run arbitrary shell commands',
        'push code or mutate history without explicit confirmation',
        'store OpenAI or GitHub secrets in browser code',
      ],
    },
  };
  fs.writeFileSync(path.join(webDataDir, 'agent_manifest.json'), JSON.stringify(manifest, null, 2), 'utf8');
}

function coerceDateISO(value) {
  if (!value) return null;
  const raw = String(value).trim();
  if (!raw) return null;
  if (/^\d{4}-\d{2}-\d{2}$/.test(raw)) return raw;
  const parsed = new Date(raw);
  if (Number.isNaN(parsed.getTime())) return null;
  return parsed.toISOString().slice(0, 10);
}

function extractDateFromName(fileName) {
  const match = String(fileName).match(/(\d{4}-\d{2}-\d{2})/);
  return match ? match[1] : null;
}

function listDatedCandidates(dir, prefix, suffix) {
  if (!fs.existsSync(dir)) return [];
  return fs
    .readdirSync(dir)
    .filter((name) => name.startsWith(prefix) && name.endsWith(suffix))
    .map((name) => ({ name, date: extractDateFromName(name) }))
    .filter((entry) => Boolean(entry.date))
    .sort((a, b) => (a.date > b.date ? 1 : -1));
}

function selectDatedAtOrBefore(candidates, snapshotDate) {
  if (!candidates.length) return null;
  if (!snapshotDate) return candidates[candidates.length - 1];
  const filtered = candidates.filter((entry) => entry.date <= snapshotDate);
  return filtered.length ? filtered[filtered.length - 1] : null;
}

function resolveParamsSource(outputRoot, snapshotDate) {
  const metricsCandidates = listDatedCandidates(outputRoot, 'metrics_snapshot_', '.json');
  const exactMetrics = metricsCandidates.find((entry) => entry.date === snapshotDate);
  const datedMetrics = exactMetrics || selectDatedAtOrBefore(metricsCandidates, snapshotDate);
  if (datedMetrics) {
    return {
      sourceType: 'metrics_snapshot_dated',
      filePath: path.join(outputRoot, datedMetrics.name),
      artifactDate: datedMetrics.date,
    };
  }

  const strategyJsonCandidates = listDatedCandidates(outputRoot, 'strategy_params_', '.json');
  const exactStrategyJson = strategyJsonCandidates.find((entry) => entry.date === snapshotDate);
  const datedStrategyJson =
    exactStrategyJson || selectDatedAtOrBefore(strategyJsonCandidates, snapshotDate);
  if (datedStrategyJson) {
    return {
      sourceType: 'strategy_params_dated_json',
      filePath: path.join(outputRoot, datedStrategyJson.name),
      artifactDate: datedStrategyJson.date,
    };
  }

  const strategyTxtCandidates = listDatedCandidates(outputRoot, 'strategy_params_', '.txt');
  const exactStrategyTxt = strategyTxtCandidates.find((entry) => entry.date === snapshotDate);
  const datedStrategyTxt = exactStrategyTxt || selectDatedAtOrBefore(strategyTxtCandidates, snapshotDate);
  if (datedStrategyTxt) {
    return {
      sourceType: 'strategy_params_dated_txt',
      filePath: path.join(outputRoot, datedStrategyTxt.name),
      artifactDate: datedStrategyTxt.date,
    };
  }

  return {
    sourceType: 'default',
    filePath: null,
    artifactDate: null,
  };
}

function resolveLocalMatchedSource(outputRoot, webRoot, snapshotDate) {
  const datedCandidates = listDatedCandidates(outputRoot, 'local_matched_games_', '.csv');
  const exact = datedCandidates.find((entry) => entry.date === snapshotDate);
  const dated = exact || selectDatedAtOrBefore(datedCandidates, snapshotDate);
  if (dated) {
    return {
      filePath: path.join(outputRoot, dated.name),
      sourceLabel: `output/${dated.name}`,
    };
  }

  const latest = findLatestFile(outputRoot, 'local_matched_games_');
  if (latest) {
    return {
      filePath: path.join(outputRoot, latest),
      sourceLabel: `output/${latest}`,
    };
  }

  const deployed = path.join(webRoot, 'local_matched_games_latest.csv');
  if (fs.existsSync(deployed)) {
    return {
      filePath: deployed,
      sourceLabel: 'web/public/data/local_matched_games_latest.csv',
    };
  }
  return null;
}

function readParamsPayload(source) {
  if (!source.filePath) return {};
  if (source.filePath.endsWith('.json')) {
    return safeReadJson(source.filePath, source.filePath);
  }
  return parseStrategyParams(fs.readFileSync(source.filePath, 'utf8'));
}

function isPlayedRow(row) {
  const result = row.result ?? row.result_raw ?? '';
  const trimmed = String(result).trim();
  return trimmed !== '' && trimmed !== '0';
}

function computeWindowFromPlayedGames(combinedRows, windowSize) {
  const playedDates = combinedRows
    .filter(isPlayedRow)
    .map((row) => coerceDateISO(row.game_date ?? row.date))
    .filter(Boolean)
    .sort();
  if (!playedDates.length) {
    throw new Error('No played games found in combined file for window computation.');
  }
  const slice = playedDates.slice(-windowSize);
  if (!slice.length) {
    throw new Error('Window selection produced zero rows.');
  }
  return {
    windowStart: slice[0],
    windowEnd: slice[slice.length - 1],
    playedCount: playedDates.length,
  };
}

function main() {
  ensureDir(webDataDir);

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

  const combinedRawText = fs.readFileSync(combinedLatestPath, 'utf8');
  const combinedParsed = parseDelimitedWithDelimiter(combinedRawText);
  const combinedHeaders = combinedParsed.headers;
  const combinedDelimiter = combinedParsed.delimiter;
  const combinedRows = combinedParsed.rows;
  const { windowStart, windowEnd, playedCount } = computeWindowFromPlayedGames(
    combinedRows,
    REQUIRED_WINDOW_SIZE
  );
  console.log(`Computed window from ${playedCount} played games.`);

  const selectedSnapshotDate = process.env.SNAPSHOT_DATE || windowEnd;
  const paramsSource = resolveParamsSource(outputDir, selectedSnapshotDate);
  const paramsPayload = readParamsPayload(paramsSource);
  const strategyParams = paramsSource.sourceType.includes('strategy_params')
    ? paramsPayload
    : {};
  const metricsSnapshot = paramsSource.sourceType === 'metrics_snapshot_dated'
    ? paramsPayload
    : null;
  const paramsUsed =
    metricsSnapshot?.params_used ||
    (Object.keys(strategyParams).length ? strategyParams : HARD_DEFAULT_PARAMS);
  const sourceFallbackUsed = Boolean(metricsSnapshot?.fallback_used ?? strategyParams?.fallback_used);
  const sourceFallbackReason = metricsSnapshot?.fallback_reason || strategyParams?.fallback_reason || null;
  const asOfDate =
    strategyParams.as_of_date ||
    metricsSnapshot?.as_of_date ||
    metricsSnapshot?.meta?.eval_base_date_max ||
    paramsSource.artifactDate ||
    selectedSnapshotDate ||
    new Date().toISOString().slice(0, 10);

  // ----------------------------
  // local matched games: prefer dated output artifact aligned to snapshot date,
  // then fallback to latest output artifact, and only then deployed file.
  // ----------------------------
  const resolvedLocalMatched = resolveLocalMatchedSource(outputDir, webDataDir, selectedSnapshotDate);
  if (!resolvedLocalMatched) {
    throw new Error('No local_matched_games source found (dated output, latest output, or deployed latest).');
  }
  const localMatchedSourcePath = resolvedLocalMatched.filePath;
  console.log(`Using local matched source: ${resolvedLocalMatched.sourceLabel}`);

  const localMatchedText = fs.readFileSync(localMatchedSourcePath, 'utf8');
  const localMatchedParsed = parseDelimitedWithDelimiter(localMatchedText);
  const localMatchedHeaders = localMatchedParsed.headers;
  const localMatchedRows = localMatchedParsed.rows;
  const { dateKey: localMatchedDateKey } = validateLocalMatchedSchema(
    localMatchedHeaders,
    localMatchedParsed.delimiter
  );

  const betLogCandidates = [
    path.join(repoRoot, '2026', 'bet_log', 'bet_log_flat_live.csv'),
    path.join(outputDir, 'bet_log_flat_live.csv'),
    path.join(webDataDir, 'bet_log_flat_live.csv'),
  ].filter((candidate, index, arr) => fs.existsSync(candidate) && arr.indexOf(candidate) === index);
  const canonicalBetLogPath = path.join(repoRoot, '2026', 'bet_log', 'bet_log_flat_live.csv');
  let selectedBetLogPath = fs.existsSync(canonicalBetLogPath) ? canonicalBetLogPath : (betLogCandidates[0] || null);
  let betLogLatestDateInFile = null;
  let betLogFreshnessWarning = null;
  if (selectedBetLogPath) {
    const scanBetLogDate = (filePath) => {
      const parsed = parseDelimitedWithDelimiter(fs.readFileSync(filePath, 'utf8'));
      if (!parsed.rows.length) return null;
      const dateKey = resolveHeader(parsed.headers, ['date', 'game_date']);
      if (!dateKey) return null;
      return parsed.rows
        .map((row) => coerceDateISO(row[dateKey]))
        .filter(Boolean)
        .sort()
        .slice(-1)[0] || null;
    };
    const datedCandidates = betLogCandidates.map((filePath) => ({
      filePath,
      latestDate: scanBetLogDate(filePath),
    }));
    const freshest = datedCandidates
      .filter((entry) => entry.latestDate)
      .sort((a, b) => (a.latestDate > b.latestDate ? -1 : 1))[0] || null;
    betLogLatestDateInFile = scanBetLogDate(selectedBetLogPath);
    if (
      freshest &&
      freshest.filePath !== selectedBetLogPath &&
      freshest.latestDate &&
      betLogLatestDateInFile &&
      freshest.latestDate > betLogLatestDateInFile
    ) {
      betLogFreshnessWarning =
        `Canonical bet log is stale (${path.relative(repoRoot, selectedBetLogPath)}:${betLogLatestDateInFile}) ` +
        `vs newer candidate (${path.relative(repoRoot, freshest.filePath)}:${freshest.latestDate}).`;
      console.warn(betLogFreshnessWarning);
    }
  }

  // ----------------------------
  // Active filters text
  // ----------------------------
  const minEV = (strategyParams.min_ev !== undefined) ? strategyParams.min_ev : paramsUsed.min_EV;

  const activeFiltersParts = [
    `HW \u2265 ${formatNumber(paramsUsed.home_win_rate_threshold, 2)}`,
    `odds ${formatNumber(paramsUsed.odds_min, 2)}\u2013${formatNumber(paramsUsed.odds_max, 2)}`,
    `p \u2265 ${formatNumber(paramsUsed.prob_threshold, 2)}`,
    `EV > ${formatMinEv(minEV)}`,
    `window ${REQUIRED_WINDOW_SIZE} games (${windowStart} \u2192 ${windowEnd})`,
  ];
  if (sourceFallbackUsed) {
    activeFiltersParts.push(`fallback (${sourceFallbackReason || 'safe_fallback_used'})`);
  }
  const activeFiltersText = activeFiltersParts.join(' | ');

  // ----------------------------
  // Windowed local matches count
  // ----------------------------
  let inWindowLocalMatches = [];
  if (!localMatchedRows.length) {
    console.warn(
      'Warning: local_matched_games_latest.csv has no rows; strategy_matches_window will be set to 0.'
    );
  } else {
    inWindowLocalMatches = localMatchedRows.filter((row) => {
      const dateValue = row[localMatchedDateKey];
      if (!dateValue) return false;
      return dateValue >= windowStart && dateValue <= windowEnd;
    });
  }

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
    snapshot_date_selected: selectedSnapshotDate,
    params_used_label: 'Historical',
    params_source_label: metricsSnapshot?.params_used_type || (paramsSource.sourceType === 'default' ? 'default' : 'strategy_params'),
    params_used: paramsUsed,
    effective_params: {
      as_of_date: asOfDate,
      window_size: REQUIRED_WINDOW_SIZE,
      home_win_rate_threshold:
        paramsUsed.home_win_rate_threshold ?? strategyParams.home_win_rate_threshold ?? null,
      odds_min: paramsUsed.odds_min ?? strategyParams.odds_min ?? null,
      odds_max: paramsUsed.odds_max ?? strategyParams.odds_max ?? null,
      prob_threshold: paramsUsed.prob_threshold ?? strategyParams.prob_threshold ?? null,
      min_ev: minEV !== undefined ? minEV : (strategyParams.min_ev ?? null),
      stake: strategyParams.stake ?? null,
      n_trades: strategyParams.n_trades ?? null,
      profit_eur: strategyParams['profit_€'] ?? strategyParams.profit_eur ?? null,
      roi_pct: strategyParams['roi_%'] ?? strategyParams.roi_pct ?? null,
    },
    params_source_file: paramsSource.filePath ? path.relative(repoRoot, paramsSource.filePath) : null,
    params_source_type: paramsSource.sourceType,
    params_artifact_date: paramsSource.artifactDate,
    fallback_used: sourceFallbackUsed || paramsSource.sourceType === 'default',
    fallback_reason: sourceFallbackUsed
      ? sourceFallbackReason
      : (paramsSource.sourceType === 'default' ? 'hardcoded_defaults' : null),
    params_sources: {
      metrics_snapshot: paramsSource.sourceType === 'metrics_snapshot_dated'
        ? path.basename(paramsSource.filePath)
        : null,
      strategy_params_json: fs.existsSync(path.join(webDataDir, 'strategy_params.json'))
        ? 'strategy_params.json'
        : null,
      strategy_params_txt: paramsSource.sourceType.includes('strategy_params')
        ? path.basename(paramsSource.filePath)
        : null,
    },
    strategy_as_of_date: strategyAsOfDate,
    bet_log_source_file: selectedBetLogPath ? path.relative(repoRoot, selectedBetLogPath) : null,
    bet_log_latest_date_in_file: betLogLatestDateInFile,
    last_update_utc: new Date().toISOString(),
    source_files: {
      combined: 'combined_latest.csv',
      combined_source: combinedIsoName ? `Kelly/${combinedIsoName}` : (combinedAccName || null),
      local_matched: 'local_matched_games_latest.csv',
      local_matched_source: path.relative(repoRoot, localMatchedSourcePath),
      bet_log: selectedBetLogPath ? 'bet_log_flat_live.csv' : null,
      bet_log_freshness_warning: betLogFreshnessWarning,
    },
    strategy_matches_window: inWindowLocalMatches.length,
  };

  // ----------------------------
  // Copy artifacts into web/public/data
  // ----------------------------
  if (paramsSource.sourceType === 'metrics_snapshot_dated' && paramsSource.filePath) {
    copyFile(paramsSource.filePath, path.join(webDataDir, 'metrics_snapshot.json'));
  } else {
    const synthesizedMetrics = {
      meta: {
        eval_base_date_max: asOfDate,
        params_source: paramsSource.filePath ? path.relative(repoRoot, paramsSource.filePath) : 'hardcoded_defaults',
      },
      params_used_type: dashboardState.params_source_label,
      params_used: paramsUsed,
      fallback_used: dashboardState.fallback_used,
      fallback_reason: dashboardState.fallback_reason,
      params_source: dashboardState.params_source_file,
    };
    fs.writeFileSync(
      path.join(webDataDir, 'metrics_snapshot.json'),
      JSON.stringify(synthesizedMetrics, null, 2),
      'utf8'
    );
  }
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

  const agentSources = [
    {
      label: 'Combined predictions latest',
      path: 'combined_latest.csv',
      source_path: combinedIsoName ? `2026/output/LightGBM/Kelly/${combinedIsoName}` : `2026/output/LightGBM/${combinedAccName}`,
    },
    {
      label: 'Local matched games latest',
      path: 'local_matched_games_latest.csv',
      source_path: path.relative(repoRoot, localMatchedSourcePath),
    },
    {
      label: 'Metrics snapshot',
      path: 'metrics_snapshot.json',
      source_path: dashboardState.params_source_file || 'web/public/data/metrics_snapshot.json',
    },
  ];

  if (selectedBetLogPath) {
    copyFile(selectedBetLogPath, path.join(webDataDir, 'bet_log_flat_live.csv'));
    agentSources.push({
      label: 'Real placed bets ledger',
      path: 'bet_log_flat_live.csv',
      source_path: path.relative(repoRoot, selectedBetLogPath),
    });
  }

  copyOptionalAgentArtifact(
    path.join(outputDir, 'betting_agent_stage1', 'stage1_daily_snapshot_latest.csv'),
    'stage1_daily_snapshot_latest.csv',
    'Stage 1 daily snapshot',
    agentSources
  );
  copyOptionalAgentArtifact(
    path.join(outputDir, 'betting_agent_stage1', 'stage1_daily_snapshot_latest.json'),
    'stage1_daily_snapshot_latest.json',
    'Stage 1 daily summary',
    agentSources
  );
  copyOptionalAgentArtifact(
    path.join(outputDir, 'setup_profitability_scan_latest.csv'),
    'setup_profitability_scan_latest.csv',
    'Setup profitability scan',
    agentSources
  );
  copyOptionalAgentArtifact(
    path.join(outputDir, 'setup_profitability_scan_summary_latest.json'),
    'setup_profitability_scan_summary_latest.json',
    'Setup profitability scan summary',
    agentSources
  );
  copyOptionalAgentArtifact(
    path.join(outputDir, 'script11_watchlist_history_latest.csv'),
    'script11_watchlist_history_latest.csv',
    'Script 11 watchlist history',
    agentSources
  );
  copyOptionalAgentArtifact(
    path.join(outputDir, 'script11_watchlist_history_summary_latest.json'),
    'script11_watchlist_history_summary_latest.json',
    'Script 11 watchlist history summary',
    agentSources
  );
  copyOptionalAgentArtifact(
    path.join(outputDir, 'actual_bets_manual.csv'),
    'actual_bets_manual.csv',
    'Manual actual bets log',
    agentSources
  );
  writeAgentManifest(agentSources);

  const strategyParamsForDashboard = {
    as_of_date: asOfDate,
    min_ev: minEV !== undefined ? minEV : (strategyParams.min_ev ?? null),
    stake: strategyParams.stake ?? null,
    params_used: {
      home_win_rate_threshold: paramsUsed.home_win_rate_threshold ?? null,
      odds_min: paramsUsed.odds_min ?? null,
      odds_max: paramsUsed.odds_max ?? null,
      prob_threshold: paramsUsed.prob_threshold ?? null,
    },
    source_type: paramsSource.sourceType,
    source_file: paramsSource.filePath ? path.relative(repoRoot, paramsSource.filePath) : null,
  };
  fs.writeFileSync(
    path.join(webDataDir, 'strategy_params.json'),
    JSON.stringify(strategyParamsForDashboard, null, 2),
    'utf8'
  );

  fs.writeFileSync(
    path.join(webDataDir, 'dashboard_state.json'),
    JSON.stringify(dashboardState, null, 2),
    'utf8'
  );

  if (paramsSource.artifactDate && paramsSource.artifactDate > selectedSnapshotDate) {
    throw new Error(
      `Snapshot consistency check failed: params artifact date ${paramsSource.artifactDate} > selected snapshot ${selectedSnapshotDate}.`
    );
  }
  if (!dashboardState.fallback_used && dashboardState.fallback_reason) {
    throw new Error('Snapshot consistency check failed: fallback_reason set while fallback_used=false.');
  }
  if (
    paramsSource.sourceType === 'metrics_snapshot_dated' &&
    paramsSource.artifactDate &&
    paramsSource.artifactDate !== selectedSnapshotDate
  ) {
    console.warn(
      `Snapshot selection warning: selected snapshot ${selectedSnapshotDate} resolved to dated metrics artifact ${paramsSource.artifactDate}.`
    );
  }
  if (
    metricsSnapshot?.meta?.eval_base_date_max &&
    paramsSource.artifactDate &&
    metricsSnapshot.meta.eval_base_date_max !== paramsSource.artifactDate
  ) {
    throw new Error(
      `Snapshot consistency check failed: metrics meta eval_base_date_max (${metricsSnapshot.meta.eval_base_date_max}) ` +
      `!= artifact date (${paramsSource.artifactDate}).`
    );
  }

  // Sanity output
  const combinedHeader = fs.readFileSync(path.join(webDataDir, 'combined_latest.csv'), 'utf8')
    .split(/\r?\n/)[0];
  console.log('combined_latest.csv header:', combinedHeader);

  console.log('local_matched_games_latest.csv rows:', localMatchedRows.length);
  console.log('strategy_matches_window:', inWindowLocalMatches.length);

  console.log('Dashboard assets prepared in web/public/data');
}

main();
