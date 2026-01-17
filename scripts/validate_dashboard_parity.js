#!/usr/bin/env node
'use strict';

const fs = require('fs');
const path = require('path');

const repoRoot = path.resolve(__dirname, '..');
const dataDir = path.join(repoRoot, 'web', 'public', 'data');

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

function formatDateUTC(date) {
  return date.toISOString().slice(0, 10);
}

function computeWindowDates(windowSize) {
  const windowEndDate = new Date();
  const windowStartDate = new Date(windowEndDate);
  windowStartDate.setUTCDate(windowEndDate.getUTCDate() - (windowSize - 1));
  return {
    windowStart: formatDateUTC(windowStartDate),
    windowEnd: formatDateUTC(windowEndDate),
  };
}

function ensure(condition, message) {
  if (!condition) {
    throw new Error(message);
  }
}

function main() {
  const dashboardState = JSON.parse(fs.readFileSync(path.join(dataDir, 'dashboard_state.json'), 'utf8'));
  const localMatchedRows = parseCsv(fs.readFileSync(path.join(dataDir, 'local_matched_games_latest.csv'), 'utf8'));

  const { windowStart: computedWindowStart, windowEnd: computedWindowEnd } = computeWindowDates(
    dashboardState.window_size
  );

  ensure(
    computedWindowStart === dashboardState.window_start,
    `Window start mismatch: expected ${computedWindowStart}, got ${dashboardState.window_start}`
  );
  ensure(
    computedWindowEnd === dashboardState.window_end,
    `Window end mismatch: expected ${computedWindowEnd}, got ${dashboardState.window_end}`
  );

  const localDateKey = localMatchedRows[0]?.date ? 'date' : (localMatchedRows[0]?.game_date ? 'game_date' : 'date');
  const inWindowMatches = localMatchedRows.filter((row) => {
    const dateValue = row[localDateKey];
    return dateValue >= dashboardState.window_start && dateValue <= dashboardState.window_end;
  });

  ensure(
    inWindowMatches.length === dashboardState.strategy_matches_window,
    `Strategy matches mismatch: expected ${inWindowMatches.length}, got ${dashboardState.strategy_matches_window}`
  );

  console.log('Dashboard parity checks passed.');
}

main();
