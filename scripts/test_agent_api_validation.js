'use strict';

const assert = require('assert');
const agent = require('../api/agent.js');
const {
  buildAgentPrompt,
  fetchDashboardSources,
  resolveManifestSourceUrl,
  sourcePathIsSafe,
  validateRequest,
} = agent._internals;

const originalEnv = { ...process.env };
const originalFetch = global.fetch;

const resetEnv = () => {
  process.env = { ...originalEnv };
  process.env.HOOPS_AGENT_DATA_ORIGINS = 'https://dash.example.com';
};

const expectAgentError = (fn, status) => {
  assert.throws(fn, (error) => error && error.status === status);
};

(async () => {
  resetEnv();

  const validRequest = {
    question: 'Explain today\'s board.',
    capability: 'read_only',
    context: {
      dashboard_state_url: 'https://dash.example.com/public/data/dashboard_state.json',
      metrics_url: 'https://dash.example.com/public/data/metrics_snapshot.json',
      agent_manifest_url: 'https://dash.example.com/public/data/agent_manifest.json',
    },
  };

  const validated = validateRequest(validRequest);
  assert.strictEqual(validated.question, validRequest.question);
  assert.strictEqual(validated.context.agent_manifest_url.origin, 'https://dash.example.com');

  expectAgentError(() => validateRequest({ ...validRequest, capability: 'write' }), 403);
  expectAgentError(() => validateRequest({ ...validRequest, question: '' }), 400);
  expectAgentError(() => validateRequest({
    ...validRequest,
    context: { ...validRequest.context, metrics_url: '/public/data/metrics_snapshot.json' },
  }), 400);
  expectAgentError(() => validateRequest({
    ...validRequest,
    context: { ...validRequest.context, metrics_url: 'https://evil.example.com/public/data/metrics_snapshot.json' },
  }), 400);

  assert.strictEqual(sourcePathIsSafe('stage1_daily_snapshot_latest.json'), true);
  assert.strictEqual(sourcePathIsSafe('nested/source.csv'), true);
  assert.strictEqual(sourcePathIsSafe('../secret.txt'), false);
  assert.strictEqual(sourcePathIsSafe('https://evil.example.com/source.csv'), false);
  assert.strictEqual(sourcePathIsSafe('/etc/passwd'), false);

  const manifestUrl = new URL('https://dash.example.com/public/data/agent_manifest.json');
  assert.strictEqual(
    resolveManifestSourceUrl(manifestUrl, 'stage1_daily_snapshot_latest.json').href,
    'https://dash.example.com/public/data/stage1_daily_snapshot_latest.json'
  );
  expectAgentError(() => resolveManifestSourceUrl(manifestUrl, '../private.csv'), 400);

  const fixtures = new Map([
    ['https://dash.example.com/public/data/dashboard_state.json', JSON.stringify({ slate_date: '2026-05-08' })],
    ['https://dash.example.com/public/data/metrics_snapshot.json', JSON.stringify({ roi: 0.12 })],
    ['https://dash.example.com/public/data/agent_manifest.json', JSON.stringify({
      read_only_sources: [
        { label: 'Stage 1 daily summary', path: 'stage1_daily_snapshot_latest.json' },
        { label: 'Unsafe path', path: '../secret.csv' },
      ],
    })],
    ['https://dash.example.com/public/data/stage1_daily_snapshot_latest.json', JSON.stringify({ canonical_signal: [] })],
  ]);

  global.fetch = async (url) => {
    assert.ok(fixtures.has(url), `unexpected fetch ${url}`);
    const body = fixtures.get(url);
    return {
      ok: true,
      status: 200,
      headers: {
        get(name) {
          if (name.toLowerCase() === 'content-length') return String(Buffer.byteLength(body));
          if (name.toLowerCase() === 'content-type') return url.endsWith('.json') ? 'application/json' : 'text/plain';
          return '';
        },
      },
      async arrayBuffer() {
        return Buffer.from(body, 'utf8');
      },
    };
  };

  const fetched = await fetchDashboardSources(validated);
  assert.deepStrictEqual(fetched.used, [
    'dashboard_state.json',
    'metrics_snapshot.json',
    'agent_manifest.json',
    'Stage 1 daily summary',
  ]);
  assert.strictEqual(fetched.warnings.length, 1);
  assert.match(fetched.warnings[0], /Unsafe path/);

  const prompt = buildAgentPrompt({ question: validRequest.question, fetched });
  assert.strictEqual(prompt.length, 2);
  assert.match(prompt[0].content, /Stage 1 daily snapshot data as the canonical source/);
  assert.match(prompt[0].content, /Never place bets/);
  assert.match(prompt[1].content, /Explain today's board/);
  assert.match(prompt[1].content, /Stage 1 daily summary/);

  global.fetch = originalFetch;
  process.env = originalEnv;
  console.log('Agent API validation tests passed.');
})().catch((error) => {
  global.fetch = originalFetch;
  process.env = originalEnv;
  console.error(error);
  process.exit(1);
});
