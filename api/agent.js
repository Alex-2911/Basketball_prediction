'use strict';

const DEFAULT_OPENAI_MODEL = 'gpt-4o-mini';
const DEFAULT_MAX_REQUEST_BYTES = 64 * 1024;
const DEFAULT_MAX_SOURCE_BYTES = 200 * 1024;
const DEFAULT_FETCH_TIMEOUT_MS = 8000;
const DEFAULT_MAX_PROMPT_CHARS_PER_SOURCE = 12000;
const REQUIRED_CONTEXT_KEYS = ['dashboard_state_url', 'metrics_url', 'agent_manifest_url'];

class AgentHttpError extends Error {
  constructor(status, message) {
    super(message);
    this.name = 'AgentHttpError';
    this.status = status;
  }
}

const parseCsvList = (value) => (value || '')
  .split(',')
  .map((item) => item.trim())
  .filter(Boolean);

const getAllowedOrigins = () => parseCsvList(
  process.env.HOOPS_AGENT_DATA_ORIGINS
    || process.env.HOOPS_DASHBOARD_ORIGIN
    || process.env.HOOPS_AGENT_ALLOWED_ORIGINS
);

const getCorsOrigins = () => parseCsvList(
  process.env.HOOPS_AGENT_ALLOWED_ORIGINS
    || process.env.HOOPS_DASHBOARD_ORIGIN
);

const toByteLimit = (value, fallback) => {
  const parsed = Number.parseInt(value, 10);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
};

const isHttpUrl = (value) => {
  try {
    const url = new URL(value);
    return url.protocol === 'http:' || url.protocol === 'https:';
  } catch (error) {
    return false;
  }
};

const assertAllowedOrigin = (url, label) => {
  const allowedOrigins = getAllowedOrigins();
  if (!allowedOrigins.length) {
    throw new AgentHttpError(
      500,
      'Backend is missing HOOPS_AGENT_DATA_ORIGINS or HOOPS_DASHBOARD_ORIGIN, so it cannot safely fetch dashboard data.'
    );
  }
  if (!allowedOrigins.includes(url.origin)) {
    throw new AgentHttpError(400, `${label} origin is not allowlisted for this Agent API.`);
  }
};

const normalizeContext = (context) => {
  if (!context || typeof context !== 'object' || Array.isArray(context)) {
    throw new AgentHttpError(400, 'context must be an object.');
  }

  const normalized = {};
  for (const key of REQUIRED_CONTEXT_KEYS) {
    const value = context[key];
    if (typeof value !== 'string' || !isHttpUrl(value)) {
      throw new AgentHttpError(400, `${key} must be an absolute http/https URL.`);
    }
    const url = new URL(value);
    assertAllowedOrigin(url, key);
    normalized[key] = url;
  }

  const origins = new Set(Object.values(normalized).map((url) => url.origin));
  if (origins.size !== 1) {
    throw new AgentHttpError(400, 'context URLs must share the same origin.');
  }

  return normalized;
};

const readRequestBody = async (req) => {
  if (req.body && typeof req.body === 'object') {
    return req.body;
  }

  if (typeof req.body === 'string') {
    try {
      return JSON.parse(req.body);
    } catch (error) {
      throw new AgentHttpError(400, 'Request body must be valid JSON.');
    }
  }

  const maxBytes = toByteLimit(process.env.HOOPS_AGENT_MAX_REQUEST_BYTES, DEFAULT_MAX_REQUEST_BYTES);
  const chunks = [];
  let total = 0;

  for await (const chunk of req) {
    total += chunk.length;
    if (total > maxBytes) {
      throw new AgentHttpError(413, 'Request body is too large.');
    }
    chunks.push(chunk);
  }

  try {
    return JSON.parse(Buffer.concat(chunks).toString('utf8'));
  } catch (error) {
    throw new AgentHttpError(400, 'Request body must be valid JSON.');
  }
};

const validateRequest = (body) => {
  if (!body || typeof body !== 'object' || Array.isArray(body)) {
    throw new AgentHttpError(400, 'Request body must be a JSON object.');
  }
  if (typeof body.question !== 'string' || !body.question.trim()) {
    throw new AgentHttpError(400, 'question must be a non-empty string.');
  }
  if (body.question.length > 4000) {
    throw new AgentHttpError(400, 'question is too long.');
  }
  if (body.capability !== 'read_only') {
    throw new AgentHttpError(403, 'capability must equal read_only.');
  }

  return {
    question: body.question.trim(),
    capability: body.capability,
    context: normalizeContext(body.context),
  };
};

const getContentType = (headers) => {
  if (!headers) return '';
  if (typeof headers.get === 'function') return headers.get('content-type') || '';
  return headers['content-type'] || headers['Content-Type'] || '';
};

const fetchWithLimit = async (url, label) => {
  const maxBytes = toByteLimit(process.env.HOOPS_AGENT_MAX_SOURCE_BYTES, DEFAULT_MAX_SOURCE_BYTES);
  const timeoutMs = toByteLimit(process.env.HOOPS_AGENT_FETCH_TIMEOUT_MS, DEFAULT_FETCH_TIMEOUT_MS);
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetch(url.href, {
      method: 'GET',
      redirect: 'error',
      signal: controller.signal,
      headers: { 'Accept': 'application/json,text/csv,text/plain;q=0.9,*/*;q=0.1' },
    });

    if (!response.ok) {
      throw new AgentHttpError(502, `Unable to fetch ${label}: HTTP ${response.status}.`);
    }

    const contentLength = Number.parseInt(response.headers.get('content-length') || '0', 10);
    if (contentLength > maxBytes) {
      throw new AgentHttpError(413, `${label} exceeds the configured source size limit.`);
    }

    const arrayBuffer = await response.arrayBuffer();
    if (arrayBuffer.byteLength > maxBytes) {
      throw new AgentHttpError(413, `${label} exceeds the configured source size limit.`);
    }

    return {
      label,
      url: url.href,
      contentType: getContentType(response.headers),
      bytes: arrayBuffer.byteLength,
      text: Buffer.from(arrayBuffer).toString('utf8'),
    };
  } finally {
    clearTimeout(timeout);
  }
};

const parseJsonSource = (source) => {
  try {
    return JSON.parse(source.text);
  } catch (error) {
    throw new AgentHttpError(502, `${source.label} was not valid JSON.`);
  }
};

const sourcePathIsSafe = (sourcePath) => {
  if (typeof sourcePath !== 'string' || !sourcePath.trim()) return false;
  if (/^[a-z][a-z0-9+.-]*:/i.test(sourcePath)) return false;
  if (sourcePath.startsWith('//')) return false;
  if (sourcePath.includes('..')) return false;
  if (sourcePath.startsWith('/')) return false;
  return true;
};

const resolveManifestSourceUrl = (manifestUrl, sourcePath) => {
  if (!sourcePathIsSafe(sourcePath)) {
    throw new AgentHttpError(400, `Manifest source path is not allowed: ${sourcePath}`);
  }

  const baseDir = new URL('.', manifestUrl.href);
  const resolved = new URL(sourcePath, baseDir.href);
  assertAllowedOrigin(resolved, `manifest source ${sourcePath}`);

  if (!resolved.href.startsWith(baseDir.href)) {
    throw new AgentHttpError(400, `Manifest source must stay under ${baseDir.href}.`);
  }

  return resolved;
};

const fetchDashboardSources = async (validated) => {
  const used = [];
  const warnings = [];
  const explicitSources = [
    ['dashboard_state.json', validated.context.dashboard_state_url],
    ['metrics_snapshot.json', validated.context.metrics_url],
    ['agent_manifest.json', validated.context.agent_manifest_url],
  ];

  const fetchedByUrl = new Map();
  for (const [label, url] of explicitSources) {
    const source = await fetchWithLimit(url, label);
    fetchedByUrl.set(url.href, source);
    used.push(label);
  }

  const manifest = parseJsonSource(fetchedByUrl.get(validated.context.agent_manifest_url.href));
  const readOnlySources = Array.isArray(manifest.read_only_sources) ? manifest.read_only_sources : [];

  for (const entry of readOnlySources) {
    if (!entry || typeof entry !== 'object') continue;
    const label = typeof entry.label === 'string' && entry.label.trim()
      ? entry.label.trim()
      : entry.path;

    try {
      const sourceUrl = resolveManifestSourceUrl(validated.context.agent_manifest_url, entry.path);
      if (!fetchedByUrl.has(sourceUrl.href)) {
        const source = await fetchWithLimit(sourceUrl, label);
        fetchedByUrl.set(sourceUrl.href, source);
        used.push(label);
      }
    } catch (error) {
      if (error instanceof AgentHttpError && [400, 413, 502].includes(error.status)) {
        warnings.push(`${label || entry.path}: ${error.message}`);
        continue;
      }
      throw error;
    }
  }

  return { sources: Array.from(fetchedByUrl.values()), used, warnings, manifest };
};

const truncate = (text, maxChars) => {
  if (text.length <= maxChars) return text;
  return `${text.slice(0, maxChars)}\n...[truncated ${text.length - maxChars} chars]`;
};

const classifySource = (source) => {
  const name = source.url.toLowerCase();
  if (name.endsWith('.json') || source.contentType.includes('json')) {
    try {
      return JSON.stringify(JSON.parse(source.text), null, 2);
    } catch (error) {
      return source.text;
    }
  }

  if (name.endsWith('.csv') || source.contentType.includes('csv')) {
    const lines = source.text.split(/\r?\n/).filter(Boolean);
    return lines.slice(0, 80).join('\n');
  }

  return source.text;
};

const buildAgentPrompt = ({ question, fetched }) => {
  const maxChars = toByteLimit(
    process.env.HOOPS_AGENT_MAX_PROMPT_CHARS_PER_SOURCE,
    DEFAULT_MAX_PROMPT_CHARS_PER_SOURCE
  );

  const sourceBlocks = fetched.sources.map((source) => {
    const body = truncate(classifySource(source), maxChars);
    return `### ${source.label}\nURL: ${source.url}\nBytes: ${source.bytes}\n\n${body}`;
  }).join('\n\n---\n\n');

  return [
    {
      role: 'system',
      content: [
        'You are the Hoops Insight read-only betting analysis agent.',
        'Use only the supplied dashboard data sources. If a detail is missing, say it is missing.',
        'Separate canonical model signals, setup-profitability candidates, near-miss/vibe/watchlist candidates, and no-bet cases.',
        'Use Stage 1 daily snapshot data as the canonical source for canonical_signal.',
        'Use setup_profitability_scan only as historical/setup support, not proof of a canonical bet.',
        'Use script11_watchlist_history only for watchlist, vibe, and near-miss context.',
        'Never place bets, never imply access to betting accounts, and never suggest that you executed an action.',
        'Never claim a real bet was placed unless actual_bets_manual.csv or another supplied real placed bets ledger explicitly says so.',
        'Return concise markdown with clear headings and caveats.'
      ].join(' '),
    },
    {
      role: 'user',
      content: `Question: ${question}\n\nDashboard sources:\n\n${sourceBlocks}`,
    },
  ];
};

const extractOpenAIText = (payload) => {
  if (typeof payload.output_text === 'string' && payload.output_text.trim()) {
    return payload.output_text;
  }

  const chunks = [];
  for (const item of payload.output || []) {
    for (const content of item.content || []) {
      if (typeof content.text === 'string') chunks.push(content.text);
    }
  }
  return chunks.join('\n').trim();
};

const callOpenAI = async (messages) => {
  const apiKey = process.env.OPENAI_API_KEY;
  if (!apiKey) {
    throw new AgentHttpError(500, 'OPENAI_API_KEY is not configured on the backend.');
  }

  const model = process.env.OPENAI_MODEL || DEFAULT_OPENAI_MODEL;
  const response = await fetch('https://api.openai.com/v1/responses', {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${apiKey}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      model,
      input: messages,
      temperature: 0.2,
      max_output_tokens: 900,
    }),
  });

  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const message = payload.error?.message || `OpenAI request failed with HTTP ${response.status}.`;
    throw new AgentHttpError(502, message);
  }

  const answer = extractOpenAIText(payload);
  if (!answer) {
    throw new AgentHttpError(502, 'OpenAI response did not include answer text.');
  }
  return answer;
};

const applyCors = (req, res) => {
  const requestOrigin = req.headers?.origin;
  const corsOrigins = getCorsOrigins();
  const allowAny = corsOrigins.includes('*');

  if (requestOrigin && (allowAny || corsOrigins.includes(requestOrigin))) {
    res.setHeader('Access-Control-Allow-Origin', allowAny ? '*' : requestOrigin);
    res.setHeader('Vary', 'Origin');
  }

  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
  res.setHeader('Access-Control-Max-Age', '86400');
};

const sendJson = (res, status, payload) => {
  res.statusCode = status;
  res.setHeader('Content-Type', 'application/json; charset=utf-8');
  res.end(JSON.stringify(payload));
};

const handler = async (req, res) => {
  applyCors(req, res);

  if (req.method === 'OPTIONS') {
    res.statusCode = 204;
    res.end();
    return;
  }

  if (req.method !== 'POST') {
    sendJson(res, 405, { error: 'Method not allowed. Use POST.' });
    return;
  }

  try {
    const body = await readRequestBody(req);
    const validated = validateRequest(body);
    const fetched = await fetchDashboardSources(validated);
    const messages = buildAgentPrompt({ question: validated.question, fetched });
    const answer = await callOpenAI(messages);

    sendJson(res, 200, {
      answer,
      used_sources: fetched.used,
      warnings: fetched.warnings,
    });
  } catch (error) {
    const status = error instanceof AgentHttpError ? error.status : 500;
    sendJson(res, status, {
      error: error.message || 'Unexpected Agent API error.',
      warnings: [],
    });
  }
};

module.exports = handler;
module.exports._internals = {
  AgentHttpError,
  buildAgentPrompt,
  fetchDashboardSources,
  normalizeContext,
  resolveManifestSourceUrl,
  sourcePathIsSafe,
  validateRequest,
};
