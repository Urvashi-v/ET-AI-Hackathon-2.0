/**
 * API client.
 *
 * Every page talks to the real FastAPI backend through this module and nowhere
 * else. There is no fixture mode and no offline sample data: if the API is down
 * the pages show an error state, because a dashboard that keeps rendering when
 * its backend is gone is lying.
 *
 * The backend serves this frontend from the same origin, so requests are
 * relative and no credentials ever reach the browser.
 */

/** Structured error carrying the backend's own machine-readable code. */
export class ApiError extends Error {
  constructor(message, { status = 0, code = 'unknown', detail = {}, requestId = null } = {}) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.code = code;
    this.detail = detail;
    this.requestId = requestId;
  }

  /** True when the failure is the network or a stopped backend, not a 4xx/5xx. */
  get isOffline() {
    return this.status === 0;
  }
}

async function request(path, { method = 'GET', body, signal, headers = {} } = {}) {
  let response;
  try {
    response = await fetch(path, {
      method,
      signal,
      headers: body instanceof FormData
        ? headers
        : { 'Content-Type': 'application/json', ...headers },
      body: body instanceof FormData ? body : body ? JSON.stringify(body) : undefined,
    });
  } catch (cause) {
    if (cause?.name === 'AbortError') throw cause;
    throw new ApiError(
      'Cannot reach the API. Check that the stack is running (docker compose ps).',
      { status: 0, code: 'network_error' },
    );
  }

  const requestId = response.headers.get('x-request-id');
  const isJson = (response.headers.get('content-type') || '').includes('application/json');
  const payload = isJson ? await response.json().catch(() => null) : await response.text();

  if (!response.ok) {
    const err = payload?.error ?? {};
    throw new ApiError(err.message || `Request failed with status ${response.status}`, {
      status: response.status,
      code: err.code || 'http_error',
      detail: err.detail || {},
      requestId,
    });
  }
  return payload;
}

export const api = {
  health: () => request('/health'),
  ready: () => request('/health/ready'),

  // --- ingestion ---
  ingestPaths: (payload) => request('/api/v1/ingest/paths', { method: 'POST', body: payload }),
  ingestUpload: (formData) => request('/api/v1/ingest', { method: 'POST', body: formData }),
  ingestJob: (jobId) => request(`/api/v1/ingest/${encodeURIComponent(jobId)}`),
  ingestJobs: (limit = 20) => request(`/api/v1/ingest?limit=${limit}`),

  // --- copilot ---
  query: (payload, signal) => request('/api/v1/query', { method: 'POST', body: payload, signal }),
  queryHealth: () => request('/api/v1/query/health'),

  // --- source documents (what makes a citation clickable) ---
  documents: (params = {}) => request(`/api/v1/documents?${new URLSearchParams(params)}`),
  document: (docId) => request(`/api/v1/documents/${encodeURIComponent(docId)}`),
  chunk: (docId, chunkId) =>
    request(
      `/api/v1/documents/${encodeURIComponent(docId)}/chunks/${encodeURIComponent(chunkId)}`,
    ),
  // Page images and raw files are served as binary, so these return URLs for
  // <img src> / <a href> rather than going through the JSON request helper.
  pageImageUrl: (docId, page) =>
    `/api/v1/documents/${encodeURIComponent(docId)}/page/${encodeURIComponent(page)}.png`,
  rawDocumentUrl: (docId) => `/api/v1/documents/${encodeURIComponent(docId)}/raw`,

  // --- assets and graph ---
  assets: (params = {}) => request(`/api/v1/assets?${new URLSearchParams(params)}`),
  assetStats: () => request('/api/v1/assets/stats'),
  asset: (id) => request(`/api/v1/assets/${encodeURIComponent(id)}`),
  graph: (id, params = {}) =>
    request(`/api/v1/graph/${encodeURIComponent(id)}?${new URLSearchParams(params)}`),
  graphSchema: () => request('/api/v1/graph/schema'),
  edgeEvidence: (assetId, edgeId) =>
    request(`/api/v1/graph/${encodeURIComponent(assetId)}/evidence/${encodeURIComponent(edgeId)}`),

  // --- drawings (P&ID) ---
  drawings: () => request('/api/v1/drawings'),
  detections: (docId, params = {}) =>
    request(`/api/v1/drawings/${encodeURIComponent(docId)}/detections?${new URLSearchParams(params)}`),
  locateAsset: (tag) => request(`/api/v1/drawings/locate/${encodeURIComponent(tag)}`),

  // --- intelligence agents ---
  lessons: (payload) => request('/api/v1/lessons', { method: 'POST', body: payload }),
  incidents: (params = {}) => request(`/api/v1/lessons/incidents?${new URLSearchParams(params)}`),
  incident: (id) => request(`/api/v1/lessons/incidents/${encodeURIComponent(id)}`),
  complianceEvaluate: (params = {}) =>
    request(`/api/v1/compliance/evaluate?${new URLSearchParams(params)}`),
  evaluateEvent: (payload) =>
    request('/api/v1/notifications/evaluate', { method: 'POST', body: payload }),

  // --- agents ---
  rca: (payload) => request('/api/v1/rca', { method: 'POST', body: payload }),
  compliance: (payload) => request('/api/v1/compliance', { method: 'POST', body: payload }),
  requirements: (params = {}) =>
    request(`/api/v1/compliance/requirements?${new URLSearchParams(params)}`),

  // --- proactive + feedback ---
  notifications: (params = {}) => request(`/api/v1/notifications?${new URLSearchParams(params)}`),
  acknowledge: (id) =>
    request(`/api/v1/notifications/${encodeURIComponent(id)}/acknowledge`, { method: 'POST' }),
  feedback: (payload) => request('/api/v1/feedback', { method: 'POST', body: payload }),
  events: (limit = 50) => request(`/api/v1/events?limit=${limit}`),
};

/**
 * Subscribe to the server's event stream.
 *
 * Returns a close function. Events arriving here were published by real
 * pipeline stages; nothing is synthesised client-side to animate the panel.
 */
export function subscribeEvents(handlers = {}, { replay = 20 } = {}) {
  const source = new EventSource(`/api/v1/events/stream?replay=${replay}`);
  for (const [name, handler] of Object.entries(handlers)) {
    if (name === 'error') continue;
    source.addEventListener(name, (event) => {
      try {
        handler(JSON.parse(event.data), event);
      } catch {
        /* a malformed frame must not tear down the stream */
      }
    });
  }
  if (handlers.error) source.addEventListener('error', handlers.error);
  return () => source.close();
}

/**
 * POST a question and consume the SSE response.
 *
 * Uses fetch rather than EventSource because EventSource cannot send a body.
 * Each stage event is dispatched as the backend emits it, so the progress the
 * user sees is the pipeline's real progress.
 */
export async function streamQuery(payload, handlers = {}, { signal } = {}) {
  const response = await fetch('/api/v1/query/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
    signal,
  });
  if (!response.ok || !response.body) {
    throw new ApiError(`Stream failed with status ${response.status}`, { status: response.status });
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    let boundary;
    while ((boundary = buffer.indexOf('\n\n')) !== -1) {
      const frame = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);

      let eventName = 'message';
      const dataLines = [];
      for (const line of frame.split('\n')) {
        if (line.startsWith('event:')) eventName = line.slice(6).trim();
        else if (line.startsWith('data:')) dataLines.push(line.slice(5).trim());
      }
      if (!dataLines.length) continue;
      const handler = handlers[eventName];
      if (!handler) continue;
      try {
        handler(JSON.parse(dataLines.join('\n')));
      } catch {
        /* ignore an unparseable frame rather than aborting the stream */
      }
    }
  }
}
