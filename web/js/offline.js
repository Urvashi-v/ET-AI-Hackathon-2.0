/**
 * Offline cache for the field view.
 *
 * What this is: a record of API responses this browser has *actually received*,
 * kept so a technician who loses signal in a pipe rack still has the asset
 * context they loaded at the gate.
 *
 * What this is emphatically not: offline operation of the system. Retrieval,
 * the graph, RCA and compliance all run server-side. Nothing is computed here,
 * nothing is synthesised, and an asset never opened online is not available
 * offline — the UI says so rather than showing an empty page that looks like a
 * loading failure.
 *
 * The honesty rules this enforces:
 *
 *  - **only real responses are cached.** Every entry was returned by the backend
 *    to this browser, with the timestamp of when.
 *  - **cached content is labelled and dated.** A reading taken from cache is
 *    shown as cached, with its age, because a four-hour-old work order list is
 *    a different thing from a current one and the difference can matter.
 *  - **the scope is enumerable.** `describe()` returns exactly what is held, so
 *    the UI can state it rather than implying the whole system is available.
 *
 * localStorage rather than a service worker: a service worker intercepting
 * fetches would make cached and live responses indistinguishable to the calling
 * code, which is precisely the property that must not exist here.
 */

const PREFIX = 'brain.cache.v1:';
const INDEX_KEY = 'brain.cache.v1.index';

/** Entries older than this are dropped on read. A stale work-order list is worse
 *  than none, because it looks current. Twenty-four hours covers a shift and a
 *  handover without covering a week. */
const MAX_AGE_MS = 24 * 60 * 60 * 1000;

/** localStorage is small (~5 MB) and shared with everything else on the origin.
 *  Oldest entries are evicted first when the budget is exceeded. */
const MAX_ENTRIES = 60;

function readIndex() {
  try {
    return JSON.parse(localStorage.getItem(INDEX_KEY) || '{}');
  } catch {
    return {};
  }
}

function writeIndex(index) {
  try {
    localStorage.setItem(INDEX_KEY, JSON.stringify(index));
  } catch {
    /* quota or private mode: the cache is an optimisation, never a requirement */
  }
}

/**
 * Store a response that actually came back from the backend.
 *
 * @param {string} key logical key, e.g. "asset:P-101B"
 * @param {unknown} payload the response body, exactly as received
 * @param {{label?: string}} meta what this is, for the scope listing
 */
export function put(key, payload, { label = key } = {}) {
  const index = readIndex();
  const entry = { key, label, cached_at: new Date().toISOString() };
  try {
    localStorage.setItem(PREFIX + key, JSON.stringify(payload));
  } catch {
    // Full or unavailable. Evict the oldest and try once; if it still fails the
    // app carries on without a cache rather than failing the request.
    const oldest = Object.values(index).sort((a, b) => a.cached_at.localeCompare(b.cached_at))[0];
    if (oldest) drop(oldest.key);
    try {
      localStorage.setItem(PREFIX + key, JSON.stringify(payload));
    } catch {
      return;
    }
  }
  index[key] = entry;

  const entries = Object.values(index).sort((a, b) => b.cached_at.localeCompare(a.cached_at));
  for (const stale of entries.slice(MAX_ENTRIES)) {
    delete index[stale.key];
    try { localStorage.removeItem(PREFIX + stale.key); } catch { /* ignore */ }
  }
  writeIndex(index);
}

/**
 * Read a cached response, or null.
 *
 * Returns the payload alongside its age so the caller can label it. A cache that
 * hands back data indistinguishable from live data is the thing this module
 * exists to avoid.
 */
export function get(key) {
  const index = readIndex();
  const entry = index[key];
  if (!entry) return null;

  const age = Date.now() - Date.parse(entry.cached_at);
  if (age > MAX_AGE_MS) {
    drop(key);
    return null;
  }
  try {
    const raw = localStorage.getItem(PREFIX + key);
    if (raw === null) return null;
    return { payload: JSON.parse(raw), cachedAt: entry.cached_at, ageMs: age };
  } catch {
    return null;
  }
}

export function drop(key) {
  const index = readIndex();
  delete index[key];
  writeIndex(index);
  try { localStorage.removeItem(PREFIX + key); } catch { /* ignore */ }
}

export function clear() {
  for (const key of Object.keys(readIndex())) {
    try { localStorage.removeItem(PREFIX + key); } catch { /* ignore */ }
  }
  writeIndex({});
}

/**
 * Exactly what is held offline, for display.
 *
 * The UI shows this verbatim. Stating "3 assets, last synchronised 11:42" is
 * honest; implying the plant fits in a phone is not.
 */
export function describe() {
  const entries = Object.values(readIndex())
    .filter((e) => Date.now() - Date.parse(e.cached_at) <= MAX_AGE_MS)
    .sort((a, b) => b.cached_at.localeCompare(a.cached_at));
  return {
    count: entries.length,
    lastSync: entries.length ? entries[0].cached_at : null,
    items: entries,
    maxAgeHours: MAX_AGE_MS / 3600000,
  };
}

/**
 * Fetch through the cache.
 *
 * Online: call the backend, cache what comes back, return it marked `fresh`.
 * Offline or failed: fall back to cache and return it marked `cached` with its
 * age. Never invents a response — if there is nothing cached, the error
 * propagates and the caller reports it.
 */
export async function through(key, loader, { label } = {}) {
  if (navigator.onLine) {
    try {
      const payload = await loader();
      put(key, payload, { label });
      return { payload, source: 'fresh', cachedAt: new Date().toISOString(), ageMs: 0 };
    } catch (error) {
      const fallback = get(key);
      if (!fallback) throw error;
      return { ...fallback, source: 'cached', error };
    }
  }
  const fallback = get(key);
  if (!fallback) {
    throw new Error(
      'Offline, and this has not been loaded on this device before. ' +
      'Only content previously retrieved online is available offline.',
    );
  }
  return { ...fallback, source: 'cached' };
}

/** Human-readable age, for the cached-data label. */
export function ageLabel(ms) {
  const minutes = Math.round(ms / 60000);
  if (minutes < 1) return 'just now';
  if (minutes < 60) return `${minutes} min ago`;
  const hours = Math.round(minutes / 60);
  return `${hours} h ago`;
}

/** Notify on connectivity change. Returns an unsubscribe function. */
export function watchConnectivity(handler) {
  const online = () => handler(true);
  const offline = () => handler(false);
  window.addEventListener('online', online);
  window.addEventListener('offline', offline);
  handler(navigator.onLine);
  return () => {
    window.removeEventListener('online', online);
    window.removeEventListener('offline', offline);
  };
}
