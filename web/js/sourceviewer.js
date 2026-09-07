/**
 * The source viewer: what a citation opens.
 *
 * A citation the reader cannot check is a claim about evidence, not evidence.
 * This panel closes that gap — click [C1] and the actual extracted span appears,
 * with the rendered source page beside it when the document is a PDF, the
 * cited sentence highlighted in the text, and the document's revision standing
 * stated plainly.
 *
 * That last part matters more than it looks. A procedure that has been
 * superseded is the single most dangerous thing this system could show without
 * comment, so currency is displayed at the top of the panel, not buried in
 * metadata.
 *
 * Everything here is fetched from the API at open time. Nothing about a source
 * is cached client-side and re-displayed later, because a stale copy of a
 * superseded document is exactly the failure the panel exists to prevent.
 */

import { api } from './api.js';
import { docTypeChip, esc, fmtDate, provBadge, renderError } from './ui.js';

let panel = null;
let lastFocused = null;

/** Build the panel lazily; most sessions never open it. */
function ensurePanel() {
  if (panel) return panel;
  panel = document.createElement('aside');
  panel.className = 'source-viewer';
  panel.hidden = true;
  panel.setAttribute('role', 'dialog');
  panel.setAttribute('aria-modal', 'true');
  panel.setAttribute('aria-label', 'Source document');
  panel.innerHTML = `
    <div class="source-viewer-head">
      <div class="stack tight" style="gap:2px">
        <strong class="source-viewer-title">Source</strong>
        <span class="small muted source-viewer-sub"></span>
      </div>
      <button type="button" class="btn ghost source-viewer-close" aria-label="Close source">Close</button>
    </div>
    <div class="source-viewer-body"></div>`;
  document.body.appendChild(panel);

  panel.querySelector('.source-viewer-close').addEventListener('click', close);
  // Escape closes, and focus returns where it came from. A panel that traps a
  // keyboard user is worse than no panel.
  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && !panel.hidden) close();
  });
  return panel;
}

export function close() {
  if (!panel || panel.hidden) return;
  panel.hidden = true;
  document.body.classList.remove('source-open');
  if (lastFocused && document.contains(lastFocused)) lastFocused.focus();
  lastFocused = null;
}

/**
 * Open the viewer on one cited chunk.
 *
 * `highlight` is the exact sentence the answer used, when the answer was
 * composed extractively. It is located in the chunk text by string match rather
 * than by stored offset: the offsets are correct, but a mismatch would silently
 * highlight the wrong words, and a failed match visibly highlights nothing.
 */
export async function openCitation({ docId, chunkId, page, highlight }) {
  const node = ensurePanel();
  lastFocused = document.activeElement;
  node.hidden = false;
  document.body.classList.add('source-open');
  node.querySelector('.source-viewer-close').focus();

  const body = node.querySelector('.source-viewer-body');
  body.innerHTML = '<p class="muted">Loading source…</p>';

  try {
    const [chunkPayload, docPayload] = await Promise.all([
      api.chunk(docId, chunkId),
      api.document(docId),
    ]);
    render(node, chunkPayload, docPayload, { page, highlight });
  } catch (error) {
    renderError(body, error, { title: 'Could not open the source document' });
  }
}

function render(node, chunkPayload, docPayload, { page, highlight }) {
  const chunk = chunkPayload.chunk;
  const doc = docPayload.document;
  const chain = docPayload.revision_chain || {};

  node.querySelector('.source-viewer-title').textContent = doc.title;
  node.querySelector('.source-viewer-sub').innerHTML = [
    docTypeChip(doc.doc_type),
    provBadge(doc.data_class),
    doc.doc_number ? `<span class="mono small">${esc(doc.doc_number)}</span>` : '',
    doc.revision ? `<span class="small muted">rev ${esc(doc.revision)}</span>` : '',
  ].filter(Boolean).join(' ');

  const pageNumber = page ?? chunk.page_from;
  const isPdf = (doc.mime_type || '') === 'application/pdf' && pageNumber;

  node.querySelector('.source-viewer-body').innerHTML = `
    ${currencyNotice(doc, chain)}
    <div class="source-grid">
      <section class="stack">
        <h4 class="section-label">Extracted text${
          pageNumber ? ` · page ${esc(pageNumber)}` : ''
        }</h4>
        ${chunk.section_path ? `<p class="small muted mono">${esc(chunk.section_path)}</p>` : ''}
        <div class="source-text">${withHighlight(chunk.text, highlight)}</div>
        <p class="small muted">
          Extracted by <code>${esc(chunk.extraction_method || 'unknown')}</code> ·
          ${esc(chunk.char_count)} characters ·
          chunk <span class="mono">${esc(chunk.chunk_id)}</span>
        </p>
        ${mentionList(chunkPayload.mentions)}
      </section>
      <section class="stack">
        ${
          isPdf
            ? `<h4 class="section-label">Source page</h4>
               <img class="source-page" alt="Page ${esc(pageNumber)} of ${esc(doc.title)}"
                    src="${api.pageImageUrl(doc.doc_id, pageNumber)}"
                    loading="lazy">`
            : `<h4 class="section-label">Source file</h4>
               <p class="small muted">
                 This document is <code>${esc(doc.mime_type || 'of unknown type')}</code>,
                 so there is no page image to render. The extracted text on the left is
                 what retrieval actually searched.
               </p>`
        }
        <a class="btn ghost" href="${api.rawDocumentUrl(doc.doc_id)}" target="_blank" rel="noopener">
          Open the original file
        </a>
      </section>
    </div>`;
}

/**
 * Currency, stated before content.
 *
 * Three cases, and the distinction is deliberate: superseded is a warning,
 * multiple formats of one revision is merely information, and an unresolved
 * conflict is an admission that the system does not know.
 */
function currencyNotice(doc, chain) {
  if (doc.revision_conflict) {
    return `<div class="notice danger">
      <strong>Revision standing unresolved.</strong>
      ${esc(doc.revision_note || '')}
      This document may or may not be the current one — check before acting on it.
    </div>`;
  }
  if (!doc.is_current) {
    const replacement = chain.superseded_by;
    return `<div class="notice warn">
      <strong>Superseded — do not work to this document.</strong>
      ${replacement
        ? `Replaced by <a href="#" data-open-doc="${esc(replacement.doc_id)}">${esc(
            replacement.title,
          )}</a>${replacement.revision ? ` (rev ${esc(replacement.revision)})` : ''}.`
        : ''}
      ${doc.valid_to ? ` Valid until ${esc(fmtDate(doc.valid_to))}.` : ''}
    </div>`;
  }
  if (doc.revision_note) {
    return `<div class="notice info">${esc(doc.revision_note)}</div>`;
  }
  return '';
}

/** Highlight the cited sentence inside the full chunk, if it can be located. */
function withHighlight(text, highlight) {
  const safe = esc(text || '');
  if (!highlight) return safe;
  const needle = esc(highlight.trim());
  if (!needle || !safe.includes(needle)) {
    // Whitespace differs between the composed claim and the stored chunk (the
    // composer rejoins soft-wrapped lines). Fall back to a whitespace-tolerant
    // match rather than showing nothing.
    const pattern = needle.split(/\s+/).map(escapeRegex).join('\\s+');
    try {
      return safe.replace(new RegExp(pattern), (m) => `<mark>${m}</mark>`);
    } catch {
      return safe;
    }
  }
  return safe.replaceAll(needle, `<mark>${needle}</mark>`);
}

function escapeRegex(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function mentionList(mentions) {
  if (!mentions?.length) return '';
  const rows = mentions
    .map(
      (m) => `<li>
        <span class="mono">${esc(m.surface_form)}</span>
        ${m.canonical_tag ? `→ <span class="mono">${esc(m.canonical_tag)}</span>` : '<span class="muted small">unresolved</span>'}
        <span class="small muted">${esc(m.tag_kind || '')} · ${Number(m.extractor_confidence ?? 0).toFixed(2)}</span>
        ${m.needs_review ? '<span class="badge warn">review</span>' : ''}
      </li>`,
    )
    .join('');
  return `<details class="stack tight">
    <summary class="small">Entities extracted from this span (${mentions.length})</summary>
    <ul class="mention-list">${rows}</ul>
  </details>`;
}

/**
 * Delegate clicks on anything carrying citation data attributes.
 *
 * Delegation rather than per-element listeners because the answer and evidence
 * panels are re-rendered on every query; re-binding each time leaks handlers.
 */
export function wireCitationLinks(root = document) {
  root.addEventListener('click', (event) => {
    const trigger = event.target.closest('[data-cite-doc]');
    if (trigger) {
      event.preventDefault();
      openCitation({
        docId: trigger.dataset.citeDoc,
        chunkId: trigger.dataset.citeChunk,
        page: trigger.dataset.citePage || null,
        highlight: trigger.dataset.citeText || null,
      });
      return;
    }
    const docLink = event.target.closest('[data-open-doc]');
    if (docLink) {
      event.preventDefault();
      window.location.href = `/ui/ingestion.html?doc=${encodeURIComponent(docLink.dataset.openDoc)}`;
    }
  });
}
