/**
 * Interactive P&ID viewer.
 *
 * A rendered page with detection boxes drawn over it, scaled from PDF points to
 * whatever size the image happens to be on screen. The overlay is SVG on top of
 * an <img> rather than burned into the raster, for three reasons: the same
 * render serves every highlight, changing the selection costs no round trip, and
 * a box stays a box when the reader zooms rather than becoming a blurry
 * rectangle.
 *
 * Everything drawn comes from `/api/v1/drawings/{doc}/detections`. There is no
 * client-side detection and no decorative geometry — if a box is on the screen,
 * a detector put a row in Postgres for it, and clicking it shows which detector
 * and how confident.
 *
 * The honest bit: line segments are hidden by default. There are hundreds on a
 * sheet, they are the least reliable detector, and drawing them all over the
 * drawing they were detected from obscures the tags that matter. They are one
 * toggle away, and the toggle says how many there are.
 */

import { api } from './api.js';
import { esc } from './ui.js';

/** Colour per detection kind. Deliberately distinct hues, not a gradient —
 *  these are categories, not magnitudes. */
const KIND_STYLE = {
  tag: { stroke: '#35c8d8', fill: 'rgba(53,200,216,.14)', label: 'Tag' },
  instrument_bubble: { stroke: '#e0a53a', fill: 'rgba(224,165,58,.10)', label: 'Instrument' },
  line_segment: { stroke: '#5b8def', fill: 'none', label: 'Line' },
  equipment_symbol: { stroke: '#3fb96a', fill: 'rgba(63,185,106,.12)', label: 'Symbol' },
};

export class DrawingViewer {
  /**
   * @param {HTMLElement} root container the viewer takes over
   * @param {{onSelect?: (detection) => void}} options
   */
  constructor(root, options = {}) {
    this.root = root;
    this.options = options;
    this.data = null;
    this.docId = null;
    this.page = 1;
    this.highlight = null;
    // Lines off by default: hundreds of them, least reliable detector, and they
    // bury the tags when drawn.
    this.visible = new Set(['tag', 'instrument_bubble', 'equipment_symbol']);
    this.root.classList.add('drawing-viewer');
  }

  async load(docId, page = 1, { highlight = null } = {}) {
    this.docId = docId;
    this.page = page;
    this.highlight = highlight ? String(highlight).toUpperCase() : null;
    this.root.innerHTML = '<p class="muted" style="padding:14px">Loading drawing…</p>';
    try {
      this.data = await api.detections(docId, { page });
      this.render();
    } catch (error) {
      this.root.innerHTML = `<div class="notice danger" style="margin:14px">
        <strong>Could not load the drawing.</strong> ${esc(error.message || String(error))}
      </div>`;
    }
  }

  /** Highlight one asset without re-fetching. */
  focus(assetTag) {
    this.highlight = assetTag ? String(assetTag).toUpperCase() : null;
    if (this.data) this.render();
  }

  toggleKind(kind) {
    if (this.visible.has(kind)) this.visible.delete(kind);
    else this.visible.add(kind);
    this.render();
  }

  render() {
    const d = this.data;
    if (!d) return;
    const geometry = d.page_geometry;
    if (!geometry) {
      this.root.innerHTML = `<div class="notice warn" style="margin:14px">
        <strong>No detections stored for this page.</strong>
        The drawing is available, but nothing has been digitised from it yet.
      </div>`;
      return;
    }

    const shown = d.detections.filter((x) => this.visible.has(x.kind));
    const counts = d.counts || {};

    this.root.innerHTML = `
      <div class="drawing-toolbar">
        <div class="row tight">
          ${Object.entries(KIND_STYLE)
            .filter(([kind]) => counts[kind])
            .map(([kind, style]) => `
              <button type="button" class="small ${this.visible.has(kind) ? '' : 'ghost'}"
                      data-toggle-kind="${kind}">
                <span class="swatch" style="background:${style.stroke}"></span>
                ${esc(style.label)} (${counts[kind]})
              </button>`).join('')}
        </div>
        <span class="small dim">${esc(d.document.title)} · page ${d.page}</span>
      </div>

      <div class="drawing-stage">
        <img class="drawing-page" src="${esc(d.page_image)}"
             alt="Page ${d.page} of ${esc(d.document.title)}">
        <svg class="drawing-overlay" viewBox="0 0 ${geometry.width} ${geometry.height}"
             preserveAspectRatio="none" aria-hidden="true">
          ${shown.map((x) => this.box(x)).join('')}
        </svg>
      </div>

      ${d.unlinked_tags.length ? `<div class="notice warn" style="margin:12px">
        <strong>${d.unlinked_tags.length} tag(s) on this sheet match no asset in the corpus:</strong>
        <span class="mono">${d.unlinked_tags.map(esc).join(', ')}</span>.
        That is the gap between what the plant has drawn and what it has recorded.
      </div>` : ''}

      <div class="drawing-legend small dim">
        ${d.linked} of ${counts.tag || 0} tag(s) resolved to a canonical asset ·
        ${(d.connections || []).length} connection(s) recovered ·
        ${Object.entries(d.detectors || {})
          .filter(([, v]) => v.state !== 'available')
          .map(([k, v]) => `<span class="badge neutral no-dot">${esc(k)}: ${esc(v.state)}</span>`)
          .join(' ')}
      </div>
      <div data-detection-detail></div>`;

    this.root.querySelectorAll('[data-toggle-kind]').forEach((button) => {
      button.addEventListener('click', () => this.toggleKind(button.dataset.toggleKind));
    });
    this.root.querySelectorAll('[data-detection]').forEach((node) => {
      node.addEventListener('click', () => {
        const detection = d.detections.find(
          (x) => String(x.detection_id) === node.dataset.detection,
        );
        if (detection) this.showDetail(detection);
      });
    });

    // Scroll the highlighted box into view; on a large sheet it is otherwise
    // off-screen and the highlight is invisible.
    const marked = this.root.querySelector('.det.is-highlight');
    if (marked) marked.closest('.drawing-stage')?.scrollIntoView({ block: 'nearest' });
  }

  box(detection) {
    const style = KIND_STYLE[detection.kind] || KIND_STYLE.tag;
    const width = Math.max(detection.x1 - detection.x0, 1);
    const height = Math.max(detection.y1 - detection.y0, 1);
    const tag = (detection.canonical_tag || detection.normalised || '').toUpperCase();
    const isHighlight = this.highlight && tag === this.highlight;
    // Line segments are drawn thinner and are not clickable: there are hundreds,
    // and a click target over each one would swallow every click on the sheet.
    const interactive = detection.kind !== 'line_segment';
    return `<rect class="det ${isHighlight ? 'is-highlight' : ''}"
      ${interactive ? `data-detection="${detection.detection_id}"` : ''}
      x="${detection.x0}" y="${detection.y0}" width="${width}" height="${height}"
      stroke="${isHighlight ? '#ff8a3d' : style.stroke}"
      stroke-width="${isHighlight ? 3 : detection.kind === 'line_segment' ? 0.6 : 1.4}"
      fill="${isHighlight ? 'rgba(255,138,61,.20)' : style.fill}"
      vector-effect="non-scaling-stroke">
      ${interactive ? `<title>${esc(detection.text || detection.kind)}${
        detection.canonical_tag ? ` → ${esc(detection.canonical_tag)}` : ''
      }</title>` : ''}
    </rect>`;
  }

  /**
   * What one detection actually is.
   *
   * Shows the detector and its parameters, not just a confidence number: "Hough
   * circle, radius 21 px against a sheet median of 20" is a claim the reader can
   * evaluate, and "0.95 confident" is not.
   */
  showDetail(detection) {
    const node = this.root.querySelector('[data-detection-detail]');
    const properties = detection.properties || {};
    node.innerHTML = `
      <div class="detection-detail">
        <div class="row" style="justify-content:space-between">
          <span class="row tight">
            <span class="badge ${detection.linked_asset_id ? 'ok' : 'neutral'} no-dot">
              ${esc(KIND_STYLE[detection.kind]?.label || detection.kind)}
            </span>
            <strong class="mono">${esc(detection.text || '(no text)')}</strong>
            ${detection.canonical_tag
              ? `<span class="badge ok no-dot mono">→ ${esc(detection.canonical_tag)}</span>`
              : '<span class="badge warn no-dot">no matching asset</span>'}
          </span>
          <button type="button" class="small ghost" data-close-detail>Close</button>
        </div>
        <dl class="kv small" style="margin-top:8px">
          <dt>Detector</dt><dd class="mono">${esc(detection.method)}</dd>
          <dt>Quality</dt><dd class="mono tabular">${Number(detection.confidence).toFixed(3)}
            <span class="dim">— detector-specific, not a probability</span></dd>
          <dt>Position</dt><dd class="mono tabular">
            ${detection.x0.toFixed(0)}, ${detection.y0.toFixed(0)} →
            ${detection.x1.toFixed(0)}, ${detection.y1.toFixed(0)} pt</dd>
          ${Object.entries(properties).map(([key, value]) => `
            <dt>${esc(key.replaceAll('_', ' '))}</dt>
            <dd class="mono tabular">${esc(String(value))}</dd>`).join('')}
        </dl>
        ${detection.canonical_tag ? `<div class="row tight" style="margin-top:8px">
          <a class="btn ghost small" href="/ui/reliability.html?asset=${encodeURIComponent(detection.canonical_tag)}">Reliability</a>
          <a class="btn ghost small" href="/ui/graph.html?asset=${encodeURIComponent(detection.canonical_tag)}">Graph</a>
          <a class="btn ghost small" href="/ui/field.html?asset=${encodeURIComponent(detection.canonical_tag)}">Field view</a>
        </div>` : ''}
      </div>`;
    node.querySelector('[data-close-detail]')?.addEventListener('click', () => {
      node.innerHTML = '';
    });
    this.options.onSelect?.(detection);
  }
}

/**
 * Open the first drawing an asset appears on, highlighted.
 *
 * Returns false when the asset appears on no digitised drawing, so the caller
 * can say so rather than showing an empty panel — "not detected on any drawing"
 * is information, and a blank box is not.
 */
export async function showAssetOnDrawing(root, assetTag) {
  const located = await api.locateAsset(assetTag);
  if (!located.appearances.length) {
    root.innerHTML = `<div class="notice info" style="margin:14px">
      <strong>Not on any digitised drawing.</strong> ${esc(located.detail || '')}
    </div>`;
    return false;
  }
  const first = located.appearances[0];
  const viewer = new DrawingViewer(root);
  await viewer.load(first.doc_id, first.page, { highlight: assetTag });
  return true;
}
