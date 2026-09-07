/**
 * Force-directed graph renderer — plain SVG, no library.
 *
 * A small Fruchterman–Reingold style simulation: repulsion between every pair
 * (Coulomb), attraction along edges (Hooke), a weak pull to centre, and a
 * cooling schedule. At the node counts a single asset's neighbourhood produces
 * (tens, not thousands) the naive O(n²) repulsion is far cheaper than the
 * quadtree that would replace it, and it is a few dozen lines instead of a
 * dependency.
 *
 * Node colour encodes the graph label — the ontology is what the picture is
 * about, so it is what the colour should carry.
 */

const LABEL_COLOURS = {
  Equipment: '#35c8d8',
  Instrument: '#7fc4a8',
  FunctionalLocation: '#e0a53a',
  Document: '#6fa8dc',
  Chunk: '#4a5a72',
  Mention: '#3d4a5f',
  WorkOrder: '#b07fd0',
  Incident: '#e05a4e',
  Inspection: '#d59a6a',
  CML: '#8b9bb4',
  MOC: '#e07fb0',
  EquipmentClass: '#5b8def',
  FailureMode: '#c96a6a',
  Requirement: '#9a8fd0',
  Loop: '#6fb8c4',
  Default: '#6f8199',
};

export function labelColour(labels = []) {
  for (const label of labels) {
    if (LABEL_COLOURS[label]) return LABEL_COLOURS[label];
  }
  return LABEL_COLOURS.Default;
}

export function nodeCaption(node) {
  const props = node.properties || {};
  return (
    props.canonical_tag || props.tag || props.fl_tag || props.wo_id || props.incident_id ||
    props.inspection_id || props.cml_id || props.moc_id || props.code || props.req_id ||
    props.title || props.chunk_id || node.labels?.[0] || node.id
  );
}

export class GraphView {
  /**
   * @param {SVGSVGElement} svg
   * @param {{onSelectNode?: Function, onSelectEdge?: Function}} handlers
   */
  constructor(svg, handlers = {}) {
    this.svg = svg;
    this.handlers = handlers;
    this.nodes = [];
    this.edges = [];
    this.frame = null;
    this.transform = { x: 0, y: 0, k: 1 };
    this.selected = null;
    this.#bindInteraction();
  }

  setData({ nodes = [], edges = [] }, anchorTag = null) {
    const width = this.svg.clientWidth || 900;
    const height = this.svg.clientHeight || 560;

    // Seed positions on a circle: a random start makes the first frames thrash
    // and the layout settle differently on every load.
    this.nodes = nodes.map((node, index) => {
      const angle = (index / Math.max(nodes.length, 1)) * Math.PI * 2;
      const isAnchor = anchorTag && nodeCaption(node) === anchorTag;
      return {
        ...node,
        x: width / 2 + Math.cos(angle) * (isAnchor ? 0 : 220),
        y: height / 2 + Math.sin(angle) * (isAnchor ? 0 : 220),
        vx: 0,
        vy: 0,
        isAnchor: Boolean(isAnchor),
        degree: 0,
      };
    });

    const byId = new Map(this.nodes.map((n) => [n.id, n]));
    this.edges = edges
      .filter((edge) => byId.has(edge.source) && byId.has(edge.target))
      .map((edge) => ({ ...edge, s: byId.get(edge.source), t: byId.get(edge.target) }));
    for (const edge of this.edges) { edge.s.degree += 1; edge.t.degree += 1; }

    this.alpha = 1;
    this.#run();
  }

  #run() {
    cancelAnimationFrame(this.frame);

    // Settle and draw once, synchronously, before handing over to the animation
    // loop.
    //
    // requestAnimationFrame does not fire at all while a page is not being
    // painted -- a background tab, a collapsed <details>, a hidden pane. Without
    // this the very first frame never runs and the graph is a blank rectangle
    // that looks exactly like a failed fetch. Stepping a few times first also
    // means the layout is already roughly settled when it does become visible,
    // instead of exploding outwards while someone watches.
    for (let i = 0; i < 40; i += 1) this.#step();
    this.#draw();

    const tick = () => {
      this.#step();
      this.#draw();
      this.alpha *= 0.985;
      if (this.alpha > 0.008) this.frame = requestAnimationFrame(tick);
    };
    this.frame = requestAnimationFrame(tick);

    // Re-settle when the container finally gets a real width. The simulation
    // centres nodes on the container, so a layout computed against a
    // zero-width box puts every node in the same place; without this they stay
    // there once the box is real.
    if (!this.observer && typeof ResizeObserver !== 'undefined') {
      let lastWidth = this.svg.clientWidth;
      this.observer = new ResizeObserver(() => {
        const width = this.svg.clientWidth;
        if (width > 0 && Math.abs(width - lastWidth) > 40) {
          lastWidth = width;
          this.alpha = 0.6;
          this.#run();
        }
      });
      this.observer.observe(this.svg);
    }
  }

  /** Release the resize observer and the animation frame. */
  destroy() {
    cancelAnimationFrame(this.frame);
    this.observer?.disconnect();
    this.observer = null;
  }

  #step() {
    const width = this.svg.clientWidth || 900;
    const height = this.svg.clientHeight || 560;
    const cx = width / 2;
    const cy = height / 2;
    const repulsion = 5200;
    const springLength = 92;
    const springK = 0.02;

    for (let i = 0; i < this.nodes.length; i += 1) {
      const a = this.nodes[i];
      for (let j = i + 1; j < this.nodes.length; j += 1) {
        const b = this.nodes[j];
        let dx = a.x - b.x;
        let dy = a.y - b.y;
        let distSq = dx * dx + dy * dy;
        if (distSq < 1) { dx = Math.random() - 0.5; dy = Math.random() - 0.5; distSq = 1; }
        const force = repulsion / distSq;
        const dist = Math.sqrt(distSq);
        const fx = (dx / dist) * force;
        const fy = (dy / dist) * force;
        a.vx += fx; a.vy += fy;
        b.vx -= fx; b.vy -= fy;
      }
    }

    for (const edge of this.edges) {
      const dx = edge.t.x - edge.s.x;
      const dy = edge.t.y - edge.s.y;
      const dist = Math.max(Math.sqrt(dx * dx + dy * dy), 1);
      const force = (dist - springLength) * springK;
      const fx = (dx / dist) * force;
      const fy = (dy / dist) * force;
      edge.s.vx += fx; edge.s.vy += fy;
      edge.t.vx -= fx; edge.t.vy -= fy;
    }

    for (const node of this.nodes) {
      if (node.fixed) { node.vx = 0; node.vy = 0; continue; }
      // The anchor is pinned harder: the picture is about that asset, so it
      // should stay where the eye expects it.
      const pull = node.isAnchor ? 0.06 : 0.012;
      node.vx += (cx - node.x) * pull;
      node.vy += (cy - node.y) * pull;
      node.vx *= 0.82;
      node.vy *= 0.82;
      node.x += node.vx * this.alpha;
      node.y += node.vy * this.alpha;
      node.x = Math.max(40, Math.min(width - 40, node.x));
      node.y = Math.max(30, Math.min(height - 30, node.y));
    }
  }

  #draw() {
    const { x, y, k } = this.transform;
    const edgeMarkup = this.edges.map((edge) => {
      const dim = this.selected && edge.s.id !== this.selected && edge.t.id !== this.selected;
      return `<line class="gv-edge" data-edge="${edge.id}"
        x1="${edge.s.x.toFixed(1)}" y1="${edge.s.y.toFixed(1)}"
        x2="${edge.t.x.toFixed(1)}" y2="${edge.t.y.toFixed(1)}"
        stroke="${dim ? '#1a2435' : '#2e3f5c'}" stroke-width="${edge.evidence_chunk_ids?.length ? 1.8 : 1}"
        ${edge.evidence_chunk_ids?.length ? '' : 'stroke-dasharray="3 3"'}><title>${escapeXml(edge.type)}</title></line>`;
    }).join('');

    const edgeLabels = this.edges
      .filter((edge) => this.selected && (edge.s.id === this.selected || edge.t.id === this.selected))
      .map((edge) => `<text class="gv-edge-label" x="${((edge.s.x + edge.t.x) / 2).toFixed(1)}"
        y="${((edge.s.y + edge.t.y) / 2 - 3).toFixed(1)}" text-anchor="middle"
        fill="#a9b8ce" font-size="8.5">${escapeXml(edge.type)}</text>`).join('');

    const nodeMarkup = this.nodes.map((node) => {
      const colour = labelColour(node.labels);
      const radius = node.isAnchor ? 13 : Math.min(10, 5 + node.degree * 0.55);
      const dim = this.selected && node.id !== this.selected &&
        !this.edges.some((e) => (e.s.id === this.selected && e.t.id === node.id) ||
                                (e.t.id === this.selected && e.s.id === node.id));
      return `<g class="gv-node" data-node="${node.id}" transform="translate(${node.x.toFixed(1)},${node.y.toFixed(1)})"
                 opacity="${dim ? 0.28 : 1}">
        <circle r="${radius}" fill="${colour}" fill-opacity="${node.isAnchor ? 0.95 : 0.72}"
                stroke="${node.id === this.selected ? '#e8eef7' : colour}" stroke-width="${node.isAnchor ? 2.2 : 1.2}"/>
        <text y="${radius + 11}" text-anchor="middle" font-size="9.5"
              fill="${node.isAnchor ? '#e8eef7' : '#a9b8ce'}"
              font-weight="${node.isAnchor ? 700 : 400}">${escapeXml(truncate(nodeCaption(node), 22))}</text>
        <title>${escapeXml((node.labels || []).join(':'))} — ${escapeXml(nodeCaption(node))}</title>
      </g>`;
    }).join('');

    this.svg.innerHTML =
      `<g transform="translate(${x},${y}) scale(${k})">${edgeMarkup}${edgeLabels}${nodeMarkup}</g>`;
  }

  #bindInteraction() {
    let dragging = null;
    let panning = null;

    this.svg.addEventListener('pointerdown', (event) => {
      const nodeEl = event.target.closest('[data-node]');
      if (nodeEl) {
        const node = this.nodes.find((n) => n.id === nodeEl.dataset.node);
        if (node) {
          dragging = node;
          node.fixed = true;
          this.svg.setPointerCapture(event.pointerId);
        }
        return;
      }
      panning = { x: event.clientX - this.transform.x, y: event.clientY - this.transform.y };
      this.svg.setPointerCapture(event.pointerId);
    });

    this.svg.addEventListener('pointermove', (event) => {
      if (dragging) {
        const rect = this.svg.getBoundingClientRect();
        dragging.x = (event.clientX - rect.left - this.transform.x) / this.transform.k;
        dragging.y = (event.clientY - rect.top - this.transform.y) / this.transform.k;
        this.#draw();
      } else if (panning) {
        this.transform.x = event.clientX - panning.x;
        this.transform.y = event.clientY - panning.y;
        this.#draw();
      }
    });

    const release = () => {
      if (dragging) { dragging.fixed = false; dragging = null; this.alpha = Math.max(this.alpha, 0.25); this.#run(); }
      panning = null;
    };
    this.svg.addEventListener('pointerup', release);
    this.svg.addEventListener('pointercancel', release);

    this.svg.addEventListener('wheel', (event) => {
      event.preventDefault();
      const factor = event.deltaY < 0 ? 1.12 : 1 / 1.12;
      this.transform.k = Math.max(0.25, Math.min(3.5, this.transform.k * factor));
      this.#draw();
    }, { passive: false });

    this.svg.addEventListener('click', (event) => {
      const edgeEl = event.target.closest('[data-edge]');
      if (edgeEl) {
        const edge = this.edges.find((e) => e.id === edgeEl.dataset.edge);
        if (edge) this.handlers.onSelectEdge?.(edge);
        return;
      }
      const nodeEl = event.target.closest('[data-node]');
      if (nodeEl) {
        this.selected = nodeEl.dataset.node;
        const node = this.nodes.find((n) => n.id === this.selected);
        this.#draw();
        if (node) this.handlers.onSelectNode?.(node);
      } else {
        this.selected = null;
        this.#draw();
      }
    });
  }

  reset() {
    this.transform = { x: 0, y: 0, k: 1 };
    this.alpha = 1;
    this.#run();
  }

  legend() {
    const present = new Set();
    for (const node of this.nodes) {
      for (const label of node.labels || []) {
        if (LABEL_COLOURS[label]) present.add(label);
      }
    }
    return [...present].sort();
  }
}

function truncate(value, max) {
  const text = String(value ?? '');
  return text.length > max ? `${text.slice(0, max - 1)}…` : text;
}

function escapeXml(value) {
  return String(value ?? '')
    .replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;').replaceAll("'", '&apos;');
}

export { LABEL_COLOURS };
