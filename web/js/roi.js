/**
 * ROI model.
 *
 * The point of this file is that nothing in it is a claim. Every output is
 * arithmetic over inputs the reader can see and change, and every input is
 * labelled with where its value comes from:
 *
 *   USER INPUT   the reader typed it, or accepted a default they can see
 *   ASSUMPTION   a plausible figure that has NOT been measured here
 *   MEASURED     a figure this project actually measured, with a method
 *   CALCULATED   derived from the above by the formula shown
 *
 * The distinction that matters most is ASSUMPTION versus MEASURED. Time saved
 * per query is an assumption — the study that would measure it is specified in
 * docs/time-to-answer.md and has **not been run**. Presenting it as measured
 * would make every downstream number a fabrication, so the calculator shows the
 * label next to the field and repeats it on the result.
 *
 * There is no hard-coded payback period, no "typical customer sees…", and no
 * default that quietly encodes an optimistic answer. Set time saved to zero and
 * the benefit is zero.
 */

/**
 * The model's inputs, with provenance and rationale for every default.
 *
 * Defaults are deliberately conservative where they are assumptions. An ROI
 * model whose defaults flatter it is a sales tool, and the first person to
 * change a number and watch the answer collapse stops believing any of it.
 */
export const FIELDS = [
  // --- population -----------------------------------------------------------
  {
    key: 'users', label: 'People using the system', unit: 'people',
    value: 40, min: 1, max: 100000, provenance: 'USER INPUT',
    help: 'Engineers, technicians and operators who would search for documented facts.',
  },
  {
    key: 'queriesPerUserPerDay', label: 'Searches per person per day', unit: 'searches',
    value: 4, min: 0, max: 200, provenance: 'USER INPUT',
    help: 'How often someone needs a fact out of a document. Count only searches, not all work.',
  },
  {
    key: 'workingDays', label: 'Working days per year', unit: 'days',
    value: 240, min: 1, max: 366, provenance: 'USER INPUT',
    help: '365 less weekends, public holidays and leave.',
  },

  // --- time -----------------------------------------------------------------
  {
    key: 'manualMinutes', label: 'Minutes to find an answer manually', unit: 'min',
    value: 12, min: 0, max: 600, provenance: 'ASSUMPTION',
    help:
      'NOT MEASURED. A placeholder until the time-to-answer study runs — see ' +
      'docs/time-to-answer.md. Replace it with your own timing before trusting any output.',
  },
  {
    key: 'assistedMinutes', label: 'Minutes with the system', unit: 'min',
    value: 3, min: 0, max: 600, provenance: 'ASSUMPTION',
    help:
      'NOT MEASURED, same study. Includes reading the answer and opening the cited ' +
      'source to check it — a figure that assumes the answer is trusted unread is ' +
      'measuring something nobody should want.',
  },
  {
    key: 'adoptionPct', label: 'Of those searches, share actually done in the system', unit: '%',
    value: 60, min: 0, max: 100, provenance: 'ASSUMPTION',
    help:
      'NOT MEASURED. Tools are used for some questions and not others. 100% assumes ' +
      'perfect adoption from day one, which no deployment achieves.',
  },

  // --- cost -----------------------------------------------------------------
  {
    key: 'loadedCostPerHour', label: 'Loaded labour cost per hour', unit: 'currency/h',
    value: 2200, min: 0, max: 1000000, provenance: 'USER INPUT',
    help: 'Salary plus employer costs, overheads and non-productive time. Your finance team has this number.',
  },

  // --- downtime -------------------------------------------------------------
  {
    key: 'downtimeEventsPerYear', label: 'Avoidable downtime events per year', unit: 'events',
    value: 6, min: 0, max: 10000, provenance: 'USER INPUT',
    help:
      'Events where the knowledge to prevent it existed somewhere and was not found — ' +
      'a repeat failure, an action raised and never closed. Count from your own records.',
  },
  {
    key: 'downtimeCostPerEvent', label: 'Cost per downtime event', unit: 'currency',
    value: 850000, min: 0, max: 1000000000, provenance: 'USER INPUT',
    help: 'Lost production, expedited parts and overtime. Your own figure.',
  },
  {
    key: 'downtimeAvoidedPct', label: 'Share of those events the system could prevent', unit: '%',
    value: 15, min: 0, max: 100, provenance: 'ASSUMPTION',
    help:
      'NOT MEASURED, and the single most sensitive input in this model. Surfacing an ' +
      'open corrective action does not close it — someone still has to act. Left low ' +
      'on purpose; anything above about 25% is a claim needing evidence.',
  },

  // --- audit ----------------------------------------------------------------
  {
    key: 'auditsPerYear', label: 'Audits and inspections per year', unit: 'audits',
    value: 4, min: 0, max: 500, provenance: 'USER INPUT',
    help: 'Statutory, certification and internal audits requiring evidence packs.',
  },
  {
    key: 'auditPrepHours', label: 'Person-hours preparing each audit', unit: 'h',
    value: 80, min: 0, max: 10000, provenance: 'USER INPUT',
    help: 'Time spent assembling evidence, not the audit itself.',
  },
  {
    key: 'auditReductionPct', label: 'Share of preparation the system removes', unit: '%',
    value: 30, min: 0, max: 100, provenance: 'ASSUMPTION',
    help:
      'NOT MEASURED. Evidence assembly is where a linked corpus helps most, but ' +
      'someone still reviews and signs what it assembled.',
  },

  // --- cost of ownership ----------------------------------------------------
  {
    key: 'implementationCost', label: 'One-off implementation cost', unit: 'currency',
    value: 4000000, min: 0, max: 1000000000, provenance: 'USER INPUT',
    help: 'Integration, document migration, entity-resolution review and change management.',
  },
  {
    key: 'annualRunCost', label: 'Annual running cost', unit: 'currency',
    value: 1200000, min: 0, max: 1000000000, provenance: 'USER INPUT',
    help: 'Infrastructure, model hosting if used, and the ongoing curation this needs to stay accurate.',
  },
];

/**
 * Run the model.
 *
 * Every returned figure carries the formula that produced it, so the page can
 * show the arithmetic rather than asking to be believed.
 */
export function calculate(input) {
  const v = Object.fromEntries(FIELDS.map((f) => [f.key, Number(input[f.key] ?? f.value)]));

  // --- search time ---------------------------------------------------------
  const searchesPerYear = v.users * v.queriesPerUserPerDay * v.workingDays;
  const assistedSearches = searchesPerYear * (v.adoptionPct / 100);
  const minutesSavedPerSearch = Math.max(0, v.manualMinutes - v.assistedMinutes);
  const hoursSaved = (assistedSearches * minutesSavedPerSearch) / 60;
  const searchBenefit = hoursSaved * v.loadedCostPerHour;

  // --- downtime ------------------------------------------------------------
  const eventsAvoided = v.downtimeEventsPerYear * (v.downtimeAvoidedPct / 100);
  const downtimeBenefit = eventsAvoided * v.downtimeCostPerEvent;

  // --- audit ---------------------------------------------------------------
  const auditHoursSaved = v.auditsPerYear * v.auditPrepHours * (v.auditReductionPct / 100);
  const auditBenefit = auditHoursSaved * v.loadedCostPerHour;

  // --- totals --------------------------------------------------------------
  const annualBenefit = searchBenefit + downtimeBenefit + auditBenefit;
  const annualNet = annualBenefit - v.annualRunCost;
  // Payback is undefined, not infinite, when the system does not pay for itself.
  // Rendering "∞ months" invites the reader to treat it as a large number rather
  // than as "never".
  const paybackMonths = annualNet > 0 ? (v.implementationCost / annualNet) * 12 : null;
  const firstYearNet = annualBenefit - v.annualRunCost - v.implementationCost;
  const roiPct = v.implementationCost + v.annualRunCost > 0
    ? (annualNet / (v.implementationCost + v.annualRunCost)) * 100
    : null;

  return {
    inputs: v,
    lines: [
      {
        key: 'searchesPerYear', label: 'Searches per year', value: searchesPerYear,
        formula: 'people × searches/person/day × working days', unit: 'searches',
      },
      {
        key: 'assistedSearches', label: 'Searches done in the system', value: assistedSearches,
        formula: 'searches/year × adoption%', unit: 'searches',
      },
      {
        key: 'minutesSavedPerSearch', label: 'Minutes saved per search',
        value: minutesSavedPerSearch, formula: 'manual minutes − assisted minutes', unit: 'min',
      },
      {
        key: 'hoursSaved', label: 'Person-hours released per year', value: hoursSaved,
        formula: 'assisted searches × minutes saved ÷ 60', unit: 'h',
      },
      {
        key: 'searchBenefit', label: 'Value of time released', value: searchBenefit,
        formula: 'hours × loaded cost/hour', unit: 'currency', group: 'benefit',
      },
      {
        key: 'eventsAvoided', label: 'Downtime events avoided', value: eventsAvoided,
        formula: 'events/year × avoidable%', unit: 'events',
      },
      {
        key: 'downtimeBenefit', label: 'Value of downtime avoided', value: downtimeBenefit,
        formula: 'events avoided × cost/event', unit: 'currency', group: 'benefit',
      },
      {
        key: 'auditHoursSaved', label: 'Audit preparation hours saved', value: auditHoursSaved,
        formula: 'audits × prep hours × reduction%', unit: 'h',
      },
      {
        key: 'auditBenefit', label: 'Value of audit time saved', value: auditBenefit,
        formula: 'audit hours × loaded cost/hour', unit: 'currency', group: 'benefit',
      },
      {
        key: 'annualBenefit', label: 'Total annual benefit', value: annualBenefit,
        formula: 'search + downtime + audit benefit', unit: 'currency', group: 'total',
      },
      {
        key: 'annualNet', label: 'Annual net benefit', value: annualNet,
        formula: 'annual benefit − annual running cost', unit: 'currency', group: 'total',
      },
      {
        key: 'firstYearNet', label: 'First-year net', value: firstYearNet,
        formula: 'annual benefit − running cost − implementation', unit: 'currency', group: 'total',
      },
      {
        key: 'paybackMonths', label: 'Payback period', value: paybackMonths,
        formula: 'implementation ÷ annual net × 12', unit: 'months', group: 'total',
        undefinedNote: 'Never — the annual net benefit is not positive at these inputs.',
      },
      {
        key: 'roiPct', label: 'Return on total cost', value: roiPct,
        formula: 'annual net ÷ (implementation + annual running) × 100', unit: '%', group: 'total',
      },
    ],
    // Which inputs are unmeasured, so the page can say how much of the answer
    // rests on them rather than leaving the reader to work it out.
    assumptions: FIELDS.filter((f) => f.provenance === 'ASSUMPTION').map((f) => f.key),
  };
}

/**
 * How much of the headline depends on the three unmeasured inputs.
 *
 * A single ROI number invites belief. Showing what happens when the assumptions
 * are halved and doubled shows the reader the range they are actually being
 * offered, which is the honest form of this calculation.
 */
export function sensitivity(input) {
  const base = calculate(input).lines.find((l) => l.key === 'annualNet').value;
  const scenarios = [
    { label: 'Assumptions halved', factor: 0.5 },
    { label: 'As entered', factor: 1 },
    { label: 'Assumptions doubled', factor: 2 },
  ];
  return scenarios.map(({ label, factor }) => {
    const adjusted = { ...input };
    for (const key of ['manualMinutes', 'downtimeAvoidedPct', 'auditReductionPct', 'adoptionPct']) {
      const field = FIELDS.find((f) => f.key === key);
      const raw = Number(input[key] ?? field.value);
      // Time saved scales through the manual figure; percentages are capped at
      // 100 because doubling 60% adoption cannot mean 120%.
      adjusted[key] = Math.min(field.max, raw * factor);
    }
    const value = calculate(adjusted).lines.find((l) => l.key === 'annualNet').value;
    return { label, value, deltaVsBase: value - base };
  });
}
