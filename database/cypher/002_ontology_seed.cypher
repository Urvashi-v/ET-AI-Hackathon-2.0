// =============================================================================
// 002_ontology_seed.cypher -- taxonomy subgraph.
//
// WHAT THIS IS: reference taxonomy only -- equipment class codes and failure
// mode codes. It contains NO operational data: no assets, no work orders, no
// incidents, no compliance findings. Those only ever come from ingested
// documents.
//
// The blueprint's principle is "model the ontology as data, not code", so these
// classes live as nodes and can be extended without a deploy. Every seeded node
// carries `data_class: 'reference_taxonomy'` and a `provenance` property, so the
// dashboard can label anything derived from them honestly.
//
// PROVENANCE NOTE ON ISO 14224: the codes and labels below are the failure-mode
// vocabulary as commonly cited in public reliability-engineering literature.
// They are used here as a controlled vocabulary for classification. They are NOT
// reproduced verbatim from the paywalled standard, and `provenance` records that.
//
// Idempotent (MERGE on the natural key).
// =============================================================================

// --- Equipment classes (mirrors EQUIPMENT_CLASS_CODES in services/common/tags.py)
UNWIND [
  {code: 'P',  label: 'Pump',                 parent: 'ROTATING'},
  {code: 'C',  label: 'Compressor',           parent: 'ROTATING'},
  {code: 'K',  label: 'Blower',               parent: 'ROTATING'},
  {code: 'M',  label: 'Motor',                parent: 'ELECTRICAL'},
  {code: 'G',  label: 'Generator',            parent: 'ELECTRICAL'},
  {code: 'A',  label: 'Agitator',             parent: 'ROTATING'},
  {code: 'FN', label: 'Fan',                  parent: 'ROTATING'},
  {code: 'E',  label: 'Heat exchanger',       parent: 'STATIC'},
  {code: 'AG', label: 'Air cooler',           parent: 'STATIC'},
  {code: 'V',  label: 'Vessel / drum',        parent: 'STATIC'},
  {code: 'D',  label: 'Drum',                 parent: 'STATIC'},
  {code: 'T',  label: 'Tower / column',       parent: 'STATIC'},
  {code: 'TK', label: 'Tank',                 parent: 'STATIC'},
  {code: 'R',  label: 'Reactor',              parent: 'STATIC'},
  {code: 'S',  label: 'Separator',            parent: 'STATIC'},
  {code: 'F',  label: 'Furnace / filter',     parent: 'STATIC'},
  {code: 'B',  label: 'Boiler',               parent: 'STATIC'},
  {code: 'H',  label: 'Heater',               parent: 'STATIC'},
  {code: 'CV', label: 'Conveyor',             parent: 'MECHANICAL_HANDLING'},
  {code: 'X',  label: 'Miscellaneous / package', parent: 'OTHER'}
] AS ec
MERGE (c:EquipmentClass {code: ec.code})
  ON CREATE SET c.created_at = datetime()
SET c.label = ec.label,
    c.parent = ec.parent,
    c.taxonomy = 'ISO14224-style equipment class',
    c.data_class = 'reference_taxonomy',
    c.provenance = 'Controlled vocabulary maintained in database/cypher/002_ontology_seed.cypher; mirrors services/common/tags.py',
    c.updated_at = datetime();

// --- Failure modes ------------------------------------------------------------
// Three levels must never be conflated (blueprint 14.1):
//   failure MODE      -- how the function was lost (observable)
//   failure MECHANISM -- the physical process (inferred from evidence)
//   root CAUSE        -- the systemic reason it was allowed (actionable)
// Only the mode is a taxonomy entry. Mechanisms and causes are asserted per
// event, with evidence, and never seeded.
UNWIND [
  {code: 'ELP', label: 'External leakage - process medium', applies: 'ROTATING'},
  {code: 'ELU', label: 'External leakage - utility medium', applies: 'ROTATING'},
  {code: 'INL', label: 'Internal leakage',                  applies: 'ROTATING'},
  {code: 'VIB', label: 'Vibration',                         applies: 'ROTATING'},
  {code: 'NOI', label: 'Noise',                             applies: 'ROTATING'},
  {code: 'OHE', label: 'Overheating',                       applies: 'ROTATING'},
  {code: 'STD', label: 'Structural deficiency',             applies: 'ALL'},
  {code: 'FTS', label: 'Fail to start on demand',           applies: 'ROTATING'},
  {code: 'STP', label: 'Fail to stop on demand',            applies: 'ROTATING'},
  {code: 'BRD', label: 'Breakdown',                         applies: 'ALL'},
  {code: 'ERO', label: 'Erratic output',                    applies: 'ROTATING'},
  {code: 'LOO', label: 'Low output',                        applies: 'ROTATING'},
  {code: 'HIO', label: 'High output',                       applies: 'ROTATING'},
  {code: 'PLU', label: 'Plugged / choked',                  applies: 'STATIC'},
  {code: 'CORR',label: 'Corrosion / wall loss',             applies: 'STATIC'},
  {code: 'AOH', label: 'Abnormal instrument reading',       applies: 'INSTRUMENT'},
  {code: 'UNK', label: 'Unknown / insufficient information', applies: 'ALL'}
] AS fm
MERGE (f:FailureMode {code: fm.code})
  ON CREATE SET f.created_at = datetime()
SET f.label = fm.label,
    f.applies_to = fm.applies,
    f.taxonomy = 'ISO14224-style failure mode',
    f.data_class = 'reference_taxonomy',
    f.provenance = 'Failure-mode vocabulary as commonly cited in public reliability literature; not verbatim standard text',
    f.updated_at = datetime();

// --- Roles used for expert routing when the copilot abstains -------------------
// Names are never seeded. Only role slots exist until a real document or a real
// operator assignment populates a :Person.
UNWIND [
  {name: 'field_technician',    scope: 'executes work at the asset'},
  {name: 'reliability_engineer',scope: 'owns failure analysis and maintenance strategy'},
  {name: 'operations_shift',    scope: 'owns run/no-run decisions'},
  {name: 'inspection_engineer', scope: 'owns integrity and condition monitoring'},
  {name: 'hse_officer',         scope: 'owns incident investigation and permits'},
  {name: 'quality_officer',     scope: 'owns non-conformance, CAPA and audits'},
  {name: 'document_owner',      scope: 'owns procedure revision control'}
] AS r
MERGE (role:Role {name: r.name})
  ON CREATE SET role.created_at = datetime()
SET role.scope = r.scope,
    role.data_class = 'reference_taxonomy',
    role.updated_at = datetime();

// --- Ontology self-description -------------------------------------------------
// A single node recording which schema version the graph was seeded with, so
// /health can report it rather than assuming.
MERGE (o:OntologyVersion {version: '0.1.0'})
  ON CREATE SET o.seeded_at = datetime()
SET o.node_labels = [
      'Site','Plant','Area','System','FunctionalLocation','Equipment','Component','Part',
      'EquipmentClass','Instrument','Loop','Line','Valve','Document','Chunk','Mention',
      'WorkOrder','FailureEvent','FailureMode','Incident','Inspection','CML','MOC',
      'SOP','Procedure','Step','Requirement','Control','Evidence','CAPA',
      'CorrectiveAction','Notification','Person','Role'
    ],
    o.relationship_types = [
      'CONTAINS','HAS_POSITION','OCCUPIED_BY','HAS_COMPONENT','USES_PART','SIBLING_OF',
      'INSTANCE_OF','FEEDS','CONNECTS','ISOLATES','MEASURES','BELONGS_TO_LOOP',
      'PART_OF','HAS_CHUNK','APPEARS_IN','RESOLVES_TO','DESCRIBES','SUPERSEDES',
      'PERFORMED_ON','RECORDED','OF_MODE','CAUSED_BY','RESOLVED_BY','INVOLVED',
      'GENERATED','CHANGED','APPLIES_TO','SATISFIES','IMPLEMENTS','PROVES','AMENDED_BY',
      'HAS_STEP','OWNED_BY','RAISED_FOR'
    ],
    o.updated_at = datetime();
