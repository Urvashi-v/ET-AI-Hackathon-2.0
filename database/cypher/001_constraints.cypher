// =============================================================================
// 001_constraints.cypher -- knowledge graph constraints and indexes.
//
// Run at every API startup. Every statement is IF NOT EXISTS, so the script is
// idempotent and doubles as a schema assertion: if it runs clean, the graph is
// shaped the way the retrieval layer expects.
//
// Uniqueness constraints are what make the ingestion MERGE pattern safe. Without
// `equip_canonical_tag` a re-ingest creates a second P-101B, and every failure
// statistic computed over the graph silently halves.
// =============================================================================

// --- Asset hierarchy (ISA-95 / ISO 14224 shape) ------------------------------
CREATE CONSTRAINT site_code IF NOT EXISTS
FOR (n:Site) REQUIRE n.code IS UNIQUE;

CREATE CONSTRAINT plant_code IF NOT EXISTS
FOR (n:Plant) REQUIRE n.code IS UNIQUE;

CREATE CONSTRAINT area_code IF NOT EXISTS
FOR (n:Area) REQUIRE n.code IS UNIQUE;

CREATE CONSTRAINT system_code IF NOT EXISTS
FOR (n:System) REQUIRE n.code IS UNIQUE;

// The permanent position in the process. Distinct from the machine occupying it:
// "does this position keep killing pumps?" and "does this machine keep failing
// wherever we install it?" are different questions with different answers.
CREATE CONSTRAINT fl_tag IF NOT EXISTS
FOR (n:FunctionalLocation) REQUIRE n.fl_tag IS UNIQUE;

CREATE CONSTRAINT equip_canonical_tag IF NOT EXISTS
FOR (n:Equipment) REQUIRE n.canonical_tag IS UNIQUE;

CREATE CONSTRAINT component_id IF NOT EXISTS
FOR (n:Component) REQUIRE n.component_id IS UNIQUE;

CREATE CONSTRAINT part_no IF NOT EXISTS
FOR (n:Part) REQUIRE n.part_no IS UNIQUE;

CREATE CONSTRAINT equipclass_code IF NOT EXISTS
FOR (n:EquipmentClass) REQUIRE n.code IS UNIQUE;

// --- Process topology (reconstructed from P&IDs) ------------------------------
CREATE CONSTRAINT instrument_tag IF NOT EXISTS
FOR (n:Instrument) REQUIRE n.tag IS UNIQUE;

CREATE CONSTRAINT loop_no IF NOT EXISTS
FOR (n:Loop) REQUIRE n.loop_no IS UNIQUE;

CREATE CONSTRAINT line_no IF NOT EXISTS
FOR (n:Line) REQUIRE n.line_no IS UNIQUE;

CREATE CONSTRAINT valve_tag IF NOT EXISTS
FOR (n:Valve) REQUIRE n.tag IS UNIQUE;

// --- Evidence layer -----------------------------------------------------------
CREATE CONSTRAINT document_id IF NOT EXISTS
FOR (n:Document) REQUIRE n.doc_id IS UNIQUE;

CREATE CONSTRAINT chunk_id IF NOT EXISTS
FOR (n:Chunk) REQUIRE n.chunk_id IS UNIQUE;

// The observation layer, deliberately separate from the entity layer so that a
// bad resolution can be undone without re-ingesting the document.
CREATE CONSTRAINT mention_id IF NOT EXISTS
FOR (n:Mention) REQUIRE n.mention_id IS UNIQUE;

// --- History ------------------------------------------------------------------
CREATE CONSTRAINT workorder_id IF NOT EXISTS
FOR (n:WorkOrder) REQUIRE n.wo_id IS UNIQUE;

CREATE CONSTRAINT notification_id IF NOT EXISTS
FOR (n:Notification) REQUIRE n.notification_id IS UNIQUE;

CREATE CONSTRAINT failureevent_id IF NOT EXISTS
FOR (n:FailureEvent) REQUIRE n.event_id IS UNIQUE;

CREATE CONSTRAINT failuremode_code IF NOT EXISTS
FOR (n:FailureMode) REQUIRE n.code IS UNIQUE;

CREATE CONSTRAINT incident_id IF NOT EXISTS
FOR (n:Incident) REQUIRE n.incident_id IS UNIQUE;

CREATE CONSTRAINT inspection_id IF NOT EXISTS
FOR (n:Inspection) REQUIRE n.inspection_id IS UNIQUE;

CREATE CONSTRAINT cml_id IF NOT EXISTS
FOR (n:CML) REQUIRE n.cml_id IS UNIQUE;

CREATE CONSTRAINT moc_id IF NOT EXISTS
FOR (n:MOC) REQUIRE n.moc_id IS UNIQUE;

CREATE CONSTRAINT capa_id IF NOT EXISTS
FOR (n:CAPA) REQUIRE n.capa_id IS UNIQUE;

CREATE CONSTRAINT correctiveaction_id IF NOT EXISTS
FOR (n:CorrectiveAction) REQUIRE n.action_id IS UNIQUE;

// --- Procedures and compliance -------------------------------------------------
CREATE CONSTRAINT sop_id IF NOT EXISTS
FOR (n:SOP) REQUIRE n.proc_id IS UNIQUE;

CREATE CONSTRAINT procedure_id IF NOT EXISTS
FOR (n:Procedure) REQUIRE n.proc_id IS UNIQUE;

CREATE CONSTRAINT step_id IF NOT EXISTS
FOR (n:Step) REQUIRE n.step_id IS UNIQUE;

// A clause number alone is not unique across standards, so req_id embeds the
// standard (e.g. "OISD-105-4.2-a") and is unique on its own. A composite NODE KEY
// over (source_standard, req_id) would express the intent more directly, but NODE
// KEY constraints require Neo4j Enterprise Edition and this stack runs Community.
// The uniqueness guarantee is identical; only the existence guarantee is lost,
// and the loader enforces that instead (scripts/load_requirements.py rejects a
// requirement missing source_standard).
CREATE CONSTRAINT requirement_id IF NOT EXISTS
FOR (n:Requirement) REQUIRE n.req_id IS UNIQUE;

CREATE CONSTRAINT control_id IF NOT EXISTS
FOR (n:Control) REQUIRE n.control_id IS UNIQUE;

CREATE CONSTRAINT evidence_id IF NOT EXISTS
FOR (n:Evidence) REQUIRE n.evidence_id IS UNIQUE;

// --- People and roles (pseudonymised; used for expert routing on abstention) ---
CREATE CONSTRAINT person_id IF NOT EXISTS
FOR (n:Person) REQUIRE n.person_id IS UNIQUE;

CREATE CONSTRAINT role_name IF NOT EXISTS
FOR (n:Role) REQUIRE n.name IS UNIQUE;

// --- Lookup indexes ------------------------------------------------------------
CREATE INDEX chunk_doc IF NOT EXISTS FOR (c:Chunk) ON (c.doc_id);
CREATE INDEX document_type IF NOT EXISTS FOR (d:Document) ON (d.doc_type);
CREATE INDEX document_source_system IF NOT EXISTS FOR (d:Document) ON (d.source_system);
CREATE INDEX equipment_class IF NOT EXISTS FOR (e:Equipment) ON (e.class_code);
CREATE INDEX equipment_site IF NOT EXISTS FOR (e:Equipment) ON (e.site);
CREATE INDEX workorder_opened IF NOT EXISTS FOR (w:WorkOrder) ON (w.opened_on);
CREATE INDEX incident_occurred IF NOT EXISTS FOR (i:Incident) ON (i.occurred_on);
CREATE INDEX inspection_date IF NOT EXISTS FOR (i:Inspection) ON (i.inspected_on);
CREATE INDEX mention_normalised IF NOT EXISTS FOR (m:Mention) ON (m.normalised);
CREATE INDEX requirement_standard IF NOT EXISTS FOR (r:Requirement) ON (r.source_standard);

// Full-text over the tag surface. Used by the query-understanding stage to link
// an entity mentioned in a question to a node in the graph.
CREATE FULLTEXT INDEX entity_text IF NOT EXISTS
FOR (n:Equipment|FunctionalLocation|Instrument)
ON EACH [n.canonical_tag, n.fl_tag, n.tag, n.description];
