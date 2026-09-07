"""Typed API contracts.

These models are the interface between the FastAPI backend and the vanilla-JS
frontend, and they are also what FastAPI turns into the OpenAPI document at
``/docs``. Two conventions run through all of them:

``data_class``
    Every value that can be rendered on a dashboard is tagged with where it came
    from -- a real source document, clearly-labelled synthetic test data, a model
    inference, or a deterministic calculation. The UI renders the badge from this
    field; it is never inferred client-side.

``status`` / ``CapabilityStatus``
    A stage that cannot run says so, names the capability and the environment
    variables required to enable it, and returns whatever real partial result it
    does have. There is no code path that fabricates a value to fill a gap.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Shared vocabulary
# ---------------------------------------------------------------------------


class DataClass(str, Enum):
    """Provenance class of a displayed value. Required on anything renderable."""

    REAL_SOURCE_DOCUMENT = "real_source_document"
    SYNTHETIC_TEST_DATA = "synthetic_test_data"
    MODEL_DERIVED = "model_derived"
    CALCULATED_METRIC = "calculated_metric"
    HUMAN_ATTESTED = "human_attested"
    REFERENCE_TAXONOMY = "reference_taxonomy"


class DocumentType(str, Enum):
    PID = "pid"
    ISOMETRIC = "isometric"
    DATASHEET = "datasheet"
    SOP = "sop"
    WORK_ORDER = "work_order"
    INSPECTION_REPORT = "inspection_report"
    INCIDENT_REPORT = "incident_report"
    MOC = "moc"
    HAZOP = "hazop"
    PERMIT = "permit"
    NCR_CAPA = "ncr_capa"
    REGULATION = "regulation"
    MANUAL = "manual"
    EMAIL = "email"
    UNKNOWN = "unknown"


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    PARTIAL = "partial"
    CANCELLED = "cancelled"


class QueryIntent(str, Enum):
    LOOKUP = "lookup"
    MULTI_HOP = "multi_hop"
    AGGREGATE = "aggregate"
    PROCEDURAL = "procedural"
    DIAGNOSTIC = "diagnostic"
    COMPARATIVE = "comparative"
    UNANSWERABLE = "unanswerable"


class ConfidenceMode(str, Enum):
    ANSWER = "ANSWER"
    ANSWER_WITH_CAVEAT = "ANSWER_WITH_CAVEAT"
    ABSTAIN_AND_ROUTE = "ABSTAIN_AND_ROUTE"
    #: Retrieval succeeded and evidence is returned, but no generation provider
    #: is configured, so no prose answer is synthesised. This is a truthful
    #: configuration state, not a quality failure.
    #: No answer text could be produced at all: extraction found no sentence that
    #: addresses the question, and no LLM is configured to attempt a synthesis.
    #: Distinct from ABSTAIN_AND_ROUTE, which means evidence was judged too weak.
    ABSTAIN_NO_ANSWER = "ABSTAIN_NO_ANSWER"


class CapabilityState(str, Enum):
    AVAILABLE = "available"
    DISABLED = "disabled"
    NOT_CONFIGURED = "provider_not_configured"
    NOT_IMPLEMENTED = "not_implemented"
    ERROR = "error"


class CapabilityStatus(BaseModel):
    """Truthful report for one pipeline stage or feature."""

    capability: str
    state: CapabilityState
    detail: str | None = None
    required_env: list[str] = Field(default_factory=list)


class ErrorBody(BaseModel):
    code: str
    message: str
    detail: dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    error: ErrorBody


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded", "down"]
    app: str
    version: str
    environment: str
    checked_at: datetime
    dependencies: dict[str, dict[str, Any]]
    providers: dict[str, dict[str, Any]]
    connectors: dict[str, str]
    config: dict[str, Any]


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------


class IngestSource(str, Enum):
    UPLOAD = "upload"
    FILESYSTEM = "filesystem"
    S3 = "s3"
    CMMS = "cmms"
    SHAREPOINT = "sharepoint"


class IngestPathRequest(BaseModel):
    """Submit files already present on a path the API container can read.

    Used for the curated corpus under ``data/`` -- the browser upload path uses
    multipart instead.
    """

    source: IngestSource = IngestSource.FILESYSTEM
    paths: list[str] = Field(min_length=1, max_length=500)
    source_system: str = Field(default="filesystem", max_length=64)
    data_class: DataClass
    recursive: bool = True
    submitted_by: str | None = Field(default=None, max_length=64)

    @field_validator("paths")
    @classmethod
    def _reject_traversal(cls, v: list[str]) -> list[str]:
        for p in v:
            if ".." in p.replace("\\", "/").split("/"):
                raise ValueError("path traversal segments are not permitted")
        return v


class AcceptedFile(BaseModel):
    filename: str
    byte_size: int
    accepted: bool
    reason: str | None = None
    content_hash: str | None = None
    duplicate_of: str | None = None


class IngestResponse(BaseModel):
    job_id: str
    status: JobStatus
    accepted: int
    rejected: int
    duplicates: int
    files: list[AcceptedFile]
    queue_depth: int
    poll: str = Field(description="URL to poll for job progress")


class StageReport(BaseModel):
    stage: str
    state: CapabilityState
    detail: str | None = None
    required_env: list[str] = Field(default_factory=list)
    items: int = 0


class ReviewItem(BaseModel):
    review_id: int
    kind: str
    subject: str
    doc_id: str | None = None
    confidence: float | None = None
    detail: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class IngestJobResponse(BaseModel):
    job_id: str
    status: JobStatus
    source: str
    source_system: str | None = None
    stage: str | None = None
    file_count: int
    processed: int
    failed: int
    skipped_duplicates: int
    chunks_created: int
    mentions_created: int
    entities_created: int
    edges_created: int
    #: Pages across every document in the job.
    pages: int = 0
    graph_nodes_created: int = 0
    #: Summed per-document processing time.
    processing_ms: int = 0
    #: Wall-clock duration from job start to finish. Larger than the sum above,
    #: because a job also spends time queued and between documents.
    duration_ms: int | None = None
    extractions_total: int = 0
    extractions_verified: int = 0
    stage_reports: list[StageReport] = Field(default_factory=list)
    documents: list[dict[str, Any]] = Field(default_factory=list)
    review_queue: list[ReviewItem] = Field(default_factory=list)
    error: str | None = None
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None


# ---------------------------------------------------------------------------
# Query / copilot
# ---------------------------------------------------------------------------


class UserContext(BaseModel):
    role: str | None = Field(default=None, max_length=64)
    site: str | None = Field(default=None, max_length=64)
    work_order: str | None = Field(default=None, max_length=64)
    asset_tag: str | None = Field(default=None, max_length=64)


class QueryRequest(BaseModel):
    question: str = Field(min_length=2, max_length=2000)
    user_ctx: UserContext = Field(default_factory=UserContext)
    mode: Literal["auto", "lexical_only", "graph_only"] = "auto"
    top_k: int | None = Field(default=None, ge=1, le=50)


class Citation(BaseModel):
    marker: str
    chunk_id: str
    doc_id: str
    doc_title: str
    doc_type: DocumentType
    data_class: DataClass
    page: int | None = None
    bbox: list[float] | None = None
    section_path: str | None = None
    snippet: str
    quote_verified: bool = False
    retriever: str
    rank: int
    score: float


class GraphFact(BaseModel):
    """A triple pulled from the knowledge graph, with its evidence pointer."""

    subject: str
    predicate: str
    object: str
    properties: dict[str, Any] = Field(default_factory=dict)
    evidence_chunk_ids: list[str] = Field(default_factory=list)
    confidence: float | None = None
    data_class: DataClass


class RetrievalLeg(BaseModel):
    strategy: Literal["lexical", "dense", "graph", "rerank"]
    state: CapabilityState
    candidates: int
    elapsed_ms: float
    detail: str | None = None
    required_env: list[str] = Field(default_factory=list)
    #: Distinguishes repeated runs of the same strategy: the main pass is
    #: ``lexical``, a decomposed pass is ``lexical:sub2``. Kept separate from
    #: ``strategy`` so the UI can still group by strategy without parsing.
    leg_id: str | None = None
    #: The sub-question this leg retrieved for, when it is not the main pass.
    sub_question: str | None = None


class ConfidenceReport(BaseModel):
    score: float
    mode: ConfidenceMode
    signals: dict[str, float]
    explanation: str


class SuggestedAction(BaseModel):
    label: str
    action_type: str
    payload: dict[str, Any] = Field(default_factory=dict)


class AnswerClaim(BaseModel):
    """One factual claim in the answer, bound to the evidence that supports it.

    The copilot's core promise is that a claim and its evidence travel together.
    Splitting the answer into claims at the API boundary -- rather than leaving a
    paragraph with markers embedded in it -- is what lets the UI highlight the
    exact sentence in the exact source, and what makes "unsupported claim" a
    measurable quantity instead of an assurance.
    """

    text: str
    marker: str
    chunk_id: str
    doc_id: str
    doc_title: str
    page: int | None = None
    char_start: int | None = None
    char_end: int | None = None
    verbatim: bool = False
    score: float | None = None


class GraphEntityRef(BaseModel):
    """A graph node that took part in answering, and how it was reached."""

    node_id: str
    label: str
    display: str
    role: str  # anchor | neighbour
    hops: int = 0
    data_class: DataClass | None = None


class QueryResponse(BaseModel):
    query_id: str
    question: str
    intent: QueryIntent
    intent_confidence: float
    intent_method: str
    sub_questions: list[str] = Field(default_factory=list)
    resolved_entities: list[dict[str, Any]] = Field(default_factory=list)
    answer: str | None = None
    answer_data_class: DataClass | None = None
    #: ``extractive`` (verbatim spans selected from the corpus) or ``abstractive``
    #: (LLM-written prose). Displayed, because the two carry different risks and
    #: the reader is entitled to know which one they are reading.
    answer_method: str | None = None
    claims: list[AnswerClaim] = Field(default_factory=list)
    abstained: bool = False
    generation: CapabilityStatus
    citations: list[Citation] = Field(default_factory=list)
    graph_facts: list[GraphFact] = Field(default_factory=list)
    graph_entities: list[GraphEntityRef] = Field(default_factory=list)
    retrieval: list[RetrievalLeg] = Field(default_factory=list)
    #: Which retrieval legs actually contributed a document to the final set.
    retrieval_sources: list[str] = Field(default_factory=list)
    confidence: ConfidenceReport
    referral: dict[str, Any] | None = None
    actions: list[SuggestedAction] = Field(default_factory=list)
    latency_ms: int


# ---------------------------------------------------------------------------
# Assets and graph
# ---------------------------------------------------------------------------


class AssetSummary(BaseModel):
    asset_id: str
    canonical_tag: str
    tag_kind: str
    class_code: str | None = None
    class_label: str | None = None
    description: str | None = None
    functional_location: str | None = None
    site: str | None = None
    data_class: DataClass
    mention_count: int
    document_count: int
    source_system_count: int
    updated_at: datetime


class AssetListResponse(BaseModel):
    total: int
    limit: int
    offset: int
    items: list[AssetSummary]


class AssetDocument(BaseModel):
    doc_id: str
    title: str
    doc_type: DocumentType
    data_class: DataClass
    source_system: str
    revision: str | None = None
    issued_on: date | None = None
    is_current: bool
    mention_count: int


class AssetDetailResponse(BaseModel):
    asset: AssetSummary
    tag_variants: list[dict[str, Any]] = Field(default_factory=list)
    documents: list[AssetDocument] = Field(default_factory=list)
    work_orders: list[dict[str, Any]] = Field(default_factory=list)
    incidents: list[dict[str, Any]] = Field(default_factory=list)
    inspections: list[dict[str, Any]] = Field(default_factory=list)
    siblings: list[dict[str, Any]] = Field(default_factory=list)
    graph_available: CapabilityStatus


class GraphNode(BaseModel):
    id: str
    labels: list[str]
    properties: dict[str, Any] = Field(default_factory=dict)
    data_class: DataClass | None = None


class GraphEdge(BaseModel):
    id: str
    type: str
    source: str
    target: str
    properties: dict[str, Any] = Field(default_factory=dict)
    evidence_chunk_ids: list[str] = Field(default_factory=list)
    confidence: float | None = None


class GraphResponse(BaseModel):
    anchor: str
    anchor_found: bool
    hops: int
    edge_types: list[str]
    as_of: date | None = None
    nodes: list[GraphNode] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)
    truncated: bool = False
    detail: str | None = None


# ---------------------------------------------------------------------------
# RCA
# ---------------------------------------------------------------------------


class EvidenceRef(BaseModel):
    chunk_id: str | None = None
    doc_id: str | None = None
    page: int | None = None
    quote: str | None = None
    quote_verified: bool = False
    data_class: DataClass


class CausalNode(BaseModel):
    level: Literal["failure_mode", "mechanism", "proximate_cause", "systemic_cause"]
    statement: str
    confidence: float
    supporting_evidence: list[EvidenceRef] = Field(default_factory=list)
    contradicting_evidence: list[EvidenceRef] = Field(default_factory=list)
    children: list[CausalNode] = Field(default_factory=list)


class Recommendation(BaseModel):
    action: str
    addresses_node: str
    rationale: str
    duplicate_of_capa: str | None = None
    evidence: list[EvidenceRef] = Field(default_factory=list)


class RCARequest(BaseModel):
    asset_tag: str = Field(min_length=1, max_length=64)
    failure_description: str = Field(min_length=3, max_length=2000)
    event_date: date | None = None


class RCAResponse(BaseModel):
    asset_tag: str
    asset_found: bool
    event: str
    status: CapabilityStatus
    #: Counts of evidence actually retrieved, per source. Integers only.
    evidence_gathered: dict[str, int] = Field(default_factory=dict)
    #: Deterministic reliability computations over that evidence. Kept separate
    #: from the counts because they are a different kind of value: they carry
    #: units, can be null, and report ``insufficient_data`` rather than a number
    #: when the event history is too thin to support one.
    reliability_metrics: dict[str, Any] = Field(default_factory=dict)
    causal_tree: CausalNode | None = None
    ruled_out: list[dict[str, Any]] = Field(default_factory=list)
    discriminating_evidence_needed: list[str] = Field(default_factory=list)
    recommendations: list[Recommendation] = Field(default_factory=list)
    similar_events: list[dict[str, Any]] = Field(default_factory=list)
    duplicate_of_open_capa: str | None = None
    overall_confidence: float | None = None
    citations: list[Citation] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Compliance
# ---------------------------------------------------------------------------


class ComplianceScope(BaseModel):
    standard: str | None = Field(default=None, max_length=64)
    plant: str | None = Field(default=None, max_length=64)
    asset_tag: str | None = Field(default=None, max_length=64)
    years: int = Field(default=3, ge=1, le=25)


class ComplianceRequest(BaseModel):
    scope: ComplianceScope = Field(default_factory=ComplianceScope)


class ComplianceGap(BaseModel):
    req_id: str
    source_standard: str
    clause: str
    obligation_text: str
    modality: str
    gap_type: Literal["no_control", "content_gap", "evidence_stale", "evidence_missing"]
    severity: Literal["high", "medium", "low"]
    asset_tag: str | None = None
    latest_evidence: date | None = None
    detail: str
    data_class: DataClass


class ComplianceResponse(BaseModel):
    scope: ComplianceScope
    status: CapabilityStatus
    requirements_in_scope: int
    satisfied: int
    partial: int
    gaps: list[ComplianceGap] = Field(default_factory=list)
    coverage_pct: float | None = None
    requirement_provenance: dict[str, int] = Field(
        default_factory=dict,
        description="Counts by text_status, so the UI can state whether requirement "
        "text is verbatim or a demo paraphrase.",
    )
    evaluated_at: datetime


# ---------------------------------------------------------------------------
# Notifications (proactive path)
# ---------------------------------------------------------------------------


class Notification(BaseModel):
    notification_id: int
    severity: Literal["critical", "high", "medium", "info"]
    title: str
    message: str
    reason: str
    asset_tag: str | None = None
    audience_role: str | None = None
    evidence: list[EvidenceRef] = Field(default_factory=list)
    pattern_id: str | None = None
    match_score: float | None = None
    data_class: DataClass
    acknowledged: bool
    created_at: datetime


class NotificationListResponse(BaseModel):
    total: int
    items: list[Notification]
    engine: CapabilityStatus


# ---------------------------------------------------------------------------
# Feedback
# ---------------------------------------------------------------------------


class LoggedCitation(BaseModel):
    marker: str
    chunk_id: str | None = None
    doc_id: str | None = None
    page: int | None = None
    quote: str | None = None
    quote_verified: bool = False
    retriever: str | None = None
    rank: int | None = None
    score: float | None = None


class QueryLogResponse(BaseModel):
    """A previously answered query, replayed from the audit log.

    Deliberately a different shape from :class:`QueryResponse`: this is the
    record of what was said at the time, not a fresh answer. Nothing is
    recomputed, which is what an audit requires.
    """

    query_id: str
    question: str
    normalised_question: str
    intent: str | None = None
    intent_confidence: float | None = None
    entities: list[dict[str, Any]] = Field(default_factory=list)
    retrieval_stats: dict[str, Any] = Field(default_factory=dict)
    answer_text: str | None = None
    confidence_score: float | None = None
    confidence_mode: str | None = None
    abstained: bool = False
    generator_provider: str = "none"
    latency_ms: int | None = None
    user_role: str | None = None
    user_site: str | None = None
    work_order_ctx: str | None = None
    citations: list[LoggedCitation] = Field(default_factory=list)
    created_at: datetime


class FeedbackRequest(BaseModel):
    query_id: str | None = None
    notification_id: int | None = None
    verdict: Literal["up", "down"]
    reason: Literal["wrong_source", "outdated", "incomplete", "wrong_asset", "other"] | None = None
    note: str | None = Field(default=None, max_length=2000)
    submitted_by: str | None = Field(default=None, max_length=64)


class FeedbackResponse(BaseModel):
    feedback_id: int
    queued_for_review: bool


CausalNode.model_rebuild()
