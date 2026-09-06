"""Central configuration.

Every tunable lives here and is read from the environment exactly once. No other
module reads ``os.environ`` directly, so the effective configuration is always
inspectable (``GET /health`` reports the non-secret subset).

Provider settings deliberately default to ``none``. A missing credential is a
*configuration state*, not an error to paper over: the affected stage reports
``provider_not_configured`` and the API degrades truthfully instead of
substituting a fake provider.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

EmbeddingProvider = Literal["none", "openai", "local"]
LLMProvider = Literal["none", "openai", "anthropic"]
RerankerProvider = Literal["none", "local"]
OCRProvider = Literal["none", "paddle", "tesseract"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- app -----------------------------------------------------------------
    app_env: str = "local"
    app_name: str = "industrial-brain"
    app_version: str = "0.1.0"
    log_level: str = "INFO"
    log_format: Literal["json", "console"] = "json"
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # --- postgres ------------------------------------------------------------
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "industrial_brain"
    postgres_user: str = "brain"
    postgres_password: SecretStr = SecretStr("")
    postgres_pool_min: int = 1
    postgres_pool_max: int = 10

    # --- neo4j ---------------------------------------------------------------
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_username: str = "neo4j"
    neo4j_password: SecretStr = SecretStr("")
    neo4j_database: str = "neo4j"

    # --- redis ---------------------------------------------------------------
    redis_url: str = "redis://localhost:6379/0"

    # --- storage -------------------------------------------------------------
    blob_root: str = "./data/blobs"

    # --- providers (all default to "not configured") -------------------------
    embedding_provider: EmbeddingProvider = "none"
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536
    embedding_local_model: str = "BAAI/bge-small-en-v1.5"

    llm_provider: LLMProvider = "none"
    llm_model: str = ""
    llm_timeout_s: int = 60
    llm_max_tokens: int = 1500

    reranker_provider: RerankerProvider = "none"
    reranker_local_model: str = "BAAI/bge-reranker-base"

    ocr_provider: OCRProvider = "none"

    openai_api_key: SecretStr = SecretStr("")
    openai_base_url: str = ""
    anthropic_api_key: SecretStr = SecretStr("")

    # --- external connectors (interfaces only; none connected by default) ----
    cmms_connector: str = "none"
    cmms_base_url: str = ""
    cmms_api_key: SecretStr = SecretStr("")
    s3_connector_enabled: bool = False
    s3_endpoint_url: str = ""
    s3_bucket: str = ""
    s3_access_key_id: SecretStr = SecretStr("")
    s3_secret_access_key: SecretStr = SecretStr("")
    sharepoint_connector_enabled: bool = False
    sharepoint_tenant_id: str = ""
    sharepoint_client_id: str = ""
    sharepoint_client_secret: SecretStr = SecretStr("")

    # --- ingestion -----------------------------------------------------------
    ingest_max_file_mb: int = 100
    ingest_allowed_extensions: str = ".pdf,.txt,.md,.csv,.json,.docx,.png,.jpg,.jpeg,.tif,.tiff"
    ingest_queue_name: str = "queue:ingest"
    ingest_worker_concurrency: int = 1
    ingest_visibility_timeout_s: int = 900

    # --- retrieval -----------------------------------------------------------
    retrieval_top_k_lexical: int = 50
    retrieval_top_k_dense: int = 50
    retrieval_top_k_graph: int = 50
    retrieval_rrf_k: int = 60
    retrieval_final_k: int = 8
    confidence_answer_threshold: float = 0.75
    confidence_caveat_threshold: float = 0.50

    # --- entity resolution ---------------------------------------------------
    er_auto_merge_threshold: float = Field(default=0.90, ge=0.0, le=1.0)
    er_review_threshold: float = Field(default=0.60, ge=0.0, le=1.0)

    @field_validator("ingest_allowed_extensions")
    @classmethod
    def _normalise_extensions(cls, v: str) -> str:
        return ",".join(e.strip().lower() for e in v.split(",") if e.strip())

    # --- derived -------------------------------------------------------------
    @property
    def allowed_extensions(self) -> set[str]:
        return {
            e if e.startswith(".") else f".{e}" for e in self.ingest_allowed_extensions.split(",")
        }

    @property
    def postgres_dsn(self) -> str:
        pwd = self.postgres_password.get_secret_value()
        return (
            f"postgresql://{self.postgres_user}:{pwd}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def max_upload_bytes(self) -> int:
        return self.ingest_max_file_mb * 1024 * 1024

    def provider_status(self) -> dict[str, dict[str, object]]:
        """Truthful provider report.

        This never claims a provider *works* -- only whether the configuration
        required to attempt it is present. Live reachability is probed by the
        health endpoint for infrastructure only.
        """

        def entry(provider: str, missing: list[str]) -> dict[str, object]:
            if provider == "none":
                state = "disabled"
            elif missing:
                state = "provider_not_configured"
            else:
                state = "configured"
            return {"provider": provider, "state": state, "missing_credentials": missing}

        emb_missing: list[str] = []
        if self.embedding_provider == "openai" and not self.openai_api_key.get_secret_value():
            emb_missing.append("OPENAI_API_KEY")

        llm_missing: list[str] = []
        if self.llm_provider == "openai" and not self.openai_api_key.get_secret_value():
            llm_missing.append("OPENAI_API_KEY")
        if self.llm_provider == "anthropic" and not self.anthropic_api_key.get_secret_value():
            llm_missing.append("ANTHROPIC_API_KEY")
        if self.llm_provider != "none" and not self.llm_model:
            llm_missing.append("LLM_MODEL")

        return {
            "embedding": entry(self.embedding_provider, emb_missing),
            "llm": entry(self.llm_provider, llm_missing),
            "reranker": entry(self.reranker_provider, []),
            "ocr": entry(self.ocr_provider, []),
        }

    def connector_status(self) -> dict[str, str]:
        return {
            "cmms": self.cmms_connector if self.cmms_connector != "none" else "not_configured",
            "s3": "configured" if self.s3_connector_enabled else "not_configured",
            "sharepoint": "configured" if self.sharepoint_connector_enabled else "not_configured",
            "filesystem_upload": "available",
        }

    def public_dict(self) -> dict[str, object]:
        """Configuration safe to expose over HTTP. Secrets are never included."""
        return {
            "app_env": self.app_env,
            "app_version": self.app_version,
            "embedding_dim": self.embedding_dim,
            "retrieval": {
                "top_k_lexical": self.retrieval_top_k_lexical,
                "top_k_dense": self.retrieval_top_k_dense,
                "top_k_graph": self.retrieval_top_k_graph,
                "rrf_k": self.retrieval_rrf_k,
                "final_k": self.retrieval_final_k,
                "answer_threshold": self.confidence_answer_threshold,
                "caveat_threshold": self.confidence_caveat_threshold,
            },
            "entity_resolution": {
                "auto_merge_threshold": self.er_auto_merge_threshold,
                "review_threshold": self.er_review_threshold,
            },
            "ingest": {
                "max_file_mb": self.ingest_max_file_mb,
                "allowed_extensions": sorted(self.allowed_extensions),
            },
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
