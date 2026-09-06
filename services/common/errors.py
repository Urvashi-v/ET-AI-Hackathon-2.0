"""Typed application errors and their HTTP mapping.

Two rules encoded here:

1. **Safe error responses.** Client-facing payloads carry a stable machine code,
   a human message and an optional detail dict. Stack traces, DSNs, driver
   messages and file paths never cross the boundary.
2. **Truthful degradation.** ``CapabilityUnavailable`` is not a failure -- it is
   the honest answer when a stage cannot run because a provider or credential is
   absent. It carries the exact environment variables the operator must supply.
"""

from __future__ import annotations

from typing import Any


class BrainError(Exception):
    """Base class. ``code`` is a stable identifier clients may branch on."""

    status_code: int = 500
    code: str = "internal_error"
    public_message: str = "An internal error occurred."

    def __init__(
        self,
        message: str | None = None,
        *,
        detail: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message or self.public_message)
        self.message = message or self.public_message
        self.detail = detail or {}

    def to_payload(self) -> dict[str, Any]:
        return {
            "error": {
                "code": self.code,
                "message": self.message,
                "detail": self.detail,
            }
        }


class ValidationError(BrainError):
    status_code = 400
    code = "validation_error"
    public_message = "The request was invalid."


class FileValidationError(ValidationError):
    code = "file_validation_error"
    public_message = "The uploaded file was rejected."


class NotFoundError(BrainError):
    status_code = 404
    code = "not_found"
    public_message = "The requested resource does not exist."


class ConflictError(BrainError):
    status_code = 409
    code = "conflict"
    public_message = "The request conflicts with the current state."


class DependencyUnavailable(BrainError):
    """An infrastructure dependency (Postgres/Neo4j/Redis) is unreachable."""

    status_code = 503
    code = "dependency_unavailable"
    public_message = "A required backing service is unavailable."


class CapabilityUnavailable(BrainError):
    """A capability cannot run because it is not configured.

    Returned with HTTP 200 inside a structured status field where the endpoint
    can still deliver partial real results (e.g. retrieval evidence without
    generation), and with 503 where nothing useful can be produced.
    """

    status_code = 503
    code = "capability_not_configured"
    public_message = "This capability requires configuration that is not present."

    def __init__(
        self,
        capability: str,
        *,
        required_env: list[str] | None = None,
        message: str | None = None,
        hint: str | None = None,
    ) -> None:
        detail: dict[str, Any] = {
            "capability": capability,
            "required_env": required_env or [],
        }
        if hint:
            detail["hint"] = hint
        super().__init__(
            message
            or (
                f"Capability '{capability}' is not configured. "
                f"Set: {', '.join(required_env or []) or '(see .env.example)'}"
            ),
            detail=detail,
        )
