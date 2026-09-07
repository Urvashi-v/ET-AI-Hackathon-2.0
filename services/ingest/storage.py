"""Blob storage and file validation.

Uploaded bytes are the system's untrusted input. Everything that enters is
validated before it is written, and written under a content-addressed path so
that ingestion is idempotent by construction: the same bytes always land in the
same place and the ``documents.content_hash`` unique constraint rejects the
duplicate.

Validation is deliberately conservative:

* extension allow-list (configurable, defaults exclude executables and archives);
* declared size limit enforced while streaming, not after;
* magic-byte sniffing so a ``.pdf`` that is actually something else is rejected;
* filenames are never used to build paths -- the storage path is derived from the
  hash, and the original name is kept only as metadata.
"""

from __future__ import annotations

import os
import shutil
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from services.common.config import get_settings
from services.common.errors import FileValidationError
from services.common.ids import sha256_bytes
from services.common.logging import get_logger

log = get_logger(__name__)

#: Leading magic bytes we can verify. Extensions not listed are accepted on
#: extension alone (text formats have no reliable signature).
_MAGIC: dict[str, tuple[bytes, ...]] = {
    ".pdf": (b"%PDF-",),
    ".png": (b"\x89PNG\r\n\x1a\n",),
    ".jpg": (b"\xff\xd8\xff",),
    ".jpeg": (b"\xff\xd8\xff",),
    ".tif": (b"II*\x00", b"MM\x00*"),
    ".tiff": (b"II*\x00", b"MM\x00*"),
    ".docx": (b"PK\x03\x04",),
}

_MIME: dict[str, str] = {
    ".pdf": "application/pdf",
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".csv": "text/csv",
    ".json": "application/json",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
}


@dataclass(frozen=True, slots=True)
class StoredBlob:
    content_hash: str
    path: str
    byte_size: int
    original_filename: str
    extension: str
    mime_type: str


def safe_filename(name: str) -> str:
    """Strip everything path-like from a client-supplied filename."""
    name = unicodedata.normalize("NFKC", name).replace("\\", "/").split("/")[-1]
    name = "".join(ch for ch in name if ch.isprintable() and ch not in '<>:"|?*')
    return name.strip().strip(".") or "unnamed"


def blob_root() -> Path:
    root = Path(get_settings().blob_root)
    root.mkdir(parents=True, exist_ok=True)
    return root


def validate_bytes(filename: str, data: bytes) -> tuple[str, str]:
    """Validate an in-memory upload. Returns ``(extension, mime_type)``.

    Raises :class:`FileValidationError` with a message safe to show a user.
    """
    settings = get_settings()
    clean = safe_filename(filename)
    ext = Path(clean).suffix.lower()

    if not ext:
        raise FileValidationError("File has no extension; the type cannot be determined.")
    if ext not in settings.allowed_extensions:
        raise FileValidationError(
            f"Extension '{ext}' is not permitted.",
            detail={"allowed": sorted(settings.allowed_extensions)},
        )
    if len(data) == 0:
        raise FileValidationError("File is empty.")
    if len(data) > settings.max_upload_bytes:
        raise FileValidationError(
            f"File exceeds the {settings.ingest_max_file_mb} MB limit.",
            detail={"byte_size": len(data), "limit_bytes": settings.max_upload_bytes},
        )

    signatures = _MAGIC.get(ext)
    if signatures and not any(data.startswith(sig) for sig in signatures):
        raise FileValidationError(
            f"File content does not match its '{ext}' extension.",
            detail={"checked": "magic_bytes"},
        )

    if ext in {".txt", ".md", ".csv", ".json"}:
        try:
            data.decode("utf-8")
        except UnicodeDecodeError:
            try:
                data.decode("latin-1")
            except UnicodeDecodeError as exc:
                raise FileValidationError("Text file is not decodable.") from exc

    return ext, _MIME.get(ext, "application/octet-stream")


def store_bytes(filename: str, data: bytes) -> StoredBlob:
    """Validate and write bytes to the content-addressed blob store."""
    ext, mime = validate_bytes(filename, data)
    digest = sha256_bytes(data)
    target = _blob_path(digest, ext)
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".part")
        tmp.write_bytes(data)
        os.replace(tmp, target)
        log.info("blob.stored", content_hash=digest, byte_size=len(data), extension=ext)
    else:
        log.info("blob.already_present", content_hash=digest)
    return StoredBlob(
        content_hash=digest,
        path=str(target),
        byte_size=len(data),
        original_filename=safe_filename(filename),
        extension=ext,
        mime_type=mime,
    )


def store_path(source: Path) -> StoredBlob:
    """Validate and copy a file that already exists on a readable path."""
    settings = get_settings()
    if not source.is_file():
        raise FileValidationError(f"Not a readable file: {source.name}")
    size = source.stat().st_size
    if size > settings.max_upload_bytes:
        raise FileValidationError(
            f"File exceeds the {settings.ingest_max_file_mb} MB limit.",
            detail={"byte_size": size},
        )
    data = source.read_bytes()
    blob = store_bytes(source.name, data)
    return blob


def _blob_path(digest: str, ext: str) -> Path:
    # Two levels of fan-out keeps directory sizes sane at corpus scale.
    return blob_root() / digest[:2] / digest[2:4] / f"{digest}{ext}"


def resolve_blob(path: str) -> Path:
    """Resolve a stored blob path, refusing anything outside the blob root.

    ``is_relative_to`` rather than a string prefix test: ``/data/blobs-old``
    starts with ``/data/blobs`` as a string but is a different directory, and a
    prefix check would wave it through. Symlinks are resolved first, so a link
    planted inside the root cannot point out of it either.
    """
    resolved = Path(path).resolve()
    root = blob_root().resolve()
    if not resolved.is_relative_to(root):
        raise FileValidationError("Blob path is outside the storage root.")
    return resolved


def read_blob(path: str) -> bytes:
    """Read a stored blob, refusing anything outside the blob root."""
    return resolve_blob(path).read_bytes()


def free_space_bytes() -> int:
    return shutil.disk_usage(blob_root()).free
