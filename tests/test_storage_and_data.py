"""File validation, content-addressed storage, and corpus determinism.

Uploaded bytes are untrusted input and the blob store is the boundary that
handles them. The synthetic corpus is a build artefact and has to be
reproducible, or the golden question set stops being valid.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from services.common.errors import FileValidationError
from services.common.ids import chunk_id, document_id, mention_id, sha256_bytes
from services.ingest import storage

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _isolated_blob_root(tmp_path, monkeypatch):
    """Point the blob store at a temp directory for the whole module."""
    from services.common.config import get_settings

    settings = get_settings()
    monkeypatch.setattr(settings, "blob_root", str(tmp_path / "blobs"))
    yield


class TestFileValidation:
    def test_accepts_a_permitted_text_file(self):
        ext, mime = storage.validate_bytes("notes.md", b"# Work order notes\n")
        assert ext == ".md" and mime == "text/markdown"

    def test_rejects_a_disallowed_extension(self):
        with pytest.raises(FileValidationError, match="not permitted"):
            storage.validate_bytes("payload.exe", b"MZ\x90\x00")

    def test_rejects_a_file_with_no_extension(self):
        with pytest.raises(FileValidationError, match="no extension"):
            storage.validate_bytes("README", b"content")

    def test_rejects_an_empty_file(self):
        with pytest.raises(FileValidationError, match="empty"):
            storage.validate_bytes("empty.txt", b"")

    def test_rejects_content_that_contradicts_its_extension(self):
        # A .pdf that is not a PDF. Extension alone is not evidence.
        with pytest.raises(FileValidationError, match="does not match"):
            storage.validate_bytes("report.pdf", b"this is plain text, not a PDF")

    def test_accepts_a_real_pdf_signature(self):
        ext, _ = storage.validate_bytes("report.pdf", b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n")
        assert ext == ".pdf"

    def test_rejects_oversized_input(self, monkeypatch):
        from services.common.config import get_settings

        monkeypatch.setattr(get_settings(), "ingest_max_file_mb", 1)
        with pytest.raises(FileValidationError, match="exceeds"):
            storage.validate_bytes("big.txt", b"x" * (2 * 1024 * 1024))

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("../../etc/passwd", "passwd"),
            ("..\\..\\windows\\system32\\config", "config"),
            ("/absolute/path/report.pdf", "report.pdf"),
            ("  spaced .txt  ", "spaced .txt"),
            ("bad<>:name.txt", "badname.txt"),
        ],
    )
    def test_filenames_are_stripped_of_everything_path_like(self, raw, expected):
        assert storage.safe_filename(raw) == expected

    def test_a_filename_is_never_used_to_build_the_storage_path(self):
        blob = storage.store_bytes("../../escape.txt", b"content")
        root = storage.blob_root().resolve()
        assert str(Path(blob.path).resolve()).startswith(str(root))
        assert "escape" not in Path(blob.path).name


class TestContentAddressedStorage:
    def test_identical_bytes_land_at_one_path(self):
        first = storage.store_bytes("a.txt", b"same content")
        second = storage.store_bytes("b.txt", b"same content")
        assert first.content_hash == second.content_hash
        assert first.path == second.path

    def test_different_bytes_land_at_different_paths(self):
        assert (
            storage.store_bytes("a.txt", b"one").path != storage.store_bytes("a.txt", b"two").path
        )

    def test_reading_outside_the_blob_root_is_refused(self):
        with pytest.raises(FileValidationError, match="outside"):
            storage.read_blob(str(REPO_ROOT / "pyproject.toml"))

    def test_round_trip(self):
        blob = storage.store_bytes("note.txt", b"round trip content")
        assert storage.read_blob(blob.path) == b"round trip content"


class TestDeterministicIdentifiers:
    def test_document_id_is_derived_from_content(self):
        digest = sha256_bytes(b"the same bytes")
        assert document_id(digest) == document_id(digest)

    def test_different_content_yields_a_different_document_id(self):
        assert document_id(sha256_bytes(b"a")) != document_id(sha256_bytes(b"b"))

    def test_chunk_and_mention_ids_are_stable(self):
        # Idempotent re-ingest depends on this: same input, same keys, so every
        # write is an upsert rather than a duplicate.
        assert chunk_id("doc_x", 3) == chunk_id("doc_x", 3)
        assert chunk_id("doc_x", 3) != chunk_id("doc_x", 4)
        assert mention_id("chk_1", "P-101B", 10) == mention_id("chk_1", "P-101B", 10)
        assert mention_id("chk_1", "P-101B", 10) != mention_id("chk_1", "P-101B", 11)


class TestSyntheticCorpus:
    """The corpus is a build artefact. If it is not reproducible, the golden
    question set silently stops matching the data it was written against."""

    GENERATOR = REPO_ROOT / "data" / "synthetic" / "generate.py"

    def _generate(self, out_dir: Path, seed: int = 20260101) -> dict:
        subprocess.run(
            [sys.executable, str(self.GENERATOR), "--out", str(out_dir), "--seed", str(seed)],
            check=True,
            capture_output=True,
            cwd=REPO_ROOT,
        )
        return json.loads((out_dir / "MANIFEST.json").read_text(encoding="utf-8"))

    def test_same_seed_produces_byte_identical_output(self, tmp_path):
        first = self._generate(tmp_path / "a")
        second = self._generate(tmp_path / "b")
        assert first["files"] == second["files"]

    def test_a_different_seed_changes_the_surface_forms(self, tmp_path):
        default = self._generate(tmp_path / "a")
        other = self._generate(tmp_path / "c", seed=99)
        assert default["files"] != other["files"]

    def test_manifest_hashes_match_the_files_on_disk(self, tmp_path):
        manifest = self._generate(tmp_path / "a")
        for entry in manifest["files"]:
            actual = hashlib.sha256((tmp_path / "a" / entry["filename"]).read_bytes()).hexdigest()
            assert actual == entry["sha256"]

    def test_output_is_labelled_synthetic_in_the_manifest(self, tmp_path):
        manifest = self._generate(tmp_path / "a")
        assert manifest["data_class"] == "synthetic_test_data"
        assert "not a real plant" in manifest["warning"].lower()

    def test_every_generated_file_carries_the_banner_in_its_own_content(self, tmp_path):
        self._generate(tmp_path / "a")
        for path in (tmp_path / "a").iterdir():
            if path.name == "MANIFEST.json":
                continue
            assert "SYNTHETIC TEST DATA" in path.read_text(encoding="utf-8")

    def test_the_corpus_contains_the_six_tag_spellings(self, tmp_path):
        self._generate(tmp_path / "a")
        text = (tmp_path / "a" / "work_orders_cmms_export.csv").read_text(encoding="utf-8")
        from services.common.tags import parse

        spellings = {line.split(",")[2] for line in text.splitlines()[2:] if line.count(",") > 3}
        resolved = {parse(s).canonical for s in spellings if parse(s).parsed}
        assert "P-101B" in resolved
        assert len(spellings) > 3, "the export must contain genuine tag variance"

    def test_siblings_both_appear_so_a_merge_bug_would_be_visible(self, tmp_path):
        self._generate(tmp_path / "a")
        text = (tmp_path / "a" / "work_orders_cmms_export.csv").read_text(encoding="utf-8")
        from services.common.tags import parse

        canonical = {
            parse(line.split(",")[2]).canonical
            for line in text.splitlines()[2:]
            if line.count(",") > 3 and parse(line.split(",")[2]).parsed
        }
        assert {"P-101A", "P-101B"} <= canonical


class TestRequirementProvenance:
    """No requirement may enter the system without traceable provenance."""

    FILE = REPO_ROOT / "data" / "requirements" / "atomised_requirements.json"

    def test_every_requirement_declares_its_text_status_and_provenance(self):
        payload = json.loads(self.FILE.read_text(encoding="utf-8"))
        for entry in payload["requirements"]:
            assert entry["text_status"] in {"verbatim", "paraphrase_for_demo"}
            assert entry["provenance_note"].strip()

    def test_nothing_shipped_claims_to_be_verbatim_standard_text(self):
        payload = json.loads(self.FILE.read_text(encoding="utf-8"))
        statuses = {e["text_status"] for e in payload["requirements"]}
        assert statuses == {"paraphrase_for_demo"}, (
            "Shipping a 'verbatim' requirement requires a real citation in "
            "provenance_note; see data/requirements/README.md"
        )

    def test_modality_is_never_flattened(self):
        payload = json.loads(self.FILE.read_text(encoding="utf-8"))
        for entry in payload["requirements"]:
            assert entry["modality"] in {"shall", "should", "may"}

    def test_the_loader_rejects_a_requirement_without_provenance(self):
        sys.path.insert(0, str(REPO_ROOT / "scripts"))
        from load_requirements import validate

        problems = validate(
            {
                "req_id": "X-1",
                "source_standard": "S",
                "clause": "1",
                "obligation_text": "t",
                "modality": "shall",
                "text_status": "verbatim",
            },
            0,
        )
        assert any("provenance_note" in p for p in problems)

    def test_the_loader_demands_a_citation_for_verbatim_text(self):
        sys.path.insert(0, str(REPO_ROOT / "scripts"))
        from load_requirements import validate

        problems = validate(
            {
                "req_id": "X-1",
                "source_standard": "S",
                "clause": "1",
                "obligation_text": "t",
                "modality": "shall",
                "text_status": "verbatim",
                "provenance_note": "from the standard",
            },
            0,
        )
        assert any("Verbatim requires a citation" in p for p in problems)
