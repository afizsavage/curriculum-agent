"""V2.13A trusted document acquisition and local cache."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import httpx

from app.agent.v213_document_contract import DocumentRecord, DocumentStatus, document_id_for_source
from app.logging_utils import get_logger

logger = get_logger(__name__)

_ALLOWED_CONTENT_TYPES = frozenset(
    {
        "application/pdf",
        "text/plain",
        "application/octet-stream",
    }
)

DownloadFn = Callable[[str], tuple[bytes, str]]


class UntrustedSourceError(ValueError):
    """Raised when a source fails trust-boundary validation."""


class DocumentHashConflictError(ValueError):
    """Raised when authoritative document content changes without a version bump."""


class DocumentStore:
    """Deterministic local cache for registered curriculum source documents."""

    def __init__(self, *, root: Path | None = None, download_fn: DownloadFn | None = None) -> None:
        self.root = root or Path("data/documents")
        self._download_fn = download_fn or self._default_download

    def document_dir(self, document_id: str) -> Path:
        return self.root / document_id

    def source_file(self, document_id: str) -> Path:
        meta = self.load_metadata(document_id)
        suffix = ".pdf" if meta and "pdf" in (meta.get("content_type") or "") else ".bin"
        if meta and meta.get("content_type") == "text/plain":
            suffix = ".txt"
        return self.document_dir(document_id) / f"source{suffix}"

    def metadata_path(self, document_id: str) -> Path:
        return self.document_dir(document_id) / "metadata.json"

    def passages_path(self, document_id: str) -> Path:
        return self.document_dir(document_id) / "passages.json"

    @staticmethod
    def content_hash(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    @staticmethod
    def validate_trusted_source(source_record: dict[str, Any]) -> str:
        """Validate a CurriculumSource API record before acquisition."""
        source_id = str(source_record.get("id") or "")
        if not source_id:
            raise UntrustedSourceError("source record missing id")
        url = (source_record.get("document_url") or "").strip()
        if not url:
            raise UntrustedSourceError(f"source {source_id} has no document_url")
        if not url.startswith(("http://", "https://")):
            raise UntrustedSourceError(f"source {source_id} document_url is not http(s)")
        verification = str(source_record.get("verification_status") or "").upper()
        if verification in {"DRAFT", "SUPERSEDED"}:
            raise UntrustedSourceError(
                f"source {source_id} verification_status={verification} is not acquirable"
            )
        return url

    def acquire(
        self,
        source_record: dict[str, Any],
        *,
        curriculum_id: str | None = None,
        allow_local_path: str | None = None,
    ) -> DocumentRecord:
        """Download or copy a document from a registered curriculum source."""
        trusted_url = self.validate_trusted_source(source_record)
        source_id = str(source_record["id"])
        document_version = source_record.get("version")
        document_id = document_id_for_source(
            source_id=source_id, document_version=document_version
        )
        doc_dir = self.document_dir(document_id)
        doc_dir.mkdir(parents=True, exist_ok=True)

        if allow_local_path:
            data = Path(allow_local_path).read_bytes()
            content_type = "text/plain" if allow_local_path.endswith(".txt") else "application/octet-stream"
        else:
            data, content_type = self._download_fn(trusted_url)
            if trusted_url != (source_record.get("document_url") or "").strip():
                raise UntrustedSourceError("download URL does not match registered document_url")

        if content_type not in _ALLOWED_CONTENT_TYPES:
            raise UntrustedSourceError(f"unsupported content type: {content_type}")

        new_hash = self.content_hash(data)
        existing = self.load_metadata(document_id)
        if existing and existing.get("content_hash") and existing["content_hash"] != new_hash:
            if existing.get("source_url") == trusted_url and existing.get("document_version") == document_version:
                raise DocumentHashConflictError(
                    f"document {document_id} content changed without version bump"
                )

        dest = self.source_file(document_id)
        if content_type == "text/plain":
            dest = doc_dir / "source.txt"
        elif "pdf" in content_type:
            dest = doc_dir / "source.pdf"
        else:
            dest = doc_dir / "source.bin"
        dest.write_bytes(data)

        retrieved_at = datetime.now(timezone.utc).isoformat()
        record = DocumentRecord(
            document_id=document_id,
            source_id=source_id,
            source_url=trusted_url,
            document_version=document_version,
            content_hash=new_hash,
            content_type=content_type,
            file_size=len(data),
            retrieved_at=retrieved_at,
            status=DocumentStatus.ACQUIRED,
            curriculum_id=curriculum_id,
            metadata={
                "source_name": source_record.get("name"),
                "authority": source_record.get("authority"),
                "verification_status": source_record.get("verification_status"),
                "document_name": source_record.get("document_name"),
            },
        )
        self.save_metadata(document_id, record.to_dict())
        return record

    def save_metadata(self, document_id: str, payload: dict[str, Any]) -> None:
        self.metadata_path(document_id).write_text(json.dumps(payload, indent=2))

    def load_metadata(self, document_id: str) -> dict[str, Any] | None:
        path = self.metadata_path(document_id)
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def load_record(self, document_id: str) -> DocumentRecord | None:
        raw = self.load_metadata(document_id)
        if not raw:
            return None
        return DocumentRecord(
            document_id=raw["document_id"],
            source_id=raw["source_id"],
            source_url=raw["source_url"],
            document_version=raw.get("document_version"),
            content_hash=raw["content_hash"],
            content_type=raw["content_type"],
            file_size=int(raw.get("file_size") or 0),
            retrieved_at=raw["retrieved_at"],
            status=DocumentStatus(raw.get("status", DocumentStatus.ACQUIRED.value)),
            curriculum_id=raw.get("curriculum_id"),
            page_count=int(raw.get("page_count") or 0),
            passage_count=int(raw.get("passage_count") or 0),
            metadata=dict(raw.get("metadata") or {}),
        )

    def mark_parsed(
        self,
        document_id: str,
        *,
        page_count: int,
        passage_count: int,
    ) -> None:
        raw = self.load_metadata(document_id)
        if not raw:
            raise FileNotFoundError(f"no metadata for document {document_id}")
        raw["status"] = DocumentStatus.PARSED.value
        raw["page_count"] = page_count
        raw["passage_count"] = passage_count
        self.save_metadata(document_id, raw)

    def _default_download(self, url: str) -> tuple[bytes, str]:
        try:
            with httpx.Client(timeout=30.0, follow_redirects=True) as client:
                response = client.get(url)
                response.raise_for_status()
                content_type = response.headers.get("content-type", "application/octet-stream")
                content_type = content_type.split(";")[0].strip().lower()
                return response.content, content_type
        except httpx.HTTPError as exc:
            logger.warning("document_store.download_failed url=%r error=%r", url, str(exc))
            raise UntrustedSourceError(f"document download failed: {exc}") from exc


__all__ = [
    "DocumentHashConflictError",
    "DocumentStore",
    "UntrustedSourceError",
]
