"""V2.13A PDF/text document parser with page boundaries."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class ParsedBlock:
    block_id: str
    text: str
    block_type: str = "paragraph"


@dataclass
class ParsedPage:
    page_number: int
    text: str
    blocks: list[ParsedBlock] = field(default_factory=list)
    headings: list[str] = field(default_factory=list)


@dataclass
class ParsedDocument:
    pages: list[ParsedPage]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def page_count(self) -> int:
        return len(self.pages)

    def to_dict(self) -> dict[str, Any]:
        return {
            "page_count": self.page_count,
            "pages": [
                {
                    "page_number": p.page_number,
                    "text": p.text,
                    "headings": list(p.headings),
                    "blocks": [
                        {"block_id": b.block_id, "text": b.text, "block_type": b.block_type}
                        for b in p.blocks
                    ],
                }
                for p in self.pages
            ],
            "metadata": dict(self.metadata),
        }


class DocumentParser:
    """Extract page-level text from PDF or plain-text curriculum documents."""

    def parse_file(self, path: Path, *, content_type: str | None = None) -> ParsedDocument:
        suffix = path.suffix.lower()
        ctype = (content_type or "").lower()
        if suffix == ".pdf" or "pdf" in ctype:
            return self.parse_pdf(path)
        return self.parse_text(path)

    def parse_pdf(self, path: Path) -> ParsedDocument:
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise RuntimeError(
                "pypdf is required for PDF parsing; install with pip install pypdf"
            ) from exc

        reader = PdfReader(str(path))
        pages: list[ParsedPage] = []
        for index, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            blocks = self._blocks_from_text(text, page_number=index)
            headings = [b.text for b in blocks if b.block_type == "heading"]
            pages.append(
                ParsedPage(
                    page_number=index,
                    text=text,
                    blocks=blocks,
                    headings=headings,
                )
            )
        return ParsedDocument(
            pages=pages,
            metadata={"parser": "pypdf", "source_path": str(path)},
        )

    def parse_text(self, path: Path) -> ParsedDocument:
        raw = path.read_text(encoding="utf-8", errors="replace")
        pages: list[ParsedPage] = []
        chunks = raw.split("\f")
        if len(chunks) == 1:
            chunks = self._split_pages_by_marker(raw)
        for index, chunk in enumerate(chunks, start=1):
            text = chunk.strip()
            if not text:
                continue
            blocks = self._blocks_from_text(text, page_number=index)
            headings = [b.text for b in blocks if b.block_type == "heading"]
            pages.append(
                ParsedPage(
                    page_number=index,
                    text=text,
                    blocks=blocks,
                    headings=headings,
                )
            )
        return ParsedDocument(
            pages=pages,
            metadata={"parser": "text", "source_path": str(path)},
        )

    @staticmethod
    def _split_pages_by_marker(raw: str) -> list[str]:
        marker = "--- PAGE "
        if marker not in raw:
            return [raw]
        parts: list[str] = []
        current: list[str] = []
        for line in raw.splitlines(keepends=True):
            if line.startswith(marker):
                if current:
                    parts.append("".join(current))
                    current = []
                continue
            current.append(line)
        if current:
            parts.append("".join(current))
        return parts or [raw]

    def _blocks_from_text(self, text: str, *, page_number: int) -> list[ParsedBlock]:
        blocks: list[ParsedBlock] = []
        paragraph: list[str] = []
        block_index = 0

        def flush_paragraph() -> None:
            nonlocal block_index
            joined = " ".join(paragraph).strip()
            paragraph.clear()
            if not joined:
                return
            blocks.append(
                ParsedBlock(
                    block_id=f"p{page_number}-b{block_index}",
                    text=joined,
                    block_type="paragraph",
                )
            )
            block_index += 1

        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                flush_paragraph()
                continue
            if self._looks_like_heading(stripped):
                flush_paragraph()
                blocks.append(
                    ParsedBlock(
                        block_id=f"p{page_number}-h{block_index}",
                        text=stripped,
                        block_type="heading",
                    )
                )
                block_index += 1
                continue
            paragraph.append(stripped)
        flush_paragraph()
        return blocks

    @staticmethod
    def _looks_like_heading(line: str) -> bool:
        if len(line) > 120:
            return False
        if line.isupper() and len(line.split()) <= 12:
            return True
        if line.endswith(":") and len(line.split()) <= 10:
            return True
        for prefix in ("Section ", "Chapter ", "Unit ", "Topic ", "Mathematics", "Science"):
            if line.startswith(prefix):
                return True
        return False


__all__ = ["DocumentParser", "ParsedBlock", "ParsedDocument", "ParsedPage"]
