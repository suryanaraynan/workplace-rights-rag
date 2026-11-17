"""Heading-aware chunking with overlap and contextual headers.

Three decisions, each worth defending in an interview:
1. Chunk *within* sections, never across them, so a chunk is about one thing.
2. Sliding window (max_words, overlap_words) for long sections, so a fact
   sitting on a boundary still appears whole in at least one chunk.
3. Prefix every chunk with "Page title > Section heading". A bare paragraph
   like "You must give 4 weeks' notice" is ambiguous; with its header the
   embedding knows it's about flexible working, which improves retrieval.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

from .ingest.parse import Document


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str          # what gets embedded (header + body)
    body: str          # body only, for display
    title: str
    heading: str
    url: str
    topics: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def _windows(words: list[str], max_words: int, overlap: int) -> list[list[str]]:
    if overlap >= max_words:
        raise ValueError("overlap_words must be smaller than max_words")
    if len(words) <= max_words:
        return [words]
    step = max_words - overlap
    out = []
    for start in range(0, len(words), step):
        out.append(words[start:start + max_words])
        if start + max_words >= len(words):
            break
    return out


def chunk_document(doc: Document, max_words: int = 200, overlap_words: int = 40) -> list[Chunk]:
    chunks: list[Chunk] = []
    for section in doc.sections:
        header = doc.title if section.heading == doc.title else f"{doc.title} > {section.heading}"
        for window in _windows(section.text.split(), max_words, overlap_words):
            body = " ".join(window)
            chunks.append(Chunk(
                chunk_id=f"{doc.doc_id}#{len(chunks)}",
                doc_id=doc.doc_id,
                text=f"{header}\n{body}",
                body=body,
                title=doc.title,
                heading=section.heading,
                url=section.url,
                topics=doc.topics,
            ))
    return chunks
