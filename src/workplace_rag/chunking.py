"""Heading-aware chunking: merge small sections, window large ones, add context.

Four decisions, each worth defending in an interview:
1. Chunk *within* a page part (one URL), never across, so citations stay exact.
2. Merge neighbouring sections smaller than `min_words`. Real GOV.UK pages
   have many 7-15 word subsections ("Job sharing", "Part time") that are
   useless alone; together they form one coherent chunk.
3. Sliding window (max_words, overlap_words) for long sections, so a fact
   sitting on a boundary still appears whole in at least one chunk.
4. Prefix every chunk with "Page title > Section heading". A bare paragraph
   like "You must give 4 weeks' notice" is ambiguous; with its header the
   embedding knows it's about flexible working, which improves retrieval.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

from .ingest.parse import Document, Section


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str          # what gets embedded (header + body)
    body: str          # body only, for display
    title: str
    heading: str
    url: str
    topic: str         # curated label
    topics: list[str]  # GOV.UK tags

    def to_dict(self) -> dict:
        return asdict(self)


def _words(s: Section) -> int:
    return len(s.text.split())


def merge_small_sections(sections: list[Section], min_words: int, max_words: int) -> list[Section]:
    """Greedily merge a too-small section with the following ones on the same
    URL, as long as the result stays within max_words. Merged-in sections keep
    their heading inline ("Job sharing: ...") so no meaning is lost."""
    merged: list[Section] = []
    for s in sections:
        prev = merged[-1] if merged else None
        if (prev is not None and prev.url == s.url and _words(prev) < min_words
                and _words(prev) + _words(s) <= max_words):
            # Shared parent heading becomes the merged heading (e.g. "Types of flexible working").
            same_parent = s.parent and s.parent in (prev.parent, prev.heading)
            heading = (prev.parent or prev.heading) if same_parent else prev.heading_path
            merged[-1] = Section(heading=heading,
                                 text=f"{prev.text} {s.heading}: {s.text}",
                                 url=prev.url)
        else:
            merged.append(Section(s.heading, s.text, s.url, s.parent))
    return merged


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


def _header(doc: Document, section: Section) -> str:
    parts = [p for p in (section.parent, section.heading) if p and p != doc.title]
    return " > ".join([doc.title, *parts])


def chunk_document(doc: Document, max_words: int = 200, overlap_words: int = 40,
                   min_words: int = 0) -> list[Chunk]:
    sections = merge_small_sections(doc.sections, min_words, max_words) if min_words else doc.sections
    chunks: list[Chunk] = []
    for section in sections:
        header = _header(doc, section)
        for window in _windows(section.text.split(), max_words, overlap_words):
            body = " ".join(window)
            chunks.append(Chunk(
                chunk_id=f"{doc.doc_id}#{len(chunks)}",
                doc_id=doc.doc_id,
                text=f"{header}\n{body}",
                body=body,
                title=doc.title,
                heading=section.heading_path,
                url=section.url,
                topic=doc.topic,
                topics=doc.topics,
            ))
    return chunks
