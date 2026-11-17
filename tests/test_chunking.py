import pytest

from workplace_rag.chunking import _windows, chunk_document
from workplace_rag.ingest.parse import Document, Section


def make_doc(n_words):
    text = " ".join(f"w{i}" for i in range(n_words))
    return Document("d", "u", "Flexible working", "", "answer", "",
                    topics=["Employment"], sections=[Section("Making a request", text, "u#s")])


def test_short_section_is_one_chunk_with_context_header():
    chunks = chunk_document(make_doc(50), max_words=200, overlap_words=40)
    assert len(chunks) == 1
    assert chunks[0].text.startswith("Flexible working > Making a request\n")
    assert chunks[0].url == "u#s" and chunks[0].topics == ["Employment"]


def test_long_section_overlaps_and_covers_every_word():
    chunks = chunk_document(make_doc(500), max_words=200, overlap_words=40)
    covered = {w for c in chunks for w in c.body.split()}
    assert covered == {f"w{i}" for i in range(500)}
    first, second = chunks[0].body.split(), chunks[1].body.split()
    assert first[-40:] == second[:40]
    assert all(len(c.body.split()) <= 200 for c in chunks)


def test_chunk_ids_are_unique():
    chunks = chunk_document(make_doc(1000))
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_invalid_overlap_rejected():
    with pytest.raises(ValueError):
        _windows(["a"] * 10, max_words=5, overlap=5)


def test_small_sibling_sections_merge_under_parent_heading():
    doc = Document("d", "u", "Flexible working", "", "guide", "", sections=[
        Section("Types of flexible working", "There are different ways.", "u"),
        Section("Job sharing", "Two people split one job.", "u", parent="Types of flexible working"),
        Section("Part time", "Fewer hours.", "u", parent="Types of flexible working"),
    ])
    chunks = chunk_document(doc, max_words=200, overlap_words=40, min_words=50)
    assert len(chunks) == 1
    assert chunks[0].text.startswith("Flexible working > Types of flexible working\n")
    assert "Job sharing: Two people" in chunks[0].body and "Part time: Fewer" in chunks[0].body


def test_sections_on_different_urls_never_merge():
    doc = Document("d", "u", "T", "", "guide", "", sections=[
        Section("A", "short", "u/a"), Section("B", "short", "u/b")])
    assert len(chunk_document(doc, min_words=50)) == 2  # citations stay exact
