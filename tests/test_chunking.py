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
