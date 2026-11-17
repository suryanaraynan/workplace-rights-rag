import json

import pytest

from workplace_rag.llm import LLMClient, LLMResponse
from workplace_rag.rag import NOT_FOUND_MESSAGE, RAGPipeline, build_messages, extract_citations
from workplace_rag.retrieval import RetrievedChunk, Retriever, build_index


def chunk(rank, score, body="SSP is £118.75 a week."):
    return RetrievedChunk(rank, score, f"c{rank}", "Statutory Sick Pay", "What you'll get",
                          "https://www.gov.uk/statutory-sick-pay", "sick_leave", body)


class FakeRetriever:
    def __init__(self, chunks):
        self.chunks, self.calls = chunks, []

    def search(self, q, k=5, topic=None):
        self.calls.append((q, k, topic))
        return self.chunks[:k]


class FakeLLM:
    def __init__(self, text):
        self.text, self.calls = text, 0

    def complete(self, messages):
        self.calls += 1
        return LLMResponse(self.text, "fake-model", 0.1, 100, 20)


# --- prompt & citations -------------------------------------------------
def test_sources_are_numbered_in_prompt():
    msgs = build_messages("How much SSP?", [chunk(1, 0.8), chunk(2, 0.7)])
    assert msgs[0]["role"] == "system" and "ONLY" in msgs[0]["content"]
    assert "[1] Statutory Sick Pay > What you'll get" in msgs[1]["content"]
    assert "[2]" in msgs[1]["content"] and msgs[1]["content"].endswith("Question: How much SSP?")


def test_extract_citations_handles_repeats_and_adjacent():
    assert extract_citations("A [2]. B [1][2]. C [10].") == [1, 2, 10]


# --- pipeline behaviour -------------------------------------------------
def test_answer_with_valid_citations(tmp_path):
    llm = FakeLLM("You get £118.75 a week [1].")
    p = RAGPipeline(FakeRetriever([chunk(1, 0.8), chunk(2, 0.6)]), llm, top_k=2,
                    log_path=tmp_path / "log.jsonl")
    r = p.answer("How much SSP?")
    assert r.cited == [1] and r.invalid_citations == [] and r.llm_called
    log = json.loads((tmp_path / "log.jsonl").read_text())
    assert log["question"] == "How much SSP?" and log["sources"][0]["chunk_id"] == "c1"


def test_hallucinated_citation_is_flagged():
    p = RAGPipeline(FakeRetriever([chunk(1, 0.8)]), FakeLLM("Something [4]."), top_k=1)
    assert p.answer("q").invalid_citations == [4]


def test_guardrail_skips_llm_when_nothing_relevant():
    llm = FakeLLM("should not be called")
    p = RAGPipeline(FakeRetriever([chunk(1, 0.12)]), llm, min_score=0.3)
    r = p.answer("What is the capital of France?")
    assert llm.calls == 0 and not r.llm_called and r.answer == NOT_FOUND_MESSAGE
    assert "0.12" in r.guardrail


def test_topic_filter_is_passed_to_retriever():
    ret = FakeRetriever([chunk(1, 0.9)])
    RAGPipeline(ret, FakeLLM("x [1]"), top_k=3).answer("q", topic="sick_leave")
    assert ret.calls == [("q", 3, "sick_leave")]


# --- index round trip (fake embedder, no model download) ------------------
class FakeEmbedder:
    model_name = "fake-embedder"
    vocab = ["sick", "holiday", "redundancy"]

    def _vec(self, text):
        v = [float(w in text.lower()) for w in self.vocab]
        return v if any(v) else [0.01, 0.01, 0.01]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


CHUNKS = [
    {"chunk_id": f"d#{i}", "doc_id": "d", "text": t, "body": t, "title": "T", "heading": "H",
     "url": "u", "topic": topic}
    for i, (t, topic) in enumerate([("sick pay rules", "sick_leave"),
                                    ("holiday entitlement", "holidays_and_time_off"),
                                    ("redundancy pay", "dismissal_and_redundancy")])
]


def test_index_round_trip_and_topic_filter(tmp_path):
    emb = FakeEmbedder()
    build_index(CHUNKS, emb, tmp_path, "test_chunks")
    r = Retriever(emb, tmp_path, "test_chunks")
    top = r.search("how much sick pay", k=1)[0]
    assert top.chunk_id == "d#0" and top.score > 0.9 and top.rank == 1
    filtered = r.search("sick", k=3, topic="holidays_and_time_off")
    assert [c.topic for c in filtered] == ["holidays_and_time_off"]


def test_index_built_with_other_model_is_rejected(tmp_path):
    build_index(CHUNKS, FakeEmbedder(), tmp_path, "test_chunks")
    other = FakeEmbedder()
    other.model_name = "different-model"
    with pytest.raises(ValueError, match="build_index"):
        Retriever(other, tmp_path, "test_chunks")


# --- LLM config -------------------------------------------------------------
def test_missing_api_key_gives_clear_error(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    cfg = {"provider": "groq", "providers": {"groq": {
        "base_url": "https://api.groq.com/openai/v1", "api_key_env": "GROQ_API_KEY", "model": "m"}}}
    with pytest.raises(RuntimeError, match=".env"):
        LLMClient.from_config(cfg)


def test_ollama_needs_no_key():
    cfg = {"provider": "ollama", "providers": {"ollama": {
        "base_url": "http://localhost:11434/v1", "api_key_env": "", "model": "llama3.2:3b"}}}
    assert LLMClient.from_config(cfg).model == "llama3.2:3b"
