"""The RAG pipeline: retrieve -> guardrail -> prompt -> generate -> check citations -> log.

Every intermediate result is returned in RAGResult, so the UI can show
*why* an answer was given: which chunks were retrieved, their scores, the
exact prompt sent, and whether the citations in the answer are valid.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .retrieval import RetrievedChunk

SYSTEM_PROMPT = """You are an assistant that helps UK line managers understand employee rights, using ONLY the official GOV.UK guidance provided in the sources.

Rules:
1. Answer only from the numbered sources. If they do not contain the answer, say: "I can't find this in the guidance I have. Please check with an HR adviser." Do not use outside knowledge.
2. Cite every factual sentence with the source number in square brackets, e.g. [1] or [2][3].
3. Be concise: a direct answer first, then key conditions or exceptions as short bullet points.
4. Quote figures (amounts, weeks, days) exactly as written in the sources.
5. This is general information, not legal advice. If the question involves a specific dispute, dismissal or legal claim, recommend speaking to an HR adviser or Acas."""

NOT_FOUND_MESSAGE = ("I can't find this in the guidance I have. "
                     "Please check with an HR adviser.")


@dataclass
class RAGResult:
    question: str
    answer: str
    sources: list[RetrievedChunk]
    cited: list[int] = field(default_factory=list)            # source numbers used in the answer
    invalid_citations: list[int] = field(default_factory=list)  # cited numbers that don't exist
    llm_called: bool = True
    guardrail: str = ""          # why the LLM was skipped, if it was
    messages: list[dict] = field(default_factory=list)  # exact prompt, for transparency
    model: str = ""
    retrieval_s: float = 0.0
    generation_s: float = 0.0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None

    def to_log(self) -> dict:
        d = asdict(self)
        d["sources"] = [{"rank": s.rank, "chunk_id": s.chunk_id, "score": s.score} for s in self.sources]
        d.pop("messages")
        d["timestamp"] = datetime.now(timezone.utc).isoformat()
        return d


def format_sources(chunks: list[RetrievedChunk]) -> str:
    return "\n\n".join(f"[{c.rank}] {c.title} > {c.heading}\n{c.body}" for c in chunks)


def build_messages(question: str, chunks: list[RetrievedChunk]) -> list[dict]:
    user = f"Sources:\n\n{format_sources(chunks)}\n\nQuestion: {question}"
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def extract_citations(text: str) -> list[int]:
    return sorted({int(n) for n in re.findall(r"\[(\d+)\]", text)})


class RAGPipeline:
    def __init__(self, retriever, llm, top_k: int = 5, min_score: float = 0.3,
                 log_path: str | Path | None = None):
        self.retriever, self.llm = retriever, llm
        self.top_k, self.min_score = top_k, min_score
        self.log_path = Path(log_path) if log_path else None

    def answer(self, question: str, topic: str | None = None) -> RAGResult:
        t0 = time.perf_counter()
        chunks = self.retriever.search(question, k=self.top_k, topic=topic)
        retrieval_s = round(time.perf_counter() - t0, 3)

        # Guardrail: if nothing relevant was retrieved, don't let the LLM guess.
        best = max((c.score for c in chunks), default=0.0)
        if best < self.min_score:
            result = RAGResult(question=question, answer=NOT_FOUND_MESSAGE, sources=chunks,
                               llm_called=False, retrieval_s=retrieval_s,
                               guardrail=f"Best match score {best:.2f} < threshold {self.min_score}")
        else:
            messages = build_messages(question, chunks)
            resp = self.llm.complete(messages)
            cited = extract_citations(resp.text)
            valid = {c.rank for c in chunks}
            result = RAGResult(
                question=question, answer=resp.text, sources=chunks, cited=cited,
                invalid_citations=[n for n in cited if n not in valid],
                messages=messages, model=resp.model, retrieval_s=retrieval_s,
                generation_s=resp.latency_s, prompt_tokens=resp.prompt_tokens,
                completion_tokens=resp.completion_tokens,
            )
        self._log(result)
        return result

    def _log(self, result: RAGResult) -> None:
        """Append every query to a JSONL log: the raw material for monitoring."""
        if not self.log_path:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result.to_log(), ensure_ascii=False) + "\n")
