"""Ask one question from the terminal and see the answer with its evidence.

Usage:  python scripts/ask.py "How much statutory sick pay can an employee get?"
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from workplace_rag.factory import load_pipeline  # noqa: E402


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit('Usage: python scripts/ask.py "your question"')
    r = load_pipeline().answer(" ".join(sys.argv[1:]))
    print(f"\n{r.answer}\n")
    if r.guardrail:
        print(f"(LLM not called: {r.guardrail})")
    print("Sources retrieved:")
    for s in r.sources:
        mark = "cited" if s.rank in r.cited else "     "
        print(f"  [{s.rank}] {mark} score={s.score:.2f}  {s.title} > {s.heading}\n         {s.url}")
    if r.invalid_citations:
        print(f"WARNING: answer cites sources that don't exist: {r.invalid_citations}")
    print(f"\nretrieval {r.retrieval_s}s | generation {r.generation_s}s | "
          f"tokens in/out {r.prompt_tokens}/{r.completion_tokens} | model {r.model or '-'}")


if __name__ == "__main__":
    main()
