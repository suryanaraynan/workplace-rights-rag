# Workplace Rights & Wellbeing Assistant

An explainable RAG + ML assistant that answers UK employee-rights and
workplace-wellbeing questions from official GOV.UK guidance — **with
citations**, and **escalation to a human** when it isn't confident.

> **Scenario:** An HR services firm wants line managers to get fast,
> consistent, sourced answers to questions like *"Can my employee get sick
> pay after 3 days off?"* — without the risk of confidently wrong advice.

## Roadmap
| Step | Component | Status |
|---|---|---|
| 1 | Ingestion, cleaning, heading-aware chunking | ✅ |
| 2 | RAG with citations + Streamlit UI | ✅ |
| 3 | Evaluation set and metrics (hit-rate, MRR, faithfulness) | ⏳ |
| 4 | Topic classifier (TF-IDF baseline vs embeddings) | ⏳ |
| 5 | Agent: classify → retrieve → answer or escalate | ⏳ |
| 6 | CI, logging, Docker, optional Azure | ⏳ |

## Quick start
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
echo "GROQ_API_KEY=your-key" > .env   # free key from console.groq.com

python scripts/ingest.py      # fetch 34 curated GOV.UK pages (458 chunks) -> data/processed/
python scripts/build_index.py # embed chunks into a local ChromaDB index
python scripts/ask.py "How much statutory sick pay can an employee get?"
streamlit run app.py          # web UI with answers, sources and scores
pytest                        # 29 tests, no network or API key needed
```

## Project structure
```
config.yaml                  settings + curated corpus with topic labels
src/workplace_rag/
  ingest/govuk_client.py     GOV.UK API client (rate-limited, cached)
  ingest/parse.py            HTML -> structured sections
  chunking.py                heading-aware chunking with overlap
  embeddings.py              local sentence-transformers embedder
  retrieval.py               ChromaDB index + retriever (topic filter)
  llm.py                     provider-agnostic LLM client (Groq/Ollama/OpenAI)
  rag.py                     prompt, guardrail, citation check, query log
  factory.py                 builds the pipeline from config
app.py                       Streamlit UI
scripts/ingest.py            step 1 pipeline
scripts/discover.py          suggests new pages for human review
scripts/build_index.py       step 2: embed + index
scripts/ask.py               ask one question from the terminal
tests/                       unit tests with fixtures
docs/interview-guide.md      why/how/alternatives for every component
```

## How it works
```
question -> embed -> top-5 similar chunks (ChromaDB) -> score < 0.30? -> "can't find this"
                                                    \-> prompt with numbered sources -> LLM
                                                        -> answer with [n] citations -> citation check -> log
```

## Design decisions
Why RAG rather than fine-tuning, long-context prompting or a plain LLM is
covered in the interview guide's *Architecture decisions* section.
See [docs/interview-guide.md](docs/interview-guide.md) — each component
documents why it exists, alternatives considered, and how it was measured.

## Data and licence
Code: MIT. Data: Contains public sector information licensed under the
[Open Government Licence v3.0](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/).
Raw data is not committed; run `scripts/ingest.py` to reproduce it.

This tool provides general information, not legal advice.
