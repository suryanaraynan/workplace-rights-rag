# Interview Guide — Workplace Rights & Wellbeing Assistant

This guide is updated after every build step. Each component answers the
same six questions: **Why · How · Alternatives · Measured · Limitations · Likely questions.**

---

## The 2-minute pitch (draft — final numbers added as steps complete)

> "I built an assistant that helps line managers answer UK employee-rights
> questions — sick pay, flexible working, redundancy — using official GOV.UK
> guidance. I framed it as a client brief: an HR services firm wants
> faster, consistent answers, every answer must cite its source, and the
> system must know when to hand off to a human adviser.
>
> It combines retrieval-augmented generation for the answers, a classical
> ML classifier that routes questions by topic and flags low-confidence ones
> for escalation, and a simple agent that ties them together. I built an
> evaluation set to measure retrieval and answer faithfulness, so every
> design choice is backed by a number, not a guess. It runs fully locally
> on free models, and the LLM is swappable by config."

**Why this project (if asked):** It mirrors real client work — a
regulated domain where wrong answers have consequences, so accuracy,
citations and knowing your limits matter more than a clever demo.

---

## Architecture decisions — why RAG, and why this kind of RAG?

### The problem's requirements (start every design answer here)
1. **Accurate, current facts.** Rates change every April (e.g. sick pay); a wrong figure has real consequences.
2. **Verifiable.** A manager must be able to click through to the official source.
3. **Knows its limits.** When the guidance doesn't cover a question, say so and escalate.
4. **Cheap and easy to update.** When GOV.UK changes, the system should update in minutes, not require retraining.
5. **Explainable.** Show why an answer was given.

Every design choice below is judged against these five.

### Option A — Plain LLM, no retrieval ("just ask ChatGPT")
- ✅ Simplest possible build.
- ❌ Knowledge frozen at the training cutoff, so figures go stale (fails 1).
- ❌ No sources to check (fails 2); confidently invents answers (fails 3).
- **Verdict:** fine for drafting an email, unacceptable for employment rights.

### Option B — Fine-tune an LLM on the guidance
- ✅ Can learn domain tone and terminology.
- ❌ Fine-tuning teaches *style and patterns* reliably, but is unreliable for injecting *facts*. Models still hallucinate details.
- ❌ Every guidance change means retraining (fails 4). Expensive, needs GPUs.
- ❌ Still no citations (fails 2).
- **Verdict:** wrong tool for knowledge. Fine-tuning is worth considering later only for *behaviour* (e.g. always answering in a house format), and it combines with RAG rather than replacing it.

### Option C — Long context: put all 34 pages in every prompt
- ✅ No retrieval step, so nothing can be "missed" by search.
- ❌ **Measured cost:** the corpus is ~54,000 tokens vs ~900 tokens for a RAG prompt (top-5 chunks): **~60× more per question**. On the Groq free tier (~8,000 tokens/minute for this model) a single request wouldn't even fit.
- ❌ Slower, and models are known to use information in the middle of very long prompts less reliably ("lost in the middle").
- ❌ Doesn't scale: 34 pages today, 3,000 pages for a real client.
- ❌ Weaker citations: harder to say exactly which paragraph supported the answer.
- **Verdict:** a legitimate choice for a small, fixed set of documents. Worth knowing as the honest baseline, and step 3 could compare it.

### Option D — Search engine only (keyword / semantic search, no LLM)
- ✅ Fully transparent, no hallucination possible, cheap.
- ❌ Returns pages, not answers; the manager still has to read and combine several sections (e.g. eligibility + rates + exceptions).
- **Verdict:** this is actually the *retrieval half* of RAG. RAG = this + an LLM that synthesises a cited answer.

### Option E — Rules / decision trees (like GOV.UK's "smart answers")
- ✅ Perfectly predictable and auditable.
- ❌ Every question path must be hand-built; can't handle free-text questions phrased in unexpected ways.
- **Verdict:** great for a few high-volume, high-stakes calculations (e.g. holiday entitlement). Could later be added as a **tool** the agent calls in step 5.

### Option F — GraphRAG / knowledge graph
- ✅ Strong when answers need connecting many entities (e.g. "which suppliers of X are affected by Y").
- ❌ The guidance is document-shaped, not entity-relationship-shaped. Building a graph adds a lot of complexity for little gain.
- **Verdict:** overkill here.

### Option G — Agentic RAG from day one (LangChain/LlamaIndex agents)
- ✅ Flexible: multi-step reasoning, tool use.
- ❌ More moving parts, harder to debug and evaluate; hard to explain without a working baseline.
- **Verdict:** build the simple pipeline first, measure it (step 3), then add an agent only where it measurably helps (step 5).

### ✅ Chosen: classic RAG (retrieve → generate with citations), plus guardrails
It meets all five requirements: answers come from current official text (1), every claim cites a numbered source with a link (2), a similarity threshold and an "I can't find this" instruction handle gaps (3), updating means re-running ingestion and re-embedding, which takes minutes (4), and every step is visible in the UI (5).

### Design choices *inside* the RAG pipeline
| Decision | Chosen | Alternatives considered | Why |
|---|---|---|---|
| Framework | Hand-written Python (~300 lines) | LangChain, LlamaIndex | Every line is understood and testable; easy to explain. Frameworks are great for speed but hide what's happening. I can say what they do under the hood. |
| Embedding model | all-MiniLM-L6-v2 (local, ~90 MB) | bge-small, OpenAI/Azure embeddings | Free, private, fast on a laptop. Step 3 compares models on the eval set. |
| Vector store | ChromaDB (local, persistent) | NumPy, FAISS, pgvector, Pinecone, Azure AI Search | Zero setup, metadata filtering (topic), persists to disk. At 463 chunks NumPy would work too; managed services matter at scale or for enterprise security. |
| Retrieval | Dense (semantic) top-5 | BM25 keyword, hybrid, + reranker | Start simple; step 3 measures whether hybrid/reranking improve hit-rate. Keyword search may help with exact terms like "SSP1 form". |
| Top-k | 5 | 3, 10 | Enough to cover eligibility + rates + exceptions; small enough to keep the prompt focused (~900 tokens). Tunable in the UI. |
| LLM | Groq `gpt-oss-120b` (hosted, free tier) | Local Ollama, OpenAI, Azure OpenAI | Fast and free; no 2 GB download. Pluggable: one config line switches to a local model if data can't leave the network. |
| Temperature | 0.1 | 0.7 default | We want faithful, repeatable answers, not creative ones. |
| Grounding | System prompt rules + numbered sources + citation check | Trust the model | Citations are verified in code: invalid source numbers are flagged. |
| "Don't know" | Similarity threshold (skip LLM) + prompt instruction | Prompt only | Two layers: the cheap threshold stops obviously off-topic questions before spending an LLM call. |

---

## Step 1 — Data ingestion and chunking

### Why it's built
RAG quality is capped by data quality. If the text is messy or split badly,
no model downstream can recover. This step turns GOV.UK pages into clean,
well-labelled chunks — the "knowledge" the assistant can draw on.

### How it works
**Plain English:** Find the relevant official pages, download them, strip
out web formatting, keep the headings, cut the text into small labelled
pieces, and save them.

**Technical:**
1. **Curated corpus** — 34 GOV.UK pages in `config.yaml`, each hand-labelled
   with one of 9 topics (sick leave, family leave, dismissal and redundancy…).
   `scripts/discover.py` uses the Search API to *suggest* new pages for review;
   nothing is added automatically (see "What the first run taught me").
2. **Fetch** — `GovUKClient` calls the public Content API (JSON, no key).
   Rate-limited (0.5 s between calls) and **cached to disk**, so re-runs are
   fast and reproducible. Moved pages (`redirect` items) are followed.
3. **Filter** — keep only `answer`, `guide`, `detailed_guide` formats; forms,
   news and transactions are noise for Q&A.
4. **Parse** — HTML → `Section`s split at h2/h3/h4; an h3 remembers its h2
   parent. Tables become `cell | cell` lines so figures like "£118.75" survive.
   Placeholder sections (`[TBC]`) are dropped. Multi-part guides get a
   **deep link per part** for precise citations.
5. **Chunk** — merge neighbouring sections under 50 words (same URL only),
   then a sliding window of 200 words with 40-word overlap. Each chunk is
   prefixed with `Page title > Parent heading > Heading`.

### Alternatives considered (and why rejected)
| Option | Why not |
|---|---|
| Auto-crawl via search (v1) | Tried it: ~40% of chunks were off-topic. In a regulated domain, a smaller clean corpus beats a big noisy one. |
| Filter on GOV.UK's own topic tags | Tags are inconsistent (minimum wage is tagged "Recruiting and hiring"; one page has no tags). Hand labels for 34 pages took minutes. |
| Scrape HTML pages | The Content API gives clean, structured JSON and metadata (topics, dates). Scraping is brittle when the site layout changes. |
| Fixed-size chunks across the whole page | Mixes unrelated topics in one chunk, which blurs the embedding. Section-aware chunks stay on one subject. |
| Whole sections as chunks | Some sections are very long; a long chunk dilutes the embedding and wastes LLM context. |
| Semantic chunking (split where embedding similarity drops) | More compute and complexity. GOV.UK content is already well structured by headings, so headings are a cheap, strong signal. Worth testing later as an experiment. |
| No overlap | A fact split across two chunks becomes unfindable. 20% overlap is a common, cheap safeguard. |

### Why the contextual header matters
A chunk like *"You must reply within 2 months"* is ambiguous on its own.
With *"Flexible working > After the application"* in front, both the
embedding and the LLM know what it's about. This is a light version of
"contextual retrieval".

### What the first run taught me (your best interview story for this step)
I didn't trust the pipeline just because the tests passed — I looked at the output.

| Metric | v1 (search crawl) | v2 (curated + fixes) |
|---|---|---|
| Documents | 66 | 34 |
| Chunks | 1,826 | 463 |
| Median chunk length | 58 words | 82 words |
| Chunks under 50 words | 600 (33%) | 46 (10%) |
| `[TBC]` placeholder chunks | 12 | 0 |

Three root causes, three fixes:
1. **Off-topic pages.** Search pulled in visas, driving tests, prison sentences.
   One emergency-recovery page alone was 266 chunks (15% of the index).
   → Curated corpus, discovery becomes a suggestion tool.
2. **Placeholder text.** Pages had sections like "Northern Ireland: [TBC]".
   → Filter placeholders at parse time.
3. **Fragmented subsections.** "Job sharing" (9 words) was its own chunk with
   no mention of flexible working. → Track parent headings and merge small
   siblings into one chunk: *"Flexible working > Types of flexible working:
   …Job sharing: … Hybrid working: …"*.

The remaining short chunks are genuine one-fact page parts; merging them
across pages would break citation links, so I kept them deliberately.

**Chunks per topic:** family leave 91, holidays & time off 80, discrimination
& disability 67, contracts & status 62, pay & pensions 49, dismissal &
redundancy 42, working hours & flexibility 34, disputes & whistleblowing 21,
sick leave 17. The imbalance matters for step 3 (eval questions should cover
small topics too) and step 4 (classifier class imbalance).

### Engineering habits to point out
- All tunables live in `config.yaml` — experiments are config changes.
- 18 unit tests, including a fake HTTP client so tests never hit the network.
- Bugs caught by tests: nested lists losing their outer text; redirect loops.
- Inspected real output rather than trusting green tests — that's how the
  data-quality issues above were found.
- Licence checked up front: Open Government Licence v3.0, attribution in README.

### Limitations / next steps
- Guidance changes (e.g. rates update each April). Production would need
  scheduled re-ingestion using `public_updated_at` to only refresh changed pages.
- Word counts approximate tokens; a tokenizer-based limit would be more exact.
- England/Scotland/Wales/NI differences are not yet tagged.

### Likely interview questions
**Q: How did you choose the chunk size?**
A: Started with 200 words (~260 tokens) — small enough that one chunk is
about one thing, big enough to hold a full rule. Then I measured retrieval
hit-rate at different sizes on my eval set rather than trusting a default.

**Q: Why keep headings?**
A: They're free, human-written summaries of what each paragraph is about.
They improve chunk boundaries, retrieval and citations.

**Q: How would you keep the knowledge base current?**
A: Scheduled job, compare `public_updated_at` against the stored version,
re-ingest only changed pages, re-embed only their chunks, and log what changed.

**Q: Tell me about a data-quality problem you found.**
A: (Use the v1 → v2 story above. Structure: what I saw in the stats → how I
investigated, by reading actual chunks → root cause → fix → measured result.)

**Q: Why not just index everything and let retrieval sort it out?**
A: Irrelevant chunks compete for the top-k slots. A visa page that happens to
mention "work" can push out the right answer, and the LLM may then answer
from the wrong source. Precision of the corpus is cheaper than fixing it later.

**Q: Why hand-label topics — doesn't that not scale?**
A: For 34 pages it's minutes of work and gives clean labels. At scale I'd
bootstrap with an LLM or a classifier, then spot-check a sample by hand.

**Q: How do you handle tables?**
A: Each row becomes a line of `cell | cell` text, so numbers stay next to
their labels. For heavily tabular data I'd consider storing tables separately
and querying them with structured tools instead.

**Q: Is the data safe to use?**
A: Yes — Open Government Licence v3.0 allows reuse with attribution. I
checked that before building, which matters in client work too.

---

## Step 2 — Retrieval-augmented answers with citations

### Why it's built
This is the core product: a manager asks a question in plain English and
gets a short answer, with each claim linked to the official paragraph it
came from, or an honest "I can't find this".

### How it works
**Plain English:** Turn the question into numbers that capture its meaning,
find the 5 guidance passages with the closest meaning, hand those passages
to the LLM with strict rules ("only use these, cite them"), then check
that the citations it gave actually exist.

**Technical (one request):**
1. **Embed the question** with the same model used for the chunks
   (all-MiniLM-L6-v2, normalised vectors). The index stores which model
   built it and refuses a mismatch: mixing models silently breaks search.
2. **Retrieve** top-k=5 by cosine similarity from ChromaDB. Optional
   `topic` metadata filter (used by the agent in step 5).
3. **Guardrail:** if the best score is below 0.30, return "can't find this"
   *without calling the LLM*. Saves cost and stops guessing on off-topic
   questions. (The threshold is provisional: step 3 calibrates it.)
4. **Prompt:** system prompt with 5 rules (sources only, cite every
   sentence, be concise, exact figures, not legal advice), then numbered
   sources `[1] Title > Heading` and the question.
5. **Generate** via an OpenAI-compatible client: Groq by default; Ollama
   or OpenAI by changing `llm.provider` in config.
6. **Verify citations in code:** extract `[n]` markers; flag numbers that
   weren't in the sources (hallucinated citations) or answers with none.
7. **Log** question, retrieved chunk IDs, scores, latency and tokens to
   `data/logs/queries.jsonl`: the starting point for monitoring.

**Explainability in the UI:** the answer sits next to its evidence. Each
source shows its similarity score, whether it was actually cited (✅), and
a link to GOV.UK. An expander shows the exact prompt sent to the model.

### Measured
_To fill in after running:_ index build time, typical retrieval and
generation latency, tokens per question. Quality metrics (hit-rate,
faithfulness) come in step 3. Until then, this step is "it works", not
"it's good". Say exactly that in interviews: it shows maturity.

### A real-world lesson
The original plan was Llama 3.3 on Groq. When building, I checked Groq's
model list: Llama 3.x had been retired for free accounts in August 2026.
Because the LLM sits behind a config-driven interface, switching to
`gpt-oss-120b` was one line. Hosted models get retired; design for it.

### Engineering habits to point out
- 28 tests; the pipeline is tested with fake retriever/LLM/embedder, so
  tests are fast, free and never call an API.
- API key in `.env`, which is git-ignored; a missing key gives a clear error.
- Same `load_pipeline()` factory used by the CLI and the UI: one source of truth.

### Limitations / next steps
- Dense retrieval can miss exact terms (form names like "SP1", acronyms):
  step 3 tests hybrid BM25 + dense.
- The citation check proves citations *exist*, not that they *support* the
  sentence: step 3 adds a faithfulness check (LLM-as-judge).
- No conversation memory: each question stands alone (fine for v1).
- Retrieved text is trusted. With user-uploaded documents I'd add
  prompt-injection defences; here all sources are official GOV.UK pages.

### Likely interview questions
**Q: Walk me through what happens when a user asks a question.**
A: (Use the 7 technical steps above, in about 60 seconds.)

**Q: How do you stop hallucinations?**
A: Layers, because no single control is enough: (1) only relevant sources
in the prompt, (2) instructions to answer only from them and to say when
they don't cover it, (3) low temperature, (4) a similarity threshold that
skips the LLM entirely for off-topic questions, (5) a code-level citation
check, and in step 3 (6) a faithfulness evaluation. Plus: the user always
sees the evidence, so a human can check.

**Q: Why cosine similarity?**
A: It compares direction, not length, so a short question and a long
passage about the same thing still score highly. With normalised vectors
it's equivalent to a dot product, which is fast.

**Q: Why top-5?**
A: Questions often need eligibility, amounts and exceptions, which sit in
different sections. Five covers that while keeping the prompt around 900
tokens. It's a tunable I'll set from eval data rather than guess.

**Q: What if the vector index and embedding model get out of sync?**
A: Searches silently return garbage. So the index stores the model name
and the retriever refuses to run on a mismatch, with a clear error.

**Q: Why not LangChain?**
A: For a ~300-line pipeline, writing it directly made every step visible
and testable, and I can explain exactly what happens. On a team project
with many integrations I'd happily use a framework. Knowing the internals
makes me better at debugging one.

**Q: How would you deploy this for a client?**
A: Containerise it, put the API key in a secrets manager, swap Chroma for
a managed vector store (e.g. Azure AI Search or pgvector) if the corpus
grows or needs enterprise security, use the client's approved LLM endpoint
via config, and ship the query logs to a monitoring dashboard. Covered in step 6.
