# Interview Guide — Workplace Rights & Wellbeing Assistant

This guide is updated after every build step. Each component answers the
same six questions: **Why · How · Alternatives · Measured · Limitations · Likely questions.**

---

## The 2-minute pitch (draft — final numbers added as steps complete)

> "I built an assistant that helps line managers answer UK employee-rights
> questions — sick pay, flexible working, redundancy — using official GOV.UK
> guidance. The framing was a consulting brief: an HR services firm wants
> faster, consistent answers, every answer must cite its source, and the
> system must know when to hand off to a human adviser.
>
> It combines retrieval-augmented generation for the answers, a classical
> ML classifier that routes questions by topic and flags low-confidence ones
> for escalation, and a simple agent that ties them together. I built an
> evaluation set to measure retrieval and answer faithfulness, so every
> design choice is backed by a number, not a guess. It runs fully locally
> on free models, and the LLM is swappable by config."

**Why this project (if asked):** It mirrors real consulting work — a
regulated domain where wrong answers have consequences, so accuracy,
citations and knowing your limits matter more than a clever demo.

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
1. **Discover** — hand-picked seed pages (guaranteed coverage of core topics)
   plus GOV.UK Search API queries (wider coverage). De-duplicated.
2. **Fetch** — `GovUKClient` calls the public Content API (JSON, no key).
   Rate-limited (0.5 s between calls) and **cached to disk**, so re-runs are
   fast and reproducible.
3. **Filter** — keep only `answer`, `guide`, `detailed_guide` formats; forms,
   news and transactions are noise for Q&A.
4. **Parse** — HTML → `Section`s split at h2/h3/h4. Tables become
   `cell | cell` lines so figures like "£118.75" survive. Multi-part guides
   get a **deep link per part** for precise citations. GOV.UK topic tags
   (taxons) are kept — they become free labels for the classifier in step 4.
5. **Chunk** — within each section, a sliding window of 200 words with
   40-word overlap. Each chunk is prefixed with `Page title > Heading`.

### Alternatives considered (and why rejected)
| Option | Why not |
|---|---|
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

### Measured
_To fill in after you run it:_ number of documents, number of chunks,
median chunk length, topic distribution. (The script prints all of these.)
Retrieval impact of chunk size gets measured in step 3.

### Engineering habits to point out
- All tunables live in `config.yaml` — experiments are config changes.
- 12 unit tests, including a fake HTTP session so tests never hit the network.
- A real bug caught by a test: nested lists were losing their outer text.
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

**Q: How do you handle tables?**
A: Each row becomes a line of `cell | cell` text, so numbers stay next to
their labels. For heavily tabular data I'd consider storing tables separately
and querying them with structured tools instead.

**Q: Is the data safe to use?**
A: Yes — Open Government Licence v3.0 allows reuse with attribution. I
checked that before building, which matters in client work too.
