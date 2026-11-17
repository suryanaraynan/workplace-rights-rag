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
