# Decisions

A running log of the significant decisions in Creator Scout, architectural and otherwise. Each entry: what we decided, why, and what it costs us. Newest entries at the bottom of each section.

**Status:** ✅ in place · 🔜 decided, not built yet · ❓ open

---

## Product and scope

### D1. Build a small, solid demo, not a system ✅
**Decision:** Scope to one flow (brief → ranked shortlist with reasons and cost) on ~300 synthetic creators, built in a weekend.
**Why:** It exists to explore Swavy's core discovery problem first-hand before an interview. Depth of reasoning matters more than feature count.
**Trade-off:** No auth, no multi-tenancy, no outreach/payments, no production hardening.

### D2. A pipeline with LLM steps, not an autonomous agent ✅
**Decision:** Code owns the control flow: parse → filter → rank → score. The LLM is called at fixed points for narrow jobs.
**Why:** The path for discovery is known in advance, so model-driven control flow buys nothing and costs predictability, latency, testability and auditability. Creator content is untrusted; an agent that reads it and then picks its next action is a prompt-injection surface.
**Trade-off:** Less flexible on unusual briefs. See D3 for the one place flexibility is worth it.

### D3. Agency only where the path is unknown: filter relaxation 🔜
**Decision:** Start with deterministic relaxation. If time allows, try a bounded agent loop where the LLM *proposes* a relaxation (tool call with a reason) and code *validates* it against policy, max 3 iterations, with a token budget. Compare both on the evals.
**Why:** When too few creators match, which constraint to loosen depends on the brief (for "natural hair in Riyadh", widening the city probably beats dropping the theme). That's a judgment call a fixed order can't make.
**Trade-off:** More cost and latency per search; must be measured, not assumed.

### D4. Text only, no vision 🔜
**Decision:** Fit scoring uses bios and captions only.
**Why:** Swavy's scoring uses vision, but images add a captioning/embedding pipeline that doesn't change the architecture. Out of scope for the weekend.
**How it would plug in:** caption each post image with a vision model, embed the caption alongside the text, and let the scorer cite image-derived evidence by post ID.

### D5. Public GitHub repo ✅
**Why:** It can be shared as a link. The API key is in `.env`, which is gitignored.

---

## Core principles

### D6. The LLM proposes, deterministic code decides ✅
**Decision:** The LLM parses briefs and explains scores. Filtering and ranking are SQL and vector math. Any LLM proposal is validated by code against explicit rules.
**Why:** Mirrors Swavy's own principle. Decisions become reproducible, testable and auditable.

### D7. Cheap steps first, cost visible on every search 🔜
**Decision:** SQL filters (free) → embedding similarity (local, free) → LLM scoring on the top ~15 only. Every search reports LLM tokens (input, output, reasoning) and data-provider calls in USD.
**Why:** LLM cost scales with candidates scored, so you shrink the set before calling the LLM. Making cost visible keeps that discipline honest.

### D8. Every LLM output is validated, never trusted ✅
**Decision:** All LLM calls go through `scout/llm.py:call_tool`: forced tool call → Pydantic validation → optional custom check → on failure, the error is sent back to the model for another attempt (max 3).
**Why:** Models, especially cheaper ones, sometimes return malformed or off-spec output. Validation turns "usually right" into "right or explicitly failed".
**Evidence (dataset generation):** 300 creators took 415+ calls, so roughly a third needed at least one retry. Failure modes seen: captions in the wrong alphabet, too few captions (output truncated), the model not calling the tool at all, and the model looping in its reasoning until it hit `max_tokens` (which the server surfaced as a dropped connection). Without validation, about 1 creator in 3 would have had broken data.

### D9. Provenance: every reason cites a real post 🔜
**Decision:** Scoring reasons must cite post IDs; code checks every cited ID was in the set sent to the model and rejects invented ones.
**Why:** Carried over from Safiyr's clinician brief: a claim you can't trace to a source shouldn't be shown to a decision-maker.

---

## Architecture and stack

### D10. Postgres + pgvector, one database ✅
**Decision:** Relational data and embeddings live in one Postgres. No separate vector database.
**Why:** One query can apply hard filters and rank by similarity; no syncing two systems.
**Trade-off:** At Swavy's scale (200M profiles), filtered approximate-nearest-neighbour search is hard in pgvector (the index returns neighbours, then `WHERE` discards most of them). That's where you'd partition by country/language or move to a dedicated search engine.

### D11. No vector index at this scale 🔜
**Decision:** Exact (sequential) similarity search, no HNSW/IVFFlat.
**Why:** ~3,000 posts compare in milliseconds, exactly. An approximate index adds tuning and recall loss for no benefit here.

### D12. psycopg 3 with plain SQL, no ORM ✅
**Why:** "Filtering is deterministic SQL" is a core design point, so the SQL should be visible and reviewable, not generated.
**Trade-off:** Manual mapping between rows and models; fine at this size.

### D13. One `schema.sql`, no migrations ✅
**Decision:** `seed.py` drops and recreates the schema.
**Why:** Migrations (Alembic) solve evolving a live database with data you must keep. Ours is synthetic and reseedable in seconds.
**Would change when:** there's real data or more than one environment.

### D14. Synchronous search, no task queue 🔜
**Decision:** No Celery/Redis. A search runs inside the HTTP request.
**Why:** A search should take ~10–30 s. A loading state is enough for a demo. A queue adds two services and failure modes.
**Would change when:** searches get long or concurrent, or results need to be pushed later.

### D15. No agent/LLM framework ✅
**Decision:** The `openai` SDK directly, plus a ~100-line helper (`scout/llm.py`).
**Why:** The whole LLM interaction (prompt, tool schema, validation, retry, cost) is visible and debuggable. Frameworks hide exactly the parts this project is about.

### D16. MiniMax M3 as the LLM, via the OpenAI-compatible API ✅
**Decision:** MiniMax M3 (open weights, hosted API, ~$0.30 / $1.20 per M input/output tokens), called with the `openai` SDK and a custom `base_url`.
**Why:** Cheap, supports tool calling, and swapping providers is a config change (`LLM_BASE_URL`, `LLM_MODEL`).
**Trade-offs:**
- It's a reasoning model: in the preview, **74% of output tokens were reasoning**, billed as output and adding latency (~50 s per generation call). Cost tracking reports reasoning tokens separately.
- Thinking arrives inline as `<think>…</think>` unless `reasoning_split` is set; we always set it.
- Tool calling is less battle-tested than frontier models', which is why D8 matters more.
- "Open source" doesn't change much here, since we call the hosted API. The benefit is price.
- `max_tokens` covers reasoning + answer: 4,000 truncated some answers, so generation uses 8,000. A runaway reasoning loop looks like a network error from the outside.
- Prices in `config.py` come from third-party listings; verify against MiniMax's own pricing.

### D17. uv, ruff, pytest; no mypy, no pre-commit ✅
**Why:** Fast, standard tooling with a lockfile for reproducibility. Type checking and hooks are good practice but slow the loop for a weekend build.

### D18. Embedding model chosen in the ranking step 🔜 ❓
**Decision:** Pick after a quick test on Gulf-dialect Arabic, Arabizi and code-switched captions. Candidates: `multilingual-e5-base` (768-d, light) vs `bge-m3` (1024-d, stronger on Arabic, heavier on CPU).
**Why:** The vector size is fixed by the model, and multilingual quality on *dialect* (not just MSA) is the real risk.

---

## Data

### D19. One row per platform account; posts are the unit of evidence ✅
**Decision:** `creators` (one per platform account) and `posts`. Embeddings will live on posts. IDs are short and readable (`c_0042`, `p_00193`).
**Why:** Per-post embeddings make provenance possible: the posts that drive a creator's rank are the ones the scorer cites. Short IDs reduce garbled citations.
**Trade-off:** No cross-platform identity (the same person on Instagram and TikTok is two creators). Identity resolution is a hard problem we explicitly skip.

### D20. Provider categories are coarse and can be wrong ✅
**Decision:** `categories` holds the provider's labels (`beauty`, not `haircare`), with ~7% deliberately wrong. They're used only for hard exclusions.
**Why:** Realistic partner data. Fine-grained matching must come from content, not labels.

### D21. `CreatorDataProvider`: ingestion plus a paid shortlist refresh ✅
**Decision:** The interface has `list_creators()` (bulk ingestion into our DB) and `get_recent_posts(creator_id)` (a paid per-creator refresh called at search time only for the ~15 shortlisted creators).
**Why:** You can't run SQL or vector search over a partner's API, so you ingest. Cached data goes stale, so you pay to refresh only what the LLM will score: cheap-first applied to data.
**Trade-off:** The synthetic provider doesn't simulate content drift (refresh returns the ingested posts), and its per-call cost ($0.002) is simulated.

### D22. Synthetic data: code decides facts, the LLM writes prose ✅
**Decision:** A seeded RNG decides every structured fact (country, city, languages, followers, engagement, true niche, edge cases, first name). MiniMax writes display names, handles, bios and captions for the given persona. Generated once (~$1), committed as `data/creators.json`; seeding never calls an LLM.
**Why:** LLMs write convincing text but produce poor distributions: asked freely, 2 of 6 preview creators were both "Lama, @lama.curls". Code-controlled facts give realistic distributions and reproducibility.
**Actual cost:** ~$1.57 for 300 creators / 3,000 captions, about 1.6M output tokens of which ~55% reasoning.
**Plan rules that exist because of model behavior:** first names picked by code (mode collapse), Arabizi capped at 5/10 posts (at 7/10 the model could loop forever).
**Trade-off:** Still synthetic. LLM-written captions are cleaner than real ones, which likely flatters embedding quality. Real-data results would be worse.

### D23. Planted edge cases, with ground truth kept out of the database ✅
**Decision:** The dataset includes Gulf/Egyptian/Lebanese dialects, MSA, Arabizi, code-switching, mixed-niche creators, mislabeled categories, follower counts just outside common ranges, decoys (beauty bio, posts about cars), prompt injections in bios, and clusters for eval briefs. The true labels go into `data/ground_truth.json`, which is never loaded into the database.
**Why:** Avoids circular evals. The system sees only bios and captions, as it would with real data; evals check whether it recovers the planted truth.
**Early finding:** a keyword search for injections on the seeded data both missed one planted injection (no trigger word) and flagged an innocent makeup artist whose bio offers "تعليمات خطوة بخطوة" (step-by-step tutorials). Injection defense must be structural (data clearly delimited in prompts, validated outputs, scores decided in code), not keyword filtering.

---

## Process

### D24. One decisions log, written as we go ✅
**Decision:** This file, instead of one file per decision.
**Why:** ~25 short decisions read better in one place, and writing them at decision time keeps them honest.

### D25. Ali writes the frontend; AI pairs on the backend ✅
**Why:** Frontend is the skill gap this project is meant to close. The rules AI assistants follow are in `CLAUDE.md`: they get constraints, not just tasks.

### D26. Frontend framework ❓
**Open:** Next.js (familiar from Safiyr, likely matches the role) vs Vite + React (less to learn). Decide when we get there.
