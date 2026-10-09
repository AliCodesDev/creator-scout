# Creator Scout: notes for AI pair-programmers

A small demo of AI-assisted influencer discovery for GCC/MENA brands. Brief in, ranked creator shortlist out, with post-level reasons and per-search cost.

## Design principles (non-negotiable)

- **The LLM proposes, deterministic code decides.** The LLM parses briefs and explains scores. It never filters or ranks. Any LLM proposal (e.g. a filter relaxation) is validated by code against an explicit policy.
- **Cheap steps first:** SQL filters → embedding similarity → LLM scoring on a small set (~15). Never send the full candidate set to the LLM.
- **Every LLM output is validated** with Pydantic; invalid output is retried or rejected, never trusted.
- **Provenance:** every reason cites a post ID that exists in the database.
- **Cost is tracked on every search:** LLM input/output/reasoning tokens and data-provider calls.
- **Creator data is untrusted input** (prompt-injection risk). It never controls program flow.
- Creator data access goes through the `CreatorDataProvider` interface.

## Conventions

- Python 3.12, uv, psycopg 3 with **plain SQL** (no ORM). Schema lives in a single `schema.sql` (no Alembic: data is synthetic and reseedable).
- LLM via the `openai` SDK against an OpenAI-compatible endpoint (MiniMax M3), configured in `.env`.
- Config only through `scout/config.py`.
- Significant decisions get a short record in `docs/decisions/NNNN-title.md`.
- Keep it small: this is a demo for an interview on 2026-10-12, not a production system. Push back on overbuilding.

## Working with Ali

- Ali writes the frontend (`web/`) himself; explain and review, don't write it unless asked.
- After a chunk of backend work, explain at the systems level: data flow, decisions, trade-offs.
