# Phase 0 — Data Collection — Design Spec

**Sub-project #1 of the Two-Tower Job Matcher.** Date: 2026-07-28.

Follows Foundation (#0). Fetches real job postings from public ATS APIs and normalizes
the four different response shapes into one clean JSONL file — the corpus every later
phase encodes, labels, indexes, and evaluates against.

## Goal

Produce `data/postings.jsonl`: 1,000+ real job postings in one schema, collected politely
and cached, from Greenhouse, Lever, Ashby, and RemoteOK.

## Scope

**In scope**
- A curated company list (`src/data/companies.py`): ~100–150 AI/ML companies with their
  ATS source + board ID.
- Fetchers (`src/data/fetch.py`): one per source, with on-disk caching and rate limiting.
- Normalizer (`src/data/normalize.py`): each source's shape → the common schema; HTML → clean
  plain text; dedupe.
- A `run` entry point that fetches, normalizes, writes `data/postings.jsonl`, and prints a
  summary (postings per source, companies that responded vs. failed).

**Out of scope**
- The resume. The system matches jobs to a resume, but the resume enters at Phase 1
  (labeling) and query time. Resume-PDF text extraction is a Phase 1 task.
- Any encoding, labeling, indexing, or evaluation.
- Filtering to ML-only roles (see Decision 1 — we deliberately keep all roles).

## Deliverable schema

`data/postings.jsonl`, one JSON object per line:

```json
{"id": "greenhouse:anthropic:4012345", "company": "Anthropic", "title": "ML Engineer",
 "location": "San Francisco, CA", "description": "<full plain text>",
 "url": "https://boards.greenhouse.io/anthropic/jobs/4012345", "source": "greenhouse"}
```

- `id` — globally unique, namespaced so postings from different sources never collide:
  `"{source}:{ats_id}:{posting_id}"` for the board-driven sources, and `"remoteok:{posting_id}"`
  for RemoteOK (no board ID).
- `company` — human-readable display name (from the curated list, not the API slug).
- `title`, `location`, `url` — as provided by the source (location may be empty).
- `description` — full plain text, HTML stripped, whitespace-normalized (see Decision 2).
- `source` — one of `greenhouse | lever | ashby | remoteok`.

## The four sources (endpoints + shape we read)

All are public, **no auth, no API key** (so Phase 0 touches no secrets and needs no `.env`).

| Source | Endpoint | Postings live at | Title | Location | Description | URL |
|---|---|---|---|---|---|---|
| Greenhouse | `https://boards-api.greenhouse.io/v1/boards/{id}/jobs?content=true` | `jobs[]` | `title` | `location.name` | `content` (HTML, escaped) | `absolute_url` |
| Lever | `https://api.lever.co/v0/postings/{id}?mode=json` | top-level array | `text` | `categories.location` | `descriptionPlain` (already plain!) | `hostedUrl` |
| Ashby | `https://api.ashbyhq.com/posting-api/job-board/{id}` | `jobs[]` | `title` | `location` | `descriptionPlain` if present else `descriptionHtml` | `jobUrl` |
| RemoteOK | `https://remoteok.com/api` | array; **skip element 0** (legal/metadata) | `position` | `location` | `description` (HTML) | `url` |

Notes that bite if ignored:
- **Greenhouse** `content` is HTML *and* HTML-entity-escaped — unescape then strip.
- **Lever** already gives `descriptionPlain`; prefer it, no stripping needed.
- **Ashby** board ID is the job-board name; prefer `descriptionPlain` when the API includes it.
- **RemoteOK** returns one aggregated feed (not per-company); its element `[0]` is a legal
  notice, not a job. It also expects a real `User-Agent` header or may reject the request.
  Its postings carry their own `company` field — use it directly (this is the one source not
  driven by the curated list).

## Design decisions (approved)

**1. Keep all job types — do not filter to ML roles.** Labeling (Phase 1) needs obviously-bad
matches (sales, recruiting, ops) as negatives, and eval needs them to be meaningful. Pull
every posting from each company and keep them all.

**2. Descriptions → clean plain text, stored in full.** Strip HTML to plain text (that is what
gets encoded); normalize whitespace; do **not** truncate — length-capping is the encoder's job
later. Use a lightweight stdlib HTML-to-text helper (`html.parser` + `html.unescape`), falling
back to `beautifulsoup4` only if real data proves it inadequate.

**3. Dedupe by content, keep the first seen.** Key on normalized `(company, title, location)`
(lowercased, whitespace-collapsed). First occurrence wins. Handles the same company appearing
on two platforms and RemoteOK overlap.

## Fetching: caching + politeness

- **Cache raw responses** to `data/raw/{source}/{id}.json` (RemoteOK: `data/raw/remoteok.json`).
  Fetchers read cache first; a re-run does zero network unless cache is missing or `--refresh`
  is passed. This makes normalization iterable without re-hitting APIs.
- **Rate limit**: a small fixed delay (~0.5–1 s) between live requests; a real `User-Agent`.
- **Failures are non-fatal**: a company whose board 404s / errors is logged and skipped; the
  run continues. A per-source timeout guards against a hung request.
- `data/` is gitignored, so raw cache and `postings.jsonl` never get committed. The **curated
  company list is source code** (`src/data/companies.py`), tracked in git.

## The ID-verification reality

Curated ATS IDs come from model knowledge with a cutoff and *will* be partly stale (renamed
boards, platform switches, shut-down companies). Expect ~60–80% to resolve on first run. The
run summary lists which companies responded vs. failed so the list can be corrected. Reaching
1,000+ postings is still comfortable because each live company posts many roles.

## Conventions (unchanged from Foundation)

- Python 3.9.6 via `.venv`; run with `.venv/bin/python`.
- ruff, line-length 120, `select = ["E", "F", "W"]`, clean.
- New dependency: `requests` (uncomment its `requirements.txt` row; install into `.venv`).
  `python-dotenv` stays commented — Phase 0 needs no key.
- Testing: inline `if __name__ == "__main__"` sanity checks per module (project convention),
  plus the `run` summary as the integration check. Normalizers are tested against small
  saved sample payloads so tests don't depend on the network.

## Testing / done criteria

- `normalize.py` sanity block: given a saved sample payload from each source, it emits records
  matching the schema (all fields present, `description` plain-text and non-empty, `id`
  namespaced correctly).
- HTML-to-text helper sanity: tags stripped, entities unescaped, whitespace collapsed.
- Dedupe sanity: two records with the same normalized `(company, title, location)` collapse to
  one.
- Integration: `run` produces `data/postings.jsonl` with **≥1,000** unique postings across
  **≥2** sources, and prints the per-source counts + responded/failed company lists.
- ruff clean; work committed on a `phase0-data` branch (first sub-project where a feature
  branch earns its keep — real network code worth reviewing before it hits `main`).

## Open fill-ins (not blockers)

- The actual curated company list (built during implementation).
- Whether the stdlib HTML stripper suffices or we add `beautifulsoup4` (decided against real
  Greenhouse/RemoteOK HTML during implementation).
