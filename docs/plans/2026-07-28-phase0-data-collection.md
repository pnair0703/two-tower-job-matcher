# Phase 0 — Data Collection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce `data/postings.jsonl` — 1,000+ real job postings in one schema, fetched politely (with on-disk caching) from Greenhouse, Lever, Ashby, and RemoteOK.

**Architecture:** Four small modules. `companies.py` is the curated target list (data as code). `normalize.py` is pure logic: each source's payload shape → the common schema, HTML → plain text, dedupe. `fetch.py` is the only networked code: cache-first JSON GET with politeness. `collect.py` ties them together and prints the run summary. Normalization is tested offline against saved sample payloads; only `collect.py` touches the live APIs.

**Tech Stack:** Python 3.9.6 (`.venv`), `requests` (new dep), stdlib `html.parser` for HTML→text, ruff.

## Global Constraints

Every task's requirements implicitly include these:

- Work on branch **`phase0-data`** (created in Task 1). Do not commit to `main`.
- Python 3.9.6: use `.venv/bin/python` / `.venv/bin/pip` explicitly. PEP 585 generics (`list[dict]`) are fine; `X | None` unions are NOT (3.10+) — use `Optional`.
- ruff clean (`.venv/bin/ruff check src/`), line-length 120, `select = ["E", "F", "W"]`.
- No pytest. Tests are inline `if __name__ == "__main__"` sanity blocks run with `.venv/bin/python -m src.data.<module>` from the repo root (module form so `src.*` imports resolve).
- Schema (every record): `{id, company, title, location, description, url, source}`. `id` is `"{source}:{ats_id}:{posting_id}"` for greenhouse/lever/ashby, `"remoteok:{posting_id}"` for RemoteOK. `source` ∈ `greenhouse|lever|ashby|remoteok`. Drop records with empty title or empty description; `location`/`url` may be empty strings.
- `data/` is gitignored: raw cache (`data/raw/…`) and `data/postings.jsonl` are never committed. The company list IS committed (it's source code).
- End every commit message with:
  `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`
  `Claude-Session: https://claude.ai/code/session_014UxjBhGiKMFHbz5fgwuhF8`

---

## Task 1: Feature branch + curated company list

**Files:**
- Create: `src/data/companies.py`
- Modify: `.gitignore` (add `.ruff_cache/`)

**Interfaces:**
- Consumes: nothing.
- Produces: `COMPANIES: list[dict]` — each `{"name": str, "source": str, "ats_id": str}`, `source` ∈ `{greenhouse, lever, ashby}` (RemoteOK is feed-level, not per-company). Tasks 2–4 read this shape; `collect.py` iterates it.

- [ ] **Step 1: Create the branch and tidy `.gitignore`**

```bash
cd /Users/pranavnair/Desktop/two_tower
git checkout -b phase0-data
printf "\n# lint cache\n.ruff_cache/\n" >> .gitignore
```

- [ ] **Step 2: Write `src/data/companies.py` with the sanity block and an EMPTY list**

```python
"""Phase 0 — curated target companies and their ATS board IDs.

ATS IDs come from model knowledge with a cutoff: some WILL be stale (renamed
boards, platform switches). collect.py verifies at fetch time and reports
which failed — expected hit rate ~60-80%, which still clears 1,000 postings.
RemoteOK is not listed here: it is one aggregate feed, fetched separately.
"""

COMPANIES: list = []


if __name__ == "__main__":
    assert len(COMPANIES) >= 100, f"want >=100 companies, have {len(COMPANIES)}"
    keys = [(c["source"], c["ats_id"]) for c in COMPANIES]
    assert len(keys) == len(set(keys)), "duplicate (source, ats_id) entries"
    assert all(c["source"] in {"greenhouse", "lever", "ashby"} for c in COMPANIES)
    assert all(c["name"].strip() and c["ats_id"].strip() for c in COMPANIES)
    per = {s: sum(1 for c in COMPANIES if c["source"] == s) for s in ("greenhouse", "lever", "ashby")}
    print(f"company list       : {len(COMPANIES)} companies {per}   OK")
```

- [ ] **Step 3: Run it — confirm it FAILS on the count**

Run: `.venv/bin/python -m src.data.companies`
Expected: `AssertionError: want >=100 companies, have 0`

- [ ] **Step 4: Fill in the curated list**

Replace `COMPANIES: list = []` with the full list:

```python
COMPANIES: list = [
    # ── Greenhouse ──────────────────────────────────────────────
    {"name": "Anthropic", "source": "greenhouse", "ats_id": "anthropic"},
    {"name": "OpenAI", "source": "greenhouse", "ats_id": "openai"},
    {"name": "Scale AI", "source": "greenhouse", "ats_id": "scaleai"},
    {"name": "Databricks", "source": "greenhouse", "ats_id": "databricks"},
    {"name": "Stripe", "source": "greenhouse", "ats_id": "stripe"},
    {"name": "Figma", "source": "greenhouse", "ats_id": "figma"},
    {"name": "Notion", "source": "greenhouse", "ats_id": "notion"},
    {"name": "Duolingo", "source": "greenhouse", "ats_id": "duolingo"},
    {"name": "Robinhood", "source": "greenhouse", "ats_id": "robinhood"},
    {"name": "Coinbase", "source": "greenhouse", "ats_id": "coinbase"},
    {"name": "Instacart", "source": "greenhouse", "ats_id": "instacart"},
    {"name": "DoorDash", "source": "greenhouse", "ats_id": "doordash"},
    {"name": "Affirm", "source": "greenhouse", "ats_id": "affirm"},
    {"name": "Samsara", "source": "greenhouse", "ats_id": "samsara"},
    {"name": "Sourcegraph", "source": "greenhouse", "ats_id": "sourcegraph"},
    {"name": "Vercel", "source": "greenhouse", "ats_id": "vercel"},
    {"name": "MongoDB", "source": "greenhouse", "ats_id": "mongodb"},
    {"name": "Elastic", "source": "greenhouse", "ats_id": "elastic"},
    {"name": "Cloudflare", "source": "greenhouse", "ats_id": "cloudflare"},
    {"name": "Twilio", "source": "greenhouse", "ats_id": "twilio"},
    {"name": "Asana", "source": "greenhouse", "ats_id": "asana"},
    {"name": "Brex", "source": "greenhouse", "ats_id": "brex"},
    {"name": "Gusto", "source": "greenhouse", "ats_id": "gusto"},
    {"name": "Reddit", "source": "greenhouse", "ats_id": "reddit"},
    {"name": "Discord", "source": "greenhouse", "ats_id": "discord"},
    {"name": "Roblox", "source": "greenhouse", "ats_id": "roblox"},
    {"name": "HashiCorp", "source": "greenhouse", "ats_id": "hashicorp"},
    {"name": "Grammarly", "source": "greenhouse", "ats_id": "grammarly"},
    {"name": "Intercom", "source": "greenhouse", "ats_id": "intercom"},
    {"name": "Amplitude", "source": "greenhouse", "ats_id": "amplitude"},
    {"name": "Mixpanel", "source": "greenhouse", "ats_id": "mixpanel"},
    {"name": "Postman", "source": "greenhouse", "ats_id": "postman"},
    {"name": "Runway", "source": "greenhouse", "ats_id": "runwayml"},
    {"name": "Cruise", "source": "greenhouse", "ats_id": "cruise"},
    {"name": "Nuro", "source": "greenhouse", "ats_id": "nuro"},
    {"name": "Applied Intuition", "source": "greenhouse", "ats_id": "appliedintuition"},
    {"name": "Tempus", "source": "greenhouse", "ats_id": "tempus"},
    {"name": "Glean", "source": "greenhouse", "ats_id": "gleanwork"},
    {"name": "Together AI", "source": "greenhouse", "ats_id": "togetherai"},
    {"name": "Groq", "source": "greenhouse", "ats_id": "groq"},
    {"name": "SambaNova", "source": "greenhouse", "ats_id": "sambanovasystems"},
    {"name": "CoreWeave", "source": "greenhouse", "ats_id": "coreweave"},
    {"name": "Pinecone", "source": "greenhouse", "ats_id": "pinecone"},
    {"name": "AssemblyAI", "source": "greenhouse", "ats_id": "assemblyai"},
    {"name": "Synthesia", "source": "greenhouse", "ats_id": "synthesia"},
    {"name": "Anduril", "source": "greenhouse", "ats_id": "andurilindustries"},
    {"name": "Shield AI", "source": "greenhouse", "ats_id": "shieldai"},
    {"name": "Skydio", "source": "greenhouse", "ats_id": "skydio"},
    {"name": "Webflow", "source": "greenhouse", "ats_id": "webflow"},
    {"name": "Upstart", "source": "greenhouse", "ats_id": "upstart"},
    {"name": "Ripple", "source": "greenhouse", "ats_id": "ripple"},
    {"name": "Chainalysis", "source": "greenhouse", "ats_id": "chainalysis"},
    {"name": "Lambda", "source": "greenhouse", "ats_id": "lambdalabs"},
    {"name": "Perplexity", "source": "greenhouse", "ats_id": "perplexityai"},
    {"name": "Airtable", "source": "greenhouse", "ats_id": "airtable"},
    {"name": "Benchling", "source": "greenhouse", "ats_id": "benchling"},
    {"name": "Checkr", "source": "greenhouse", "ats_id": "checkr"},
    {"name": "Flexport", "source": "greenhouse", "ats_id": "flexport"},
    {"name": "GitLab", "source": "greenhouse", "ats_id": "gitlab"},
    {"name": "Dropbox", "source": "greenhouse", "ats_id": "dropbox"},
    {"name": "Pinterest", "source": "greenhouse", "ats_id": "pinterest"},
    {"name": "Lyft", "source": "greenhouse", "ats_id": "lyft"},
    {"name": "Airbnb", "source": "greenhouse", "ats_id": "airbnb"},
    {"name": "Datadog", "source": "greenhouse", "ats_id": "datadog"},
    {"name": "Hugging Face", "source": "greenhouse", "ats_id": "huggingface"},
    {"name": "Stability AI", "source": "greenhouse", "ats_id": "stabilityai"},
    {"name": "Wayve", "source": "greenhouse", "ats_id": "wayve"},
    {"name": "Aurora", "source": "greenhouse", "ats_id": "aurorainnovation"},
    {"name": "Motional", "source": "greenhouse", "ats_id": "motional"},
    {"name": "Faire", "source": "greenhouse", "ats_id": "faire"},
    {"name": "Gong", "source": "greenhouse", "ats_id": "gong"},
    {"name": "Attentive", "source": "greenhouse", "ats_id": "attentive"},
    {"name": "Abnormal Security", "source": "greenhouse", "ats_id": "abnormalsecurity"},
    {"name": "Snyk", "source": "greenhouse", "ats_id": "snyk"},
    {"name": "Neuralink", "source": "greenhouse", "ats_id": "neuralink"},
    {"name": "Zipline", "source": "greenhouse", "ats_id": "flyzipline"},
    {"name": "Docker", "source": "greenhouse", "ats_id": "docker"},
    {"name": "Rippling", "source": "greenhouse", "ats_id": "rippling"},
    # ── Lever ───────────────────────────────────────────────────
    {"name": "Cohere", "source": "lever", "ats_id": "cohere"},
    {"name": "Mistral AI", "source": "lever", "ats_id": "mistral"},
    {"name": "Palantir", "source": "lever", "ats_id": "palantir"},
    {"name": "Plaid", "source": "lever", "ats_id": "plaid"},
    {"name": "Spotify", "source": "lever", "ats_id": "spotify"},
    {"name": "Canva", "source": "lever", "ats_id": "canva"},
    {"name": "Whatnot", "source": "lever", "ats_id": "whatnot"},
    {"name": "Highspot", "source": "lever", "ats_id": "highspot"},
    {"name": "Voleon", "source": "lever", "ats_id": "voleon"},
    {"name": "Zoox", "source": "lever", "ats_id": "zoox"},
    {"name": "Cerebras", "source": "lever", "ats_id": "cerebras"},
    {"name": "Weights & Biases", "source": "lever", "ats_id": "wandb"},
    {"name": "Netlify", "source": "lever", "ats_id": "netlify"},
    {"name": "Mux", "source": "lever", "ats_id": "mux"},
    # ── Ashby ───────────────────────────────────────────────────
    {"name": "Ramp", "source": "ashby", "ats_id": "ramp"},
    {"name": "Linear", "source": "ashby", "ats_id": "linear"},
    {"name": "Replit", "source": "ashby", "ats_id": "replit"},
    {"name": "Modal", "source": "ashby", "ats_id": "modal"},
    {"name": "ElevenLabs", "source": "ashby", "ats_id": "elevenlabs"},
    {"name": "Writer", "source": "ashby", "ats_id": "writer"},
    {"name": "Sierra", "source": "ashby", "ats_id": "sierra"},
    {"name": "Supabase", "source": "ashby", "ats_id": "supabase"},
    {"name": "PostHog", "source": "ashby", "ats_id": "posthog"},
    {"name": "Cognition", "source": "ashby", "ats_id": "cognition"},
    {"name": "Mercor", "source": "ashby", "ats_id": "mercor"},
    {"name": "Vanta", "source": "ashby", "ats_id": "vanta"},
    {"name": "Deel", "source": "ashby", "ats_id": "deel"},
    {"name": "Clay", "source": "ashby", "ats_id": "clay"},
    {"name": "Cursor", "source": "ashby", "ats_id": "cursor"},
    {"name": "Harvey", "source": "ashby", "ats_id": "harvey"},
    {"name": "Decagon", "source": "ashby", "ats_id": "decagon"},
    {"name": "Browserbase", "source": "ashby", "ats_id": "browserbase"},
    {"name": "LangChain", "source": "ashby", "ats_id": "langchain"},
    {"name": "Astral", "source": "ashby", "ats_id": "astral"},
    {"name": "Baseten", "source": "ashby", "ats_id": "baseten"},
]
```

- [ ] **Step 5: Run it — confirm it PASSES**

Run: `.venv/bin/python -m src.data.companies`
Expected: `company list       : 113 companies {'greenhouse': 78, 'lever': 14, 'ashby': 21}   OK`

- [ ] **Step 6: Lint + commit**

```bash
.venv/bin/ruff check src/
git add .gitignore src/data/companies.py
git commit -m "feat(phase0): curated company list (113 AI/ML companies across 3 ATS platforms)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_014UxjBhGiKMFHbz5fgwuhF8"
```

---

## Task 2: Normalizer (`src/data/normalize.py`)

**Files:**
- Modify: `src/data/normalize.py` (currently a one-line stub)

**Interfaces:**
- Consumes: raw payloads shaped like each API's real response (fetched by Task 3, but tested here with inline samples).
- Produces (Task 4 calls all of these):
  - `html_to_text(raw: str) -> str`
  - `normalize_greenhouse(company: str, ats_id: str, payload: dict) -> list[dict]`
  - `normalize_lever(company: str, ats_id: str, payload: list) -> list[dict]`
  - `normalize_ashby(company: str, ats_id: str, payload: dict) -> list[dict]`
  - `normalize_remoteok(payload: list) -> list[dict]`
  - `dedupe(records: list[dict]) -> list[dict]` — first occurrence wins on normalized `(company, title, location)`.

- [ ] **Step 1: Replace the stub with docstring + imports + sanity block ONLY**

```python
"""Phase 0 — normalize each source's shape into one JSONL schema.

One record: {id, company, title, location, description, url, source}.
Descriptions: full plain text (HTML stripped, entities unescaped, whitespace
normalized). Records missing a title or description are dropped. Dedupe keys
on normalized (company, title, location), first seen wins.
"""

import html as html_lib
import re
from html.parser import HTMLParser


if __name__ == "__main__":
    # --- html_to_text: tags stripped, entities unescaped, whitespace collapsed ---
    got = html_to_text("<p>We build &amp; ship.</p><script>bad()</script><ul><li>fast</li></ul>")
    assert got == "We build & ship.\nfast", repr(got)
    print("html_to_text       : tags/script gone, entities unescaped, whitespace tidy   OK")

    # --- greenhouse: escaped HTML content, nested location ---
    gh = {"jobs": [{"id": 4012345, "title": "ML Engineer", "absolute_url": "https://boards.greenhouse.io/acme/jobs/4012345",
                    "location": {"name": "SF, CA"}, "content": "&lt;p&gt;We build &amp;amp; ship models.&lt;/p&gt;"},
                   {"id": 4012346, "title": "No Desc", "absolute_url": "u", "location": {"name": "NY"}, "content": ""}]}
    rows = normalize_greenhouse("Acme", "acme", gh)
    assert len(rows) == 1, "empty-description job must be dropped"
    r = rows[0]
    assert r == {"id": "greenhouse:acme:4012345", "company": "Acme", "title": "ML Engineer", "location": "SF, CA",
                 "description": "We build & ship models.", "url": "https://boards.greenhouse.io/acme/jobs/4012345",
                 "source": "greenhouse"}, r
    print("greenhouse         : escaped HTML -> text, schema exact, empty desc dropped   OK")

    # --- lever: top-level array, descriptionPlain preferred as-is ---
    lv = [{"id": "abc-123", "text": "Research Engineer", "hostedUrl": "https://jobs.lever.co/acme/abc-123",
           "categories": {"location": "Toronto, ON"}, "descriptionPlain": "Join us.\n\nBuild encoders."}]
    r = normalize_lever("Acme", "acme", lv)[0]
    assert r["id"] == "lever:acme:abc-123" and r["description"] == "Join us.\n\nBuild encoders."
    assert r["location"] == "Toronto, ON" and r["source"] == "lever"
    print("lever              : descriptionPlain used verbatim, schema OK   OK")

    # --- ashby: descriptionPlain preferred over descriptionHtml ---
    ab = {"jobs": [{"id": "d1e2", "title": "AI Engineer", "location": "Remote",
                    "jobUrl": "https://jobs.ashbyhq.com/acme/d1e2",
                    "descriptionPlain": "Plain wins.", "descriptionHtml": "<p>HTML loses.</p>"}]}
    r = normalize_ashby("Acme", "acme", ab)[0]
    assert r["id"] == "ashby:acme:d1e2" and r["description"] == "Plain wins." and r["source"] == "ashby"
    print("ashby              : plain preferred over HTML, schema OK   OK")

    # --- remoteok: element 0 is legal notice, jobs carry own company ---
    rk = [{"legal": "API terms..."},
          {"id": "123456", "position": "ML Engineer", "company": "Beta AI", "location": "Worldwide",
           "url": "https://remoteok.com/remote-jobs/123456", "description": "<p>Train models &amp; ship.</p>"}]
    rows = normalize_remoteok(rk)
    assert len(rows) == 1 and rows[0]["id"] == "remoteok:123456"
    assert rows[0]["company"] == "Beta AI" and rows[0]["description"] == "Train models & ship."
    print("remoteok           : legal entry skipped, own company field used   OK")

    # --- dedupe: same (company, title, location) collapses, first wins ---
    a = {"id": "x1", "company": "Acme", "title": "ML  Engineer", "location": "SF, CA"}
    b = {"id": "x2", "company": "acme", "title": "ml engineer", "location": "sf, ca"}
    c = {"id": "x3", "company": "Acme", "title": "ML Engineer", "location": "NY"}
    out = dedupe([a, b, c])
    assert [r["id"] for r in out] == ["x1", "x3"], out
    print("dedupe             : case/spacing-insensitive, first wins, distinct kept   OK")

    print("\nall normalize sanity checks passed.")
```

- [ ] **Step 2: Run — confirm FAIL**

Run: `.venv/bin/python -m src.data.normalize`
Expected: `NameError: name 'html_to_text' is not defined`

- [ ] **Step 3: Implement, inserted between imports and the sanity block**

```python
_BLOCK_TAGS = {"p", "br", "div", "li", "ul", "ol", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "table", "section"}


class _TextExtractor(HTMLParser):
    """Collects text nodes; block-level tags become newlines; script/style skipped."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self._skip = False

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self._skip = True
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self._skip = False

    def handle_data(self, data):
        if not self._skip:
            self.parts.append(data)


def html_to_text(raw: str) -> str:
    """HTML -> plain text: tags stripped, entities unescaped, whitespace normalized."""
    parser = _TextExtractor()
    parser.feed(raw)
    parser.close()
    text = "".join(parser.parts)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def _record(id_, company, title, location, description, url, source):
    """Build one schema record; returns None if title or description is empty."""
    title, description = title.strip(), description.strip()
    if not (title and description):
        return None
    return {"id": id_, "company": company, "title": title, "location": (location or "").strip(),
            "description": description, "url": url or "", "source": source}


def normalize_greenhouse(company: str, ats_id: str, payload: dict) -> list:
    """Greenhouse: jobs[]; content is HTML-entity-ESCAPED HTML; location nested."""
    rows = []
    for job in payload.get("jobs", []):
        desc = html_to_text(html_lib.unescape(job.get("content") or ""))
        rec = _record(f"greenhouse:{ats_id}:{job['id']}", company, job.get("title") or "",
                      (job.get("location") or {}).get("name"), desc, job.get("absolute_url"), "greenhouse")
        if rec:
            rows.append(rec)
    return rows


def normalize_lever(company: str, ats_id: str, payload: list) -> list:
    """Lever: top-level array; descriptionPlain is already plain text — use verbatim."""
    rows = []
    for job in payload:
        desc = (job.get("descriptionPlain") or "").strip() or html_to_text(job.get("description") or "")
        rec = _record(f"lever:{ats_id}:{job['id']}", company, job.get("text") or "",
                      (job.get("categories") or {}).get("location"), desc, job.get("hostedUrl"), "lever")
        if rec:
            rows.append(rec)
    return rows


def normalize_ashby(company: str, ats_id: str, payload: dict) -> list:
    """Ashby: jobs[]; prefer descriptionPlain when present, else strip descriptionHtml."""
    rows = []
    for job in payload.get("jobs", []):
        desc = (job.get("descriptionPlain") or "").strip() or html_to_text(job.get("descriptionHtml") or "")
        rec = _record(f"ashby:{ats_id}:{job['id']}", company, job.get("title") or "",
                      job.get("location"), desc, job.get("jobUrl") or job.get("applyUrl"), "ashby")
        if rec:
            rows.append(rec)
    return rows


def normalize_remoteok(payload: list) -> list:
    """RemoteOK: one aggregate feed; element 0 is a legal notice; jobs carry their own company."""
    rows = []
    for item in payload:
        if not isinstance(item, dict) or not item.get("id") or not item.get("position"):
            continue  # legal notice / malformed entries
        company = (item.get("company") or "").strip()
        if not company:
            continue
        rec = _record(f"remoteok:{item['id']}", company, item.get("position") or "",
                      item.get("location"), html_to_text(item.get("description") or ""), item.get("url"), "remoteok")
        if rec:
            rows.append(rec)
    return rows


def _dedupe_key(r: dict) -> tuple:
    squash = lambda s: re.sub(r"\s+", " ", s.lower()).strip()  # noqa: E731
    return (squash(r["company"]), squash(r["title"]), squash(r.get("location") or ""))


def dedupe(records: list) -> list:
    """Drop repeats of the same normalized (company, title, location); first seen wins."""
    seen, out = set(), []
    for r in records:
        key = _dedupe_key(r)
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out
```

- [ ] **Step 4: Run — confirm PASS**

Run: `.venv/bin/python -m src.data.normalize`
Expected: six `OK` lines then `all normalize sanity checks passed.`

- [ ] **Step 5: Lint + commit**

```bash
.venv/bin/ruff check src/
git add src/data/normalize.py
git commit -m "feat(phase0): normalizers — 4 source shapes to one schema, html-to-text, dedupe

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_014UxjBhGiKMFHbz5fgwuhF8"
```

---

## Task 3: Fetchers with cache + politeness (`src/data/fetch.py`)

**Files:**
- Modify: `src/data/fetch.py` (currently a one-line stub)
- Modify: `requirements.txt` (uncomment `requests`)

**Interfaces:**
- Consumes: `requests` (installed this task). No other project modules.
- Produces (Task 4 calls these; all return the parsed JSON payload or `None` on any failure):
  - `fetch_greenhouse(ats_id: str, raw_dir=RAW_DIR, refresh: bool = False) -> Optional[dict]`
  - `fetch_lever(ats_id: str, raw_dir=RAW_DIR, refresh: bool = False) -> Optional[list]`
  - `fetch_ashby(ats_id: str, raw_dir=RAW_DIR, refresh: bool = False) -> Optional[dict]`
  - `fetch_remoteok(raw_dir=RAW_DIR, refresh: bool = False) -> Optional[list]`
  - Cache layout: `{raw_dir}/{source}/{ats_id}.json`, RemoteOK at `{raw_dir}/remoteok.json`. Cache-first unless `refresh`.

- [ ] **Step 1: Install the new dependency**

In `requirements.txt`, change the line `# requests            # ATS API calls (Greenhouse / Lever / Ashby / RemoteOK)` to `requests       # ATS API calls (Greenhouse / Lever / Ashby / RemoteOK)` (keep it in the Phase 0 section). Then:

```bash
.venv/bin/pip install requests
.venv/bin/python -c "import requests; print('requests', requests.__version__)"
```
Expected: installs cleanly, prints a version.

- [ ] **Step 2: Replace the stub with docstring + imports + sanity block ONLY**

```python
"""Phase 0 — ATS fetchers (Greenhouse / Lever / Ashby / RemoteOK) -> raw JSON cache.

Cache-first: a fetched payload is written to data/raw/... and reused on every
later run unless refresh=True. Politeness: real User-Agent, a fixed delay after
each LIVE request, hard timeout. Any failure (HTTP error, timeout, bad JSON)
returns None — collect.py logs the company and moves on.
"""

import json
import time
from pathlib import Path
from typing import Optional  # noqa: F401  (used in signatures)

import requests


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        raw = Path(td)

        # --- cache-first: planted cache is returned, no network touched ---
        (raw / "greenhouse").mkdir(parents=True)
        (raw / "greenhouse" / "acme.json").write_text(json.dumps({"jobs": [{"id": 1}]}))
        assert fetch_greenhouse("acme", raw_dir=raw) == {"jobs": [{"id": 1}]}
        print("cache-first check  : planted cache returned as-is   OK")

        # --- failure path: unresolvable host -> None, and nothing cached ---
        assert _get_json("https://nonexistent.invalid/x", raw / "bad.json") is None
        assert not (raw / "bad.json").exists()
        print("failure check      : bad host -> None, no cache file written   OK")

        # --- cache write path: successful fetch persists (simulated via planted write) ---
        (raw / "remoteok.json").write_text(json.dumps([{"legal": "..."}]))
        assert fetch_remoteok(raw_dir=raw) == [{"legal": "..."}]
        print("layout check       : remoteok cache path honored   OK")

    print("\nall fetch sanity checks passed (offline).")
```

- [ ] **Step 3: Run — confirm FAIL**

Run: `.venv/bin/python -m src.data.fetch`
Expected: `NameError: name 'fetch_greenhouse' is not defined`

- [ ] **Step 4: Implement, inserted between imports and the sanity block**

```python
RAW_DIR = Path("data/raw")
USER_AGENT = "job-matcher-research/0.1 (github.com/pnair0703/two-tower-job-matcher)"
REQUEST_DELAY_S = 0.6   # pause after every live request — be polite
TIMEOUT_S = 20


def _get_json(url: str, cache_path: Path, refresh: bool = False):
    """Cache-first JSON GET. Returns parsed payload, or None on any failure."""
    cache_path = Path(cache_path)
    if cache_path.exists() and not refresh:
        return json.loads(cache_path.read_text())
    try:
        resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT_S)
        time.sleep(REQUEST_DELAY_S)
        resp.raise_for_status()
        payload = resp.json()
    except (requests.RequestException, ValueError):
        return None
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload))
    return payload


def fetch_greenhouse(ats_id: str, raw_dir=RAW_DIR, refresh: bool = False) -> Optional[dict]:
    url = f"https://boards-api.greenhouse.io/v1/boards/{ats_id}/jobs?content=true"
    return _get_json(url, Path(raw_dir) / "greenhouse" / f"{ats_id}.json", refresh)


def fetch_lever(ats_id: str, raw_dir=RAW_DIR, refresh: bool = False) -> Optional[list]:
    url = f"https://api.lever.co/v0/postings/{ats_id}?mode=json"
    return _get_json(url, Path(raw_dir) / "lever" / f"{ats_id}.json", refresh)


def fetch_ashby(ats_id: str, raw_dir=RAW_DIR, refresh: bool = False) -> Optional[dict]:
    url = f"https://api.ashbyhq.com/posting-api/job-board/{ats_id}"
    return _get_json(url, Path(raw_dir) / "ashby" / f"{ats_id}.json", refresh)


def fetch_remoteok(raw_dir=RAW_DIR, refresh: bool = False) -> Optional[list]:
    return _get_json("https://remoteok.com/api", Path(raw_dir) / "remoteok.json", refresh)
```

- [ ] **Step 5: Run — confirm PASS (offline)**

Run: `.venv/bin/python -m src.data.fetch`
Expected: three `OK` lines then `all fetch sanity checks passed (offline).`

- [ ] **Step 6: Lint + commit**

```bash
.venv/bin/ruff check src/
git add src/data/fetch.py requirements.txt
git commit -m "feat(phase0): cache-first polite fetchers for 4 ATS sources

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_014UxjBhGiKMFHbz5fgwuhF8"
```

---

## Task 4: Pipeline entry point + live collection run (`src/data/collect.py`)

**Files:**
- Create: `src/data/collect.py`

**Interfaces:**
- Consumes: `COMPANIES` (Task 1), all `fetch_*` (Task 3), all `normalize_*` + `dedupe` (Task 2).
- Produces: `data/postings.jsonl` (gitignored artifact) and a printed run summary. CLI: `--refresh` (ignore cache), `--limit N` (first N companies — smoke runs), `--out PATH` (default `data/postings.jsonl`).

- [ ] **Step 1: Write `src/data/collect.py`**

```python
"""Phase 0 — run the pipeline: fetch all sources, normalize, dedupe, write JSONL.

Usage (from repo root):
    .venv/bin/python -m src.data.collect               # cache-first full run
    .venv/bin/python -m src.data.collect --limit 5     # smoke: first 5 companies
    .venv/bin/python -m src.data.collect --refresh     # ignore cache, re-hit APIs
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from src.data import fetch, normalize
from src.data.companies import COMPANIES

_FETCH = {"greenhouse": fetch.fetch_greenhouse, "lever": fetch.fetch_lever, "ashby": fetch.fetch_ashby}
_NORM = {"greenhouse": normalize.normalize_greenhouse, "lever": normalize.normalize_lever,
         "ashby": normalize.normalize_ashby}


def collect(companies: list, refresh: bool = False) -> tuple:
    """Fetch + normalize every company and RemoteOK. Returns (records, failed_labels)."""
    records, failed = [], []
    for c in companies:
        payload = _FETCH[c["source"]](c["ats_id"], refresh=refresh)
        if payload is None:
            failed.append(f"{c['name']} ({c['source']}:{c['ats_id']})")
            continue
        rows = _NORM[c["source"]](c["name"], c["ats_id"], payload)
        records.extend(rows)
        print(f"  {c['name']:<22} {c['source']:<10} {len(rows):>4} postings")
    payload = fetch.fetch_remoteok(refresh=refresh)
    if payload is None:
        failed.append("RemoteOK (feed)")
    else:
        rows = normalize.normalize_remoteok(payload)
        records.extend(rows)
        print(f"  {'(aggregate feed)':<22} {'remoteok':<10} {len(rows):>4} postings")
    return records, failed


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--refresh", action="store_true", help="ignore cache, re-hit the APIs")
    ap.add_argument("--limit", type=int, default=None, help="only the first N companies (smoke run)")
    ap.add_argument("--out", default="data/postings.jsonl")
    args = ap.parse_args()

    companies = COMPANIES[: args.limit] if args.limit else COMPANIES
    records, failed = collect(companies, refresh=args.refresh)
    unique = normalize.dedupe(records)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for r in unique:
            f.write(json.dumps(r) + "\n")

    per_source = Counter(r["source"] for r in unique)
    print("\n── summary ────────────────────────────────")
    print(f"companies tried    : {len(companies)} (+ RemoteOK feed)")
    print(f"responded          : {len(companies) - sum(1 for x in failed if not x.startswith('RemoteOK'))}")
    print(f"failed             : {len(failed)}")
    for label in failed:
        print(f"    - {label}")
    print(f"postings fetched   : {len(records)}")
    print(f"after dedupe       : {len(unique)}  ({len(records) - len(unique)} duplicates removed)")
    for src, n in per_source.most_common():
        print(f"    {src:<10} {n:>5}")
    print(f"written            : {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke run — 5 companies, live**

Run: `.venv/bin/python -m src.data.collect --limit 5 --out data/postings_smoke.jsonl`
Expected: per-company lines print (some may fail — fine), RemoteOK fetches, a summary prints, and `data/postings_smoke.jsonl` exists with >0 lines. This is the first live network use; if EVERY source fails, STOP — check connectivity before proceeding.

- [ ] **Step 3: Inspect one record for schema + plain text**

Run: `.venv/bin/python -c "import json; r=json.loads(open('data/postings_smoke.jsonl').readline()); print(sorted(r.keys())); print(r['id'], '|', r['title'][:40]); print(r['description'][:200])"`
Expected: keys `['company','description','id','location','source','title','url']`; description reads as clean prose (no `<p>`, no `&amp;`). If HTML artifacts appear, STOP and fix `html_to_text` before the full run (this is the spec's stdlib-vs-beautifulsoup decision point).

- [ ] **Step 4: Full live run**

Run: `.venv/bin/python -m src.data.collect`
Expected: takes a few minutes (~113 companies × ~1s politeness). Summary shows failed companies (expected: a decent handful — stale IDs), total after dedupe **≥ 1,000**, from **≥ 2** sources. If under 1,000, STOP and report — we extend the company list rather than lower the bar.

- [ ] **Step 5: Clean up smoke artifact, lint, commit**

```bash
rm data/postings_smoke.jsonl
.venv/bin/ruff check src/
git status --short   # must show ONLY src/data/collect.py — data/ stays untracked
git add src/data/collect.py
git commit -m "feat(phase0): collect pipeline — fetch, normalize, dedupe, write postings.jsonl

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_014UxjBhGiKMFHbz5fgwuhF8"
```

---

## Task 5: Prune dead companies + README status

**Files:**
- Modify: `src/data/companies.py` (remove/fix entries the run proved dead, ONLY if needed to hold ≥1,000)
- Modify: `README.md` (flip Phase 0 status)

**Interfaces:**
- Consumes: Task 4's run summary (the failed list).
- Produces: the final Phase 0 state on the branch, ready for review/merge.

- [ ] **Step 1: Review the failed-company list from Task 4's summary**

If total unique postings ≥ 1,000: leave the list as-is (failed entries cost one polite request per full refresh — harmless documentation of the attempt). If < 1,000: replace failed entries with additional companies and re-run `collect` (cache makes this cheap — only new companies hit the network).

- [ ] **Step 2: Flip README status**

In `README.md`, change `- [ ] Phase 0 — data collection (ATS APIs → JSONL)` to `- [x] Phase 0 — data collection (ATS APIs → JSONL)`.

- [ ] **Step 3: Final verification + commit**

```bash
.venv/bin/python -m src.data.companies
.venv/bin/python -m src.data.normalize
.venv/bin/python -m src.data.fetch
.venv/bin/ruff check src/
wc -l data/postings.jsonl   # expect >= 1000
git add README.md src/data/companies.py
git commit -m "chore(phase0): mark phase 0 complete

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_014UxjBhGiKMFHbz5fgwuhF8"
```

- [ ] **Step 4: Finish the branch**

Use the superpowers:finishing-a-development-branch skill: run all sanity checks one last time, then present merge/PR options for `phase0-data` → `main`.

---

## Self-Review

**1. Spec coverage:** curated list → Task 1; fetchers + cache + politeness + non-fatal failures → Task 3; normalize (escaped Greenhouse HTML, Lever plain-verbatim, Ashby plain-preferred, RemoteOK legal-skip + own company) → Task 2; keep-all-roles (no filtering anywhere) ✓; full-plain-text (no truncation anywhere) ✓; dedupe first-wins → Task 2; run entry + summary + `--refresh` → Task 4; ≥1,000 from ≥2 sources → Task 4 Step 4 + Task 5; stale-ID reality → Task 5 Step 1; `requests` dep → Task 3 Step 1; feature branch → Task 1/Global; stdlib-vs-bs4 decision point → Task 4 Step 3. No gaps.

**2. Placeholder scan:** no TBD/TODO; all code complete; expected outputs stated for every run step. ✓

**3. Type consistency:** `COMPANIES` dicts (`name/source/ats_id`) match `collect()`'s access; `fetch_*` signatures (`ats_id, raw_dir, refresh`) match sanity usage and `_FETCH` dispatch (keyword `refresh=` only — `raw_dir` defaults); `normalize_*` `(company, ats_id, payload)` and `normalize_remoteok(payload)` match `_NORM` dispatch and sanity calls; `dedupe(records)` consistent. `Optional` used (3.9-safe). ✓
