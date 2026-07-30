"""Phase 0 — normalize each source's shape into one JSONL schema.

One record: {id, company, title, location, description, url, source}.
Descriptions: full plain text (HTML stripped, entities unescaped, whitespace
normalized). Records missing a title or description are dropped. Dedupe keys
on normalized (company, title, location), first seen wins.
"""

import html as html_lib
import re
from html.parser import HTMLParser

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


def _repair_mojibake(text: str) -> str:
    """Undo UTF-8-decoded-as-Latin-1 double-encoding (seen in ~half of RemoteOK's
    descriptions, upstream of us). Round-tripping already-clean text through
    latin-1->utf-8 either fails (any char above codepoint 255 - left untouched)
    or returns the identical string, so this is a safe no-op elsewhere."""
    try:
        return text.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text


def _record(id_, company, title, location, description, url, source):
    """Build one schema record; returns None if title or description is empty."""
    title, description = title.strip(), _repair_mojibake(description).strip()
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
