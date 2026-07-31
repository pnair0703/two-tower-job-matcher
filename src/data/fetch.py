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
