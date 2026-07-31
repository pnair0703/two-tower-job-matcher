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
