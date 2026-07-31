# Two-Tower Job Matcher

Retrieval system that ranks scraped job postings against a resume. Two encoders
("towers") map jobs and resumes into a shared vector space; nearest neighbours are
the matches. The encoder is built two ways — a from-scratch transformer and a
fine-tuned pretrained sentence encoder — and both are benchmarked against a keyword
baseline. **The comparison is the deliverable.**

## Results

Held-out test set. Filled in Phase 5.

| System               | P@5 | P@10 | R@20 | NDCG@10 | Latency p50 | Latency p99 |
|----------------------|-----|------|------|---------|-------------|-------------|
| BM25 baseline        | TBD | TBD  | TBD  | TBD     | TBD         | TBD         |
| Tower A (scratch)    | TBD | TBD  | TBD  | TBD     | TBD         | TBD         |
| Tower B (pretrained) | TBD | TBD  | TBD  | TBD     | TBD         | TBD         |

## Status

- [x] Foundation — repo scaffold + bidirectional attention blocks
- [x] Phase 0 — data collection (ATS APIs → JSONL) — 13,073 postings from 81 companies + RemoteOK
- [ ] Phase 1 — labels (LLM scoring + human validation)
- [ ] Phase 2 — Tower A (from-scratch encoder)
- [ ] Phase 3 — Tower B (pretrained encoder)
- [ ] Phase 4 — retrieval + serving (FAISS)
- [ ] Phase 5 — evaluation (three-system comparison)

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Architecture

Two encoders behind one interface. Jobs are embedded offline into a FAISS index;
the resume is embedded at query time; nearest neighbours are returned. Details land
as each phase ships.
