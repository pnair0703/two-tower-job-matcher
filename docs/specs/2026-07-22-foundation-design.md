# Foundation — Design Spec

**Sub-project #0 of the Two-Tower Job Matcher.** Date: 2026-07-22.

This is the first of seven sequential sub-projects (see `BUILD_TWO_TOWER.md` for the
full plan). Each sub-project gets its own spec → plan → build → commit cycle. Foundation
makes the repo skeleton real and lands the reusable, bidirectional attention blocks.

## Goal

Stand up the repository skeleton and vendor the transformer attention code, stripped
down to bidirectional encoder blocks with padding-mask support. Nothing in this
sub-project trains, fetches, or scores anything — it is the foundation the later phases
build on.

## Scope

**In scope**
- Repo scaffold: directory tree, `.gitignore`, `.env.example`, `requirements.txt`,
  `README.md` stub, git repo (already initialized on `main`).
- `src/model/attention.py`: vendored from `../personal_project1/transformer_project/model.py`,
  stripped to `Head`, `MultiHeadAttention`, `FeedForward`, `Block`, made bidirectional
  with an optional padding mask, with an inline sanity-check block.

**Out of scope** (later sub-projects)
- Tokenizer, towers, contrastive loss (Phase 2 / sub-project #3).
- Any data fetching, labeling, indexing, serving, or evaluation.
- Installing the full dependency stack — Foundation installs only `torch` + `ruff`.

## Conventions (matched from the user's existing projects)

- Python **3.9.6** via a local `.venv` (same interpreter as `transformer_project`, so the
  vendored code runs unchanged). Risk: a 2026-era `sentence-transformers` / `faiss-cpu` /
  `numpy` may refuse to install on 3.9 in a later phase; if so, bump to a brew-installed
  3.11/3.12 then.
- **ruff**, line-length 120, `select = ["E", "F", "W"]`.
- `.env` + `.env.example` for secrets (the LLM API key, used in Phase 1). `.env` gitignored.
- Inline `if __name__ == "__main__"` sanity checks (as in `transformer_project`), not pytest.

## Repository layout

Built in-place in `/Users/pranavnair/Desktop/two_tower`. The directory name is cosmetic;
the GitHub remote can be named `job-matcher` later. `BUILD_TWO_TOWER.md` stays at the root
as the reference spec.

```
two_tower/
├── src/
│   ├── __init__.py
│   ├── model/
│   │   ├── __init__.py
│   │   ├── attention.py      # REAL: vendored + stripped bidirectional blocks
│   │   ├── towers.py         # stub — Phase 2
│   │   └── loss.py           # stub — Phase 2
│   ├── data/
│   │   ├── __init__.py
│   │   ├── fetch.py          # stub — Phase 0
│   │   ├── normalize.py      # stub — Phase 0
│   │   └── label.py          # stub — Phase 1
│   ├── eval/
│   │   ├── __init__.py
│   │   ├── metrics.py        # stub — Phase 5
│   │   ├── baseline.py       # stub — Phase 5
│   │   └── run_eval.py       # stub — Phase 5
│   ├── index.py              # stub — Phase 4
│   └── serve.py              # stub — Phase 4
├── data/                     # gitignored — postings, labels, indices
├── notebooks/
├── docs/specs/               # design specs (this file)
├── README.md
├── requirements.txt
├── .gitignore
└── .env.example
```

Stub modules carry a one-line docstring naming their phase (e.g. `"""Phase 0 — ATS
fetchers → raw JSON cache."""`) so the roadmap is visible and package imports don't break.

## The attention strip-down (the core work)

Source: `Head`, `MultiHeadAttention`, `FeedForward`, `Block` from `transformer_project/model.py`.
Deleted: `GPTConfig`, `GPT`, token/position embeddings, generation code.

Three substantive edits:

### 1. Remove causal masking → bidirectional

Delete the `tril` buffer and the `scores.masked_fill(self.tril[:T,:T] == 0, -inf)` line. An
encoder must let every position attend to every other position in both directions. This also
lets `Head`, `MultiHeadAttention`, and `Block` **drop their `block_size` argument** — it
existed only to size the causal buffer. (The position-embedding table that still needs a max
length lives in the tower, a later sub-project, not in attention.)

### 2. Optional padding mask (approach A — boolean key-padding mask)

`forward(x, mask=None)` where `mask` is a boolean tensor of shape `(B, T)`, `True` = real
token, `False` = padding. Batched job/resume texts have different lengths; short ones are
padded, and attention must not let real tokens attend to pad tokens (they would poison every
representation). Implementation: before softmax, `masked_fill` the pad **keys** to `-inf`.

- `scores` is `(B, T_query, T_key)`; the mask `(B, T_key)` expands to `(B, 1, T_key)` and
  masks pad keys across all query rows.
- `mask=None` means "no padding" — used by the sanity check and any equal-length batch.
- The **same** boolean mask is later reused by the towers' mean-pooling so pad positions do
  not drag the average. One mask concept, defined once, threaded through attention → pooling.

Rejected alternatives: **B** additive float mask `(B,1,T)` of `0`/`-inf` (matches
`nn.TransformerEncoder`, but more machinery than needed here); **C** no mask, avoid ragged
batches by bucketing/`batch_size=1` (simplest code, but loses batching efficiency and the
padding interview answer).

### 3. Provenance header + rewritten sanity check

Header comment at the top of `attention.py`:

```python
# Transformer blocks adapted from my from-scratch implementation:
# github.com/pnair0703/<transformer-repo>        # TODO: fill real repo URL
# Architecture reused; weights are NOT — trained fresh on job/resume text.
```

The `if __name__ == "__main__"` block is rewritten to assert the encoder's properties (the
opposite of the causal decoder's):

- **Shape:** `(B,T,C)` in → `(B,T,C)` out for `MultiHeadAttention`.
- **Softmax:** every attention row sums to 1.
- **Bidirectional:** a position's output *does* change when a *later* position's input
  changes (i.e. it attends to the "future") — the inverse of the old no-peeking check.
- **Padding mask:** with a mask marking the last positions as pad, no query places any
  attention weight on those pad keys, and changing a pad token's input leaves all real
  outputs unchanged.

Class name stays `Head` (provenance with the transformer repo), not `SelfAttention`.

## Supporting files

- **`.gitignore`:** `data/`, `*.faiss`, `.env`, `*.pt`, `__pycache__/`, `*.pyc`, `.venv/`,
  `.DS_Store`.
- **`.env.example`:** `LLM_API_KEY=your_key_here` (real key in gitignored `.env`, Phase 1).
- **`requirements.txt`:** lists the full intended stack grouped by phase in comments (torch,
  numpy; requests/httpx; sentence-transformers, transformers; faiss-cpu; rank-bm25 or
  scikit-learn; fastapi, uvicorn; python-dotenv; ruff). Only `torch` + `ruff` are actually
  installed in Foundation; later phases install their own rows.
- **`README.md`:** leads with the three-system results table (all cells `TBD`), per the
  guide's guardrail that the README leads with results, not architecture. Columns: System |
  P@5 | P@10 | R@20 | NDCG@10 | Latency p50 | Latency p99. Rows: BM25 baseline, Tower A,
  Tower B.

## Testing / done criteria

- `python src/model/attention.py` runs and prints all sanity checks passing (shape, softmax,
  bidirectional, padding mask).
- `ruff check src/` is clean.
- Skeleton committed. Commit sequence: (1) this design spec, (2) scaffold + `.gitignore` +
  `.env.example` + `requirements.txt` + README stub + empty structure, (3) vendored
  bidirectional `attention.py`. (Guide's "first commit before any code" = commit 2.)

## Open fill-ins (not blockers)

- Real transformer-repo URL for the provenance header (currently `<transformer-repo>`).
