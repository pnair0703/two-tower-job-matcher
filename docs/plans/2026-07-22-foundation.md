# Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up the repository skeleton and land the vendored transformer attention code, stripped to bidirectional encoder blocks with an optional padding mask.

**Architecture:** Two tasks. Task 1 creates the committed empty skeleton (dir tree, package stubs, `.gitignore`, `.env.example`, `requirements.txt`, README with the blank results table) plus the `.venv`. Task 2 vendors the four attention classes from the from-scratch GPT, removes causal masking, adds a boolean key-padding mask, and proves the encoder properties with an inline sanity block.

**Tech Stack:** Python 3.9.6, PyTorch, ruff. No pytest (repo uses inline `__main__` sanity checks).

## Global Constraints

Every task's requirements implicitly include these:

- **Python 3.9.6** via local `.venv` (same interpreter as `transformer_project`). Use `.venv/bin/python` and `.venv/bin/pip` explicitly — shell activation does not persist across commands here.
- **ruff**, line-length **120**, `select = ["E", "F", "W"]`. `ruff check src/` must be clean.
- **No pytest.** Tests are inline `if __name__ == "__main__"` sanity checks, run with `.venv/bin/python <file>`.
- **Class name `Head`** (provenance with the transformer repo), not `SelfAttention`.
- **Mask contract:** boolean tensor shape `(B, T)`, `True` = real token, `False` = pad. `mask=None` means no padding. Pad **keys** get `-inf` before softmax. This same tensor is reused by tower mean-pooling in a later sub-project.
- Repo built **in-place** in `/Users/pranavnair/Desktop/two_tower`, already `git init`-ed on `main`, remote `origin` = `github.com/pnair0703/two-tower-job-matcher` (public).
- **`BUILD_TWO_TOWER.md` is gitignored** — it must never be committed.
- End every commit message with:
  `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`

---

## Task 1: Repo scaffold (empty structure + venv)

**Files:**
- Create: `.gitignore`, `.env.example`, `requirements.txt`, `README.md`
- Create: `src/__init__.py`, `src/model/__init__.py`, `src/data/__init__.py`, `src/eval/__init__.py`
- Create stubs: `src/model/towers.py`, `src/model/loss.py`, `src/data/fetch.py`, `src/data/normalize.py`, `src/data/label.py`, `src/eval/metrics.py`, `src/eval/baseline.py`, `src/eval/run_eval.py`, `src/index.py`, `src/serve.py`
- Create: `data/.gitkeep`, `notebooks/.gitkeep`
- Create: `.venv/` (via `python3 -m venv`)

**Interfaces:**
- Consumes: nothing (first code task).
- Produces: importable packages `src`, `src.model`, `src.data`, `src.eval`; a working `.venv` with `torch` and `ruff` installed. Task 2 adds `src/model/attention.py` into the `src.model` package.

- [ ] **Step 1: Create the directory tree and the virtualenv**

```bash
cd /Users/pranavnair/Desktop/two_tower
mkdir -p src/model src/data src/eval data notebooks
python3 -m venv .venv
.venv/bin/python --version   # expect: Python 3.9.6
```

- [ ] **Step 2: Write `.gitignore`**

```gitignore
# data + model artifacts
data/
*.faiss
*.pt

# secrets
.env

# python
__pycache__/
*.pyc
.venv/

# os / editor cruft
.DS_Store

# local build guide — reference only, not for the public repo
BUILD_TWO_TOWER.md
```

- [ ] **Step 3: Write `.env.example`**

```dotenv
# LLM API key for Phase 1 batch labeling.
# Copy this file to .env (gitignored) and paste your real key there.
LLM_API_KEY=your_key_here
```

- [ ] **Step 4: Write `requirements.txt`** (only Foundation rows uncommented; later phases uncomment theirs)

```text
# ── Foundation (installed now) ───────────────────────────────
torch          # transformer blocks + all model code (pulls in numpy)
ruff           # linting: line-length 120, select E/F/W

# ── Phase 0: data collection ─────────────────────────────────
# requests            # ATS API calls (Greenhouse / Lever / Ashby / RemoteOK)
# python-dotenv       # load .env

# ── Phase 1: labeling ────────────────────────────────────────
# anthropic           # LLM scoring for label generation

# ── Phase 2/3: encoders ──────────────────────────────────────
# sentence-transformers   # Tower B pretrained encoder
# transformers            # dependency of sentence-transformers

# ── Phase 4: retrieval + serving ─────────────────────────────
# faiss-cpu           # vector index
# fastapi             # serving
# uvicorn             # ASGI server

# ── Phase 5: evaluation ──────────────────────────────────────
# rank-bm25           # BM25 baseline
# scikit-learn        # TF-IDF alternative + metrics helpers
```

- [ ] **Step 5: Install Foundation dependencies**

Run:
```bash
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
```
Expected: `torch` and `ruff` install successfully. If `torch` fails to build/install on 3.9.6, STOP and report — that triggers the Python-version bump noted in the spec.

- [ ] **Step 6: Write `README.md`** (leads with the results table, per the guide's guardrail)

```markdown
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

- [ ] Foundation — repo scaffold + bidirectional attention blocks
- [ ] Phase 0 — data collection (ATS APIs → JSONL)
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
```

- [ ] **Step 7: Write the package `__init__.py` files** (all four empty)

Create `src/__init__.py`, `src/model/__init__.py`, `src/data/__init__.py`, `src/eval/__init__.py` — each an empty file (0 bytes).

```bash
: > src/__init__.py
: > src/model/__init__.py
: > src/data/__init__.py
: > src/eval/__init__.py
```

- [ ] **Step 8: Write the stub modules** (docstring only — names the phase, keeps imports working)

Each file contains exactly one line — its module docstring:

- `src/model/towers.py` → `"""Phase 2 — JobTower + ResumeTower (own encoder), same interface as Tower B."""`
- `src/model/loss.py` → `"""Phase 2 — contrastive loss with in-batch negatives."""`
- `src/data/fetch.py` → `"""Phase 0 — ATS fetchers (Greenhouse / Lever / Ashby / RemoteOK) → raw JSON cache."""`
- `src/data/normalize.py` → `"""Phase 0 — normalize each source's shape into one JSONL schema."""`
- `src/data/label.py` → `"""Phase 1 — LLM scoring + human label merge."""`
- `src/eval/metrics.py` → `"""Phase 5 — precision@k, recall@k, NDCG (hand-written)."""`
- `src/eval/baseline.py` → `"""Phase 5 — BM25 / TF-IDF keyword baseline."""`
- `src/eval/run_eval.py` → `"""Phase 5 — run all three systems, emit one results table."""`
- `src/index.py` → `"""Phase 4 — FAISS index build + query."""`
- `src/serve.py` → `"""Phase 4 — FastAPI or CLI serving."""`

- [ ] **Step 9: Write the `.gitkeep` placeholders** (so empty tracked dirs survive; `data/` itself is gitignored, so keep its `.gitkeep` force-addable later — for now only `notebooks/` needs one)

```bash
: > notebooks/.gitkeep
```
(Do NOT create `data/.gitkeep` — `data/` is gitignored and stays untracked.)

- [ ] **Step 10: Verify the skeleton imports and lints clean**

Run:
```bash
.venv/bin/ruff check src/
.venv/bin/python -c "import src, src.model, src.data, src.eval; print('packages import OK')"
```
Expected: ruff reports no issues; the import prints `packages import OK`.

- [ ] **Step 11: Confirm `BUILD_TWO_TOWER.md` is ignored, then commit**

Run and confirm `BUILD_TWO_TOWER.md` does NOT appear in the staged list:
```bash
git check-ignore BUILD_TWO_TOWER.md   # expect it to echo the filename (ignored)
git add .
git status --short                     # BUILD_TWO_TOWER.md and .DS_Store must NOT be listed
git commit -m "scaffold: empty structure, gitignore, requirements, README stub

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```
Expected: commit succeeds; `git show --stat HEAD` lists the scaffold files but not `BUILD_TWO_TOWER.md`.

---

## Task 2: Bidirectional attention blocks (`attention.py`)

**Files:**
- Create: `src/model/attention.py`
- Modify: `README.md` (flip Foundation status to done)

**Interfaces:**
- Consumes: `.venv` with torch (Task 1).
- Produces the reusable encoder API that Phase 2's towers will import:
  - `Head(n_embd: int, head_size: int, dropout: float = 0.0)` with `forward(x, mask=None) -> Tensor` and `_weights(x, mask=None) -> Tensor`.
  - `MultiHeadAttention(n_embd: int, n_head: int, dropout: float = 0.0)` with `forward(x, mask=None) -> Tensor`.
  - `FeedForward(n_embd: int, dropout: float = 0.0)` with `forward(x) -> Tensor`.
  - `Block(n_embd: int, n_head: int, dropout: float = 0.0)` with `forward(x, mask=None) -> Tensor`.
  - All accept/return `(B, T, n_embd)`; `mask` is `(B, T)` bool, `True` = real token.

- [ ] **Step 1: Write `attention.py` with the header, imports, and the sanity block ONLY (no classes yet)**

Create `src/model/attention.py` with exactly this content (classes deliberately absent so the sanity block fails first):

```python
# Transformer blocks adapted from my from-scratch implementation:
# github.com/pnair0703/<transformer-repo>        # TODO: fill real repo URL
# Architecture reused; weights are NOT — trained fresh on job/resume text.
"""attention.py — bidirectional transformer encoder blocks.

Vendored from the from-scratch GPT (a decoder) and adapted into an ENCODER:

    Head / MultiHeadAttention   self-attention over the whole sequence
    FeedForward                 per-position MLP
    Block                       attention + FFN, residuals + pre-LayerNorm

Two changes from the decoder this came from:
  1. No causal mask. A decoder hides the future so it can predict it; an encoder
     must let every position see every other, both directions, to build the best
     representation. The `tril` buffer and its masked_fill are gone.
  2. Optional padding mask. We batch job/resume texts of different lengths and pad
     the short ones. forward(x, mask=...) takes a boolean (B, T) mask (True = real
     token) and drops attention to pad KEYS so padding can't poison real tokens.
     The same mask is reused downstream by mean-pooling in the towers.

Attention is still written by hand — understanding it is the point.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F


if __name__ == "__main__":
    torch.manual_seed(0)
    B, T, C, n_head = 4, 8, 32, 4
    x = torch.randn(B, T, C)

    # --- shape ---
    mha = MultiHeadAttention(n_embd=C, n_head=n_head)
    out = mha(x)
    assert out.shape == (B, T, C), out.shape
    print(f"shape check        : in {tuple(x.shape)} -> out {tuple(out.shape)}   OK")

    # --- softmax rows sum to 1 ---
    head = Head(n_embd=C, head_size=C // n_head)
    w = head._weights(x)
    row_sums = w.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums)), "rows must sum to 1"
    print("softmax check      : every attention row sums to 1   OK")

    # --- bidirectional: scrambling a LATER input must change an EARLIER output ---
    # (the decoder version asserted the OPPOSITE — this inversion is the whole edit)
    t = 3
    x2 = x.clone()
    x2[:, t + 1:] = torch.randn(B, T - (t + 1), C)
    out2 = mha(x2)
    changed = not torch.allclose(out[:, : t + 1], out2[:, : t + 1], atol=1e-6)
    assert changed, "earlier outputs ignored later inputs — attention isn't bidirectional!"
    print(f"bidirectional check: inputs after pos {t} DID move earlier outputs   OK")

    # --- padding mask: mark the last 3 positions as pad; no weight may land there ---
    mask = torch.ones(B, T, dtype=torch.bool)
    mask[:, -3:] = False
    w_masked = head._weights(x, mask)
    assert (w_masked[:, :, -3:] == 0).all(), "a query put weight on a pad key!"
    print("pad-mask check     : no attention weight lands on pad keys   OK")

    # --- pad isolation: changing pad tokens must not move real-token outputs ---
    x3 = x.clone()
    x3[:, -3:] = torch.randn(B, 3, C)
    out_a = mha(x, mask)
    out_b = mha(x3, mask)
    assert torch.allclose(out_a[:, :-3], out_b[:, :-3], atol=1e-6), \
        "changing a pad token moved a real output — the mask is leaking!"
    print("pad-isolation check: real outputs unchanged when pad tokens change   OK")

    print("\nall attention sanity checks passed.")
```

- [ ] **Step 2: Run the sanity block to confirm it FAILS**

Run:
```bash
.venv/bin/python src/model/attention.py
```
Expected: FAIL with `NameError: name 'MultiHeadAttention' is not defined`.

- [ ] **Step 3: Insert the four classes above the `if __name__` block**

Insert this block immediately after the `from torch.nn import functional as F` import (and its blank line), before `if __name__ == "__main__":`:

```python
class Head(nn.Module):
    """A single head of bidirectional self-attention.

    Each position emits a query/key/value; scores every position against every
    other (scaled dot product); softmaxes into weights; returns the weighted
    average of the values. Output: (B, T, head_size). No causal mask — every
    position attends to every other. An optional padding mask removes pad KEYS.
    """

    def __init__(self, n_embd: int, head_size: int, dropout: float = 0.0):
        super().__init__()
        # No bias: LayerNorm/embeddings handle offsets; Q/K/V are conventionally
        # bias-free. These are the only learned parts of a head.
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def _weights(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """The attention pattern: (B, T, T), each row summing to 1.

        Factored out of forward() so the sanity check can inspect the exact
        weights. mask is (B, T) bool, True = real token; pad KEYS get -inf so
        softmax gives them exactly 0 weight. (Dropout is applied in forward.)
        """
        B, T, C = x.shape
        k = self.key(x)                                  # (B, T, head_size)
        q = self.query(x)                                # (B, T, head_size)

        # Match every query against every key: (B,T,hs) @ (B,hs,T) -> (B,T,T).
        # Scale by 1/sqrt(head_size) so softmax inputs stay tame (large dot
        # products make softmax peaky -> vanishing gradients).
        scores = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5   # (B, T, T)

        if mask is not None:
            # mask (B, T_key) -> (B, 1, T_key): drop pad KEYS for every query row.
            scores = scores.masked_fill(~mask[:, None, :], float("-inf"))

        return F.softmax(scores, dim=-1)                 # (B, T, T)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        wei = self._weights(x, mask)                     # (B, T, T)
        wei = self.dropout(wei)                          # drop some attention links
        v = self.value(x)                                # (B, T, head_size)
        return wei @ v                                   # (B, T, head_size)


class MultiHeadAttention(nn.Module):
    """Several attention heads in parallel, concatenated then projected.

    Each head can specialize; we split n_embd into n_head chunks, attend within
    each, glue outputs back to width n_embd, then project to let heads mix.
    """

    def __init__(self, n_embd: int, n_head: int, dropout: float = 0.0):
        super().__init__()
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"
        head_size = n_embd // n_head
        self.heads = nn.ModuleList(
            [Head(n_embd, head_size, dropout) for _ in range(n_head)]
        )
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # Concatenate along the feature dim: n_head * head_size == n_embd.
        out = torch.cat([h(x, mask) for h in self.heads], dim=-1)   # (B, T, n_embd)
        return self.dropout(self.proj(out))                          # (B, T, n_embd)


class FeedForward(nn.Module):
    """Per-position MLP: expand 4x, GELU, shrink back. Runs on each position
    independently — mixing across positions is attention's job."""

    def __init__(self, n_embd: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    """One encoder block: attention then feed-forward, each in a residual with
    pre-LayerNorm.

        x = x + attn(norm(x), mask)   # mix context across positions
        x = x + ffn(norm(x))          # think, per position

    The residual gives gradients a clean path; each sublayer learns only a
    correction. Pre-norm (normalize the sublayer INPUT) is the stable convention.
    """

    def __init__(self, n_embd: int, n_head: int, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = MultiHeadAttention(n_embd, n_head, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffn = FeedForward(n_embd, dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.ffn(self.ln2(x))
        return x
```

- [ ] **Step 4: Run the sanity block to confirm it PASSES**

Run:
```bash
.venv/bin/python src/model/attention.py
```
Expected output (all five lines, then the summary):
```
shape check        : in (4, 8, 32) -> out (4, 8, 32)   OK
softmax check      : every attention row sums to 1   OK
bidirectional check: inputs after pos 3 DID move earlier outputs   OK
pad-mask check     : no attention weight lands on pad keys   OK
pad-isolation check: real outputs unchanged when pad tokens change   OK

all attention sanity checks passed.
```

- [ ] **Step 5: Lint clean**

Run:
```bash
.venv/bin/ruff check src/model/attention.py
```
Expected: no issues. (If E501 fires on any line, wrap it to ≤120 chars and re-run.)

- [ ] **Step 6: Flip README Foundation status to done, then commit**

Edit `README.md`: change `- [ ] Foundation — repo scaffold + bidirectional attention blocks` to `- [x] Foundation — repo scaffold + bidirectional attention blocks`.

Then:
```bash
git add src/model/attention.py README.md
git commit -m "feat: vendored bidirectional attention blocks

Stripped the from-scratch GPT attention to an encoder: removed causal
masking, added an optional boolean key-padding mask, dropped block_size
from the attention path. Inline sanity block asserts bidirectionality,
softmax normalization, and padding isolation.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```
Expected: commit succeeds; `git log --oneline` shows the spec, scaffold, and attention commits.

---

## Self-Review

**1. Spec coverage** (against `docs/specs/2026-07-22-foundation-design.md`):
- Repo scaffold / dir tree / `.gitignore` / `.env.example` / `requirements.txt` / README stub → Task 1. ✓
- `BUILD_TWO_TOWER.md` gitignored → Task 1 Steps 2 & 11. ✓
- Full-stack `requirements.txt` grouped by phase, only torch+ruff installed → Task 1 Steps 4–5. ✓
- README leads with 3-system results table → Task 1 Step 6. ✓
- attention.py: `Head`/`MHA`/`FeedForward`/`Block`, `GPT`/`GPTConfig`/embeddings/generation deleted → Task 2 (only the four classes are added). ✓
- Remove causal mask + drop `block_size` from attention → Task 2 Step 3 (no `tril`, no `block_size` args). ✓
- Boolean key-padding mask, `True`=real, pad keys→-inf, `mask=None` default → Task 2 Step 3 (`_weights`). ✓
- Provenance header with `<transformer-repo>` TODO → Task 2 Step 1. ✓
- Rewritten inline sanity checks (shape, softmax, bidirectional, pad-mask, pad-isolation) → Task 2 Steps 1 & 4. ✓
- Class name `Head` → Task 2 Step 3. ✓
- Done criteria: sanity passes, ruff clean, committed in the spec's commit order (spec → scaffold → attention) → Tasks 1–2. ✓

**2. Placeholder scan:** the only `<...>`/TODO is the transformer-repo URL — an intentional fill-in tracked in the spec's "Open fill-ins", not a plan gap. No "TBD" outside the README results cells (intentional — no results yet). No vague "handle edge cases" steps. ✓

**3. Type consistency:** `Head(n_embd, head_size, dropout)`, `MultiHeadAttention(n_embd, n_head, dropout)`, `FeedForward(n_embd, dropout)`, `Block(n_embd, n_head, dropout)` — signatures in the Interfaces block match the code in Task 2 Step 3. `forward(x, mask=None)` everywhere except `FeedForward.forward(x)`. `_weights(x, mask=None)` used by the sanity block matches its definition. ✓
