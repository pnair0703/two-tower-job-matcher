# Transformer blocks adapted from my from-scratch implementation:
# github.com/pnair0703/nair-gpt
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
