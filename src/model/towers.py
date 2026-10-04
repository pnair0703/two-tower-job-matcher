"""Phase 2 — JobTower + ResumeTower (own encoder), same interface as Tower B."""

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model.attention import Block
from sentence_transformers import SentenceTransformer


class TowerA(nn.Module):
    """Encoder tower trained from scratch."""

    def __init__(self, vocab_size=10000, embedding_dim=384, num_blocks=6, num_heads=8):
        super().__init__()
        self.embedding_dim = embedding_dim

        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.positional_embedding = nn.Embedding(512, embedding_dim)  # Max sequence length

        self.blocks = nn.ModuleList([
            Block(embedding_dim, num_heads, dropout=0.1)
            for _ in range(num_blocks)
        ])

        self.ln_final = nn.LayerNorm(embedding_dim)

    def forward(self, token_ids):
        """
        Args:
            token_ids: (batch_size, seq_len) LongTensor

        Returns:
            embeddings: (batch_size, embedding_dim) normalized float32
        """
        batch_size, seq_len = token_ids.shape

        # Embed tokens + add positional embeddings
        x = self.token_embedding(token_ids)  # (B, T, D)
        pos = torch.arange(seq_len, device=token_ids.device).unsqueeze(0)  # (1, T)
        x = x + self.positional_embedding(pos)  # Broadcast

        # Apply transformer blocks (bidirectional, no causal mask)
        for block in self.blocks:
            x = block(x, mask=None)  # No mask = all positions see each other

        # Final layer norm
        x = self.ln_final(x)  # (B, T, D)

        # Mean pooling over positions
        embeddings = x.mean(dim=1)  # (B, D)

        # L2 normalize
        embeddings = F.normalize(embeddings, p=2, dim=-1)  # (B, D)

        return embeddings


class TowerB(nn.Module):
    """Encoder tower using fine-tuned pretrained sentence transformer."""

    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        super().__init__()
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

    def forward(self, texts):
        """
        Args:
            texts: List[str] of job postings or resumes

        Returns:
            embeddings: (batch_size, embedding_dim) normalized float32
        """
        embeddings = self.model.encode(texts, convert_to_tensor=True, normalize_embeddings=True)
        return embeddings


class JobResumePair:
    """Helper to tokenize and pair jobs + resumes."""

    def __init__(self, vocab_size=10000, max_seq_len=512):
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.word_to_id = {}
        self.id_counter = 1

    def build_vocab(self, texts):
        """Build vocabulary from texts."""
        words = set()
        for text in texts:
            words.update(text.lower().split())

        for word in sorted(words):
            if len(self.word_to_id) < self.vocab_size - 1:
                self.word_to_id[word] = self.id_counter
                self.id_counter += 1

    def tokenize(self, text):
        """Convert text to token IDs."""
        tokens = [self.word_to_id.get(w, 0) for w in text.lower().split()]
        # Pad or truncate
        if len(tokens) < self.max_seq_len:
            tokens += [0] * (self.max_seq_len - len(tokens))
        else:
            tokens = tokens[:self.max_seq_len]
        return torch.tensor(tokens, dtype=torch.long)

    def save_vocab(self, path):
        """Save vocab to JSON so inference can rebuild the same tokenizer."""
        with open(path, "w") as f:
            json.dump({
                "vocab_size": self.vocab_size,
                "max_seq_len": self.max_seq_len,
                "word_to_id": self.word_to_id,
            }, f)

    @classmethod
    def load_vocab(cls, path):
        """Load a tokenizer from a vocab JSON saved by save_vocab."""
        with open(path, "r") as f:
            data = json.load(f)
        tokenizer = cls(vocab_size=data["vocab_size"], max_seq_len=data["max_seq_len"])
        tokenizer.word_to_id = data["word_to_id"]
        tokenizer.id_counter = len(tokenizer.word_to_id) + 1
        return tokenizer
