import pytest
import torch
from src.model.towers import TowerA, TowerB


def test_tower_a_forward():
    """Test TowerA forward pass."""
    tower = TowerA(vocab_size=5000, embedding_dim=256, num_blocks=2)
    batch_size, seq_len = 4, 128
    token_ids = torch.randint(0, 5000, (batch_size, seq_len))

    embeddings = tower(token_ids)

    assert embeddings.shape == (batch_size, 256)
    assert torch.allclose(torch.norm(embeddings, dim=-1), torch.ones(batch_size), atol=1e-6)


def test_tower_a_norm():
    """Test that TowerA outputs are L2-normalized."""
    tower = TowerA()
    token_ids = torch.randint(0, 1000, (2, 100))
    embeddings = tower(token_ids)

    norms = torch.norm(embeddings, dim=-1)
    assert torch.allclose(norms, torch.ones(2), atol=1e-6)


def test_tower_b_forward():
    """Test TowerB forward pass."""
    tower = TowerB()
    texts = ["Machine learning engineer needed", "Sales role, no tech required"]

    embeddings = tower(texts)

    assert embeddings.shape[0] == len(texts)
    assert embeddings.shape[1] == tower.embedding_dim
    expected_norms = torch.ones(len(texts), device=embeddings.device)
    assert torch.allclose(torch.norm(embeddings, dim=-1), expected_norms, atol=1e-6)
