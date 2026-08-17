"""Phase 2 — contrastive loss with in-batch negatives."""

import torch
import torch.nn.functional as F


def contrastive_loss(job_embeddings, resume_embeddings, scores, temperature=0.07):
    """
    Contrastive loss for job-resume matching using in-batch negatives.

    Args:
        job_embeddings: (batch_size, embedding_dim) normalized embeddings
        resume_embeddings: (batch_size, embedding_dim) normalized embeddings
        scores: (batch_size,) relevance scores (0-3)
        temperature: scaling factor for logits

    Returns:
        loss: scalar torch tensor
    """
    # Compute similarity matrix: (batch_size, batch_size)
    # All jobs vs all resumes, using dot product (embeddings are L2-normalized)
    similarity_matrix = torch.matmul(job_embeddings, resume_embeddings.T)  # (B, B)

    # Scale by temperature
    logits = similarity_matrix / temperature

    # Labels: convert scores to binary (positive if score >= 2)
    labels = (scores >= 2).float().unsqueeze(1)  # (B, 1)

    # Compute contrastive loss using cosine similarity and in-batch negatives
    # For each job-resume pair, we want to push positive pairs (score >= 2) closer
    # and negative pairs (score < 2) apart

    batch_size = job_embeddings.shape[0]

    # Create positive and negative masks
    pos_mask = torch.eye(batch_size, device=job_embeddings.device, dtype=torch.bool)  # (B, B)
    neg_mask = ~pos_mask

    # Compute loss for each sample
    loss = 0.0
    for i in range(batch_size):
        # Positive logits (diagonal)
        pos_logit = logits[i, i]

        # Negative logits (off-diagonal)
        neg_logits = logits[i, neg_mask[i]]

        # Log-softmax loss
        # We want pos_logit to be large, neg_logits to be small
        logits_combined = torch.cat([pos_logit.unsqueeze(0), neg_logits])
        labels_combined = torch.zeros(len(logits_combined), device=job_embeddings.device, dtype=torch.long)

        loss += F.cross_entropy(logits_combined.unsqueeze(0), labels_combined.unsqueeze(0))

    loss = loss / batch_size
    return loss
