import torch
from torch.utils.data import Dataset, DataLoader
import logging

logger = logging.getLogger("train_utils")


class JobResumePairDataset(Dataset):
    """Dataset of (job posting, resume) pairs with labels."""

    def __init__(self, labels, postings_by_id, resume_text, tokenizer, split="train"):
        """
        Args:
            labels: List[Dict] with posting_id, score, split
            postings_by_id: Dict[posting_id] -> posting dict
            resume_text: str
            tokenizer: function to convert text → token IDs
            split: "train", "val", or "test"
        """
        self.labels = [l for l in labels if l["split"] == split]
        self.postings_by_id = postings_by_id
        self.resume_text = resume_text
        self.tokenizer = tokenizer
        self.split = split

        logger.info(f"Loaded {len(self.labels)} {split} samples")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label_dict = self.labels[idx]
        posting_id = label_dict["posting_id"]
        posting = self.postings_by_id[posting_id]

        # Concatenate job description
        job_text = f"{posting['title']} {posting['company']} {posting['description']}"

        # Tokenize
        job_tokens = self.tokenizer(job_text)
        resume_tokens = self.tokenizer(self.resume_text)

        # Label (0-3)
        score = label_dict["score"]

        return {
            "job_tokens": job_tokens,
            "resume_tokens": resume_tokens,
            "score": torch.tensor(score, dtype=torch.float),
            "posting_id": posting_id,
        }


def create_dataloaders(labels, postings, resume_text, tokenizer, batch_size=32):
    """Create train/val/test dataloaders."""
    postings_by_id = {p["id"]: p for p in postings}

    train_dataset = JobResumePairDataset(labels, postings_by_id, resume_text, tokenizer, split="train")
    val_dataset = JobResumePairDataset(labels, postings_by_id, resume_text, tokenizer, split="val")
    test_dataset = JobResumePairDataset(labels, postings_by_id, resume_text, tokenizer, split="test")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def save_checkpoint(model, optimizer, epoch, loss, path):
    """Save model checkpoint."""
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "loss": loss,
    }, path)
    logger.info(f"Saved checkpoint to {path}")


def load_checkpoint(model, optimizer, path):
    """Load model checkpoint."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]
    logger.info(f"Loaded checkpoint from {path} (epoch {epoch}, loss {loss})")
    return epoch, loss
