import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from src.model.towers import TowerA, TowerB, JobResumePair
from src.model.train_utils import create_dataloaders, save_checkpoint
from src.model.loss import contrastive_loss
from src.data.s3_utils import read_jsonl_local
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

VOCAB_SIZE = 5000
EMBEDDING_DIM = 384


def train_epoch(model, train_loader, optimizer, criterion, device, tower_type="a"):
    """Train one epoch."""
    model.train()
    total_loss = 0

    for batch_idx, batch in enumerate(train_loader):
        score = batch["score"].to(device)

        optimizer.zero_grad()

        if tower_type == "a":
            job_tokens = batch["job_tokens"].to(device)
            resume_tokens = batch["resume_tokens"].to(device)
            job_embedding = model(job_tokens)
            resume_embedding = model(resume_tokens)
        else:
            # TowerB expects raw text, tokenizes internally
            job_embedding = model(batch["job_text"])
            resume_embedding = model(batch["resume_text"])

        # Contrastive loss
        loss = criterion(job_embedding, resume_embedding, score)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if batch_idx % 10 == 0:
            logger.info(f"Batch {batch_idx}: loss {loss.item():.4f}")

    avg_loss = total_loss / len(train_loader)
    logger.info(f"Epoch loss: {avg_loss:.4f}")
    return avg_loss


def validate(model, val_loader, criterion, device, tower_type="a"):
    """Validate on held-out set."""
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in val_loader:
            score = batch["score"].to(device)

            if tower_type == "a":
                job_tokens = batch["job_tokens"].to(device)
                resume_tokens = batch["resume_tokens"].to(device)
                job_embedding = model(job_tokens)
                resume_embedding = model(resume_tokens)
            else:
                job_embedding = model(batch["job_text"])
                resume_embedding = model(batch["resume_text"])

            loss = criterion(job_embedding, resume_embedding, score)
            total_loss += loss.item()

    avg_loss = total_loss / len(val_loader)
    logger.info(f"Validation loss: {avg_loss:.4f}")
    return avg_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tower", default="a", choices=["a", "b"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--resume-file", default="data/resume.txt", help="Path to resume text file")
    parser.add_argument("--local", action="store_true", help="Train locally (not on SageMaker)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on device: {device}")

    # Load data from S3 (or locally if --local)
    if args.local:
        input_dir = Path("data")
    else:
        # SageMaker mounts input at /opt/ml/input/data/training
        input_dir = Path("/opt/ml/input/data/training")

    labels_path = input_dir / "labels.jsonl"
    postings_path = input_dir / "postings.jsonl"

    logger.info(f"Loading labels from {labels_path}")
    labels = read_jsonl_local(labels_path)
    logger.info(f"Loading postings from {postings_path}")
    postings = read_jsonl_local(postings_path)

    resume_text = Path(args.resume_file).read_text()

    # Create model + tokenizer
    tokenizer_fn = None
    vocab = None
    if args.tower == "a":
        model = TowerA(vocab_size=VOCAB_SIZE, embedding_dim=EMBEDDING_DIM)
        vocab = JobResumePair(vocab_size=VOCAB_SIZE)
        job_texts = [f"{p['title']} {p['company']} {p['description']}" for p in postings]
        vocab.build_vocab(job_texts + [resume_text])
        tokenizer_fn = vocab.tokenize
    else:
        model = TowerB()

    model.to(device)

    train_loader, val_loader, test_loader = create_dataloaders(
        labels, postings, resume_text, tokenizer_fn, batch_size=args.batch_size
    )

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = contrastive_loss

    # Output dir
    output_dir = Path("/opt/ml/model") if not args.local else Path("models")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Train
    best_val_loss = float("inf")
    for epoch in range(args.epochs):
        logger.info(f"Epoch {epoch+1}/{args.epochs}")
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, args.tower)
        val_loss = validate(model, val_loader, criterion, device, args.tower)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), output_dir / "model.pt")
            logger.info(f"✓ New best val loss {val_loss:.4f}, saved model.pt")

    if args.tower == "a":
        vocab.save_vocab(output_dir / "vocab.json")
        logger.info(f"✓ Saved vocab to {output_dir}/vocab.json")

    logger.info(f"✓ Training complete, best val loss {best_val_loss:.4f}")

    # Upload to S3 (if not local)
    if not args.local:
        try:
            from src.data.s3_utils import upload_to_s3
            from src.utils.config import load_config
            config = load_config()
            upload_to_s3(output_dir / "model.pt", f"models/tower_{args.tower}/final.pt", config["s3_bucket"])
            if args.tower == "a":
                upload_to_s3(output_dir / "vocab.json", f"models/tower_{args.tower}/vocab.json", config["s3_bucket"])
        except Exception as e:
            logger.warning(f"Failed to upload to S3: {e}")


if __name__ == "__main__":
    main()
