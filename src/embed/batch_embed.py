import argparse
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from src.model.towers import TowerA, TowerB
from src.data.s3_utils import read_jsonl_from_s3, download_from_s3, upload_to_s3, write_jsonl
from src.utils.config import load_config, get_s3_paths
from src.utils.logging import setup_logging

logger = setup_logging("batch_embed")

def batch_embed_jobs(model, postings, batch_size=64, tower_type="a", device="cpu"):
    """Embed all job postings with a trained model."""
    logger.info(f"Embedding {len(postings)} postings with Tower {tower_type.upper()}")

    embeddings = []
    model.eval()

    with torch.no_grad():
        for i in tqdm(range(0, len(postings), batch_size)):
            batch_postings = postings[i:i+batch_size]

            if tower_type == "a":
                # Tokenize (placeholder — implement tokenizer)
                texts = [f"{p['title']} {p['description']}" for p in batch_postings]
                # TODO: tokenize texts to tokens
                # embeddings_batch = model(tokens.to(device))
            else:
                # TowerB (sentence-transformer) expects text
                texts = [f"{p['title']} {p['company']} {p['description']}" for p in batch_postings]
                embeddings_batch = model(texts)

            for j, posting in enumerate(batch_postings):
                embedding = embeddings_batch[j].cpu().numpy().tolist()
                embeddings.append({
                    "posting_id": posting["id"],
                    f"embedding_{tower_type}": embedding,
                })

    logger.info(f"✓ Embedded {len(embeddings)} postings")
    return embeddings

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tower", required=True, choices=["a", "b"])
    parser.add_argument("--model-path", help="Local path to trained model")
    parser.add_argument("--postings-s3", default="postings/postings.jsonl")
    parser.add_argument("--output-s3", default="embeddings/jobs_tower_x.jsonl")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--local", action="store_true")
    args = parser.parse_args()

    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load postings
    logger.info("Loading postings")
    postings = read_jsonl_from_s3(args.postings_s3, config["s3_bucket"])
    logger.info(f"Loaded {len(postings)} postings")

    # Load model
    if args.tower == "a":
        model = TowerA()
        if args.model_path:
            model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))
    else:
        model = TowerB()
        if args.model_path:
            model.model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))

    model.to(device)

    # Embed
    embeddings = batch_embed_jobs(model, postings, args.batch_size, args.tower, device)

    # Save locally first
    temp_path = Path(f"/tmp/embeddings_{args.tower}.jsonl")
    write_jsonl(embeddings, temp_path)

    # Upload to S3
    if not args.local:
        upload_to_s3(temp_path, args.output_s3.replace("_x", f"_{args.tower}"), config["s3_bucket"])

    logger.info(f"✓ Saved embeddings to {args.output_s3}")

if __name__ == "__main__":
    main()
