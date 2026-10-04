import os
from dotenv import load_dotenv
from pathlib import Path

def load_config():
    """Load configuration from .env and environment variables."""
    load_dotenv()

    return {
        "claude_api_key": os.getenv("CLAUDE_API_KEY"),
        "aws_region": os.getenv("AWS_REGION", "us-east-1"),
        "s3_bucket": os.getenv("S3_BUCKET", "two-tower-pnair"),
        "sagemaker_role_arn": os.getenv("SAGEMAKER_ROLE_ARN"),
        "training_image_uri": os.getenv("TRAINING_IMAGE_URI"),
        "project_name": os.getenv("PROJECT_NAME", "two-tower"),
        "log_level": os.getenv("LOG_LEVEL", "INFO"),
    }

def get_s3_paths():
    """Return S3 paths for all artifacts."""
    return {
        "postings": "postings/postings.jsonl",
        "labels": "labels/labels.jsonl",
        "labels_sampled": "labels/sampled_400.jsonl",
        "model_a": "models/tower_a/final.pt",
        "model_b": "models/tower_b/final.pt",
        "embeddings_a": "embeddings/jobs_tower_a.jsonl",
        "embeddings_b": "embeddings/jobs_tower_b.jsonl",
        "index_a": "indices/tower_a.faiss",
        "index_b": "indices/tower_b.faiss",
        "index_map": "indices/posting_id_map.pkl",
        "results": "results/eval_results.json",
    }
