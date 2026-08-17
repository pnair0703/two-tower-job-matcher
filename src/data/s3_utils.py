import boto3
import json
from pathlib import Path
from tenacity import retry, stop_after_attempt, wait_exponential
from src.utils.logging import setup_logging

logger = setup_logging("s3_utils")
s3_client = boto3.client("s3")

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2))
def download_from_s3(s3_key, bucket, local_path, max_retries=3):
    """Download file from S3 with retry logic."""
    logger.info(f"Downloading s3://{bucket}/{s3_key} → {local_path}")
    try:
        s3_client.download_file(bucket, s3_key, str(local_path))
        logger.info(f"✓ Downloaded {local_path}")
    except Exception as e:
        logger.error(f"Failed to download {s3_key}: {e}")
        raise

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2))
def upload_to_s3(local_path, s3_key, bucket):
    """Upload file to S3 with retry logic."""
    logger.info(f"Uploading {local_path} → s3://{bucket}/{s3_key}")
    try:
        s3_client.upload_file(str(local_path), bucket, s3_key)
        logger.info(f"✓ Uploaded to s3://{bucket}/{s3_key}")
    except Exception as e:
        logger.error(f"Failed to upload {local_path}: {e}")
        raise

def list_s3_objects(prefix, bucket):
    """List all objects in S3 with given prefix."""
    response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
    if "Contents" in response:
        return [obj["Key"] for obj in response["Contents"]]
    return []

def read_jsonl_from_s3(s3_key, bucket, local_cache=None):
    """Download and read JSONL file from S3."""
    if local_cache and Path(local_cache).exists():
        logger.info(f"Using cached {local_cache}")
        return read_jsonl_local(local_cache)

    logger.info(f"Reading JSONL from s3://{bucket}/{s3_key}")
    temp_file = Path(f"/tmp/{Path(s3_key).name}")
    download_from_s3(s3_key, bucket, temp_file)
    data = read_jsonl_local(temp_file)
    temp_file.unlink()
    return data

def read_jsonl_local(path):
    """Read JSONL file and return list of dicts."""
    data = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def write_jsonl(data, path):
    """Write list of dicts to JSONL file."""
    with open(path, "w") as f:
        for record in data:
            f.write(json.dumps(record) + "\n")
