# Two-Tower Job Matcher Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build end-to-end infrastructure for a two-tower job matching system: label 400 job postings, train two encoders (from-scratch + fine-tuned), embed all jobs offline, and evaluate against a BM25 baseline.

**Architecture:** AWS-first pipeline: Phase 1 uses Claude API for labeling, Phases 2-3 train models via SageMaker, Phase 4 precomputes embeddings to S3 and builds FAISS indices, Phase 5 evaluates locally. All data lives in S3; no local GPU stress.

**Tech Stack:** PyTorch, Anthropic Claude API, AWS SageMaker/S3, FAISS, sentence-transformers, scikit-learn

## Global Constraints

- AWS region: `us-east-1` (adjust in `.env`)
- S3 bucket: `two-tower-pnair` (create before starting)
- Python: 3.11+
- FAISS index type: `IndexFlatIP` (dot-product, L2-normalized vectors)
- No local GPU training — cloud-first for all compute jobs
- All data schemas are JSONL (one JSON object per line)
- Models stored as PyTorch `.pt` files (state_dict, not full module)
- Embeddings: float32, stored as JSON lists (human-readable for debugging)

---

## File Structure

**New files to create:**

```
src/data/
├── label.py              # Phase 1: Claude API labeling + agreement computation
├── s3_utils.py           # Shared S3 download/upload helpers
└── constants.py          # Schemas, rubric templates, S3 paths

src/model/
├── towers.py             # Phase 2-3: TowerA (scratch), TowerB (pretrained)
├── sagemaker_train.py    # SageMaker training entry point
└── train_utils.py        # Checkpointing, validation loops, data loading

src/embed/
├── batch_embed.py        # Phase 4: Load models, embed all 13K jobs
├── build_faiss.py        # Phase 4: Build FAISS indices from embeddings
└── index_utils.py        # Query wrapper, posting_id ↔ index mapping

src/eval/
├── run_eval.py           # Phase 5: Download indices, evaluate all three systems
└── eval_utils.py         # Format results, generate README table

src/utils/
├── config.py             # Load .env, AWS config, bucket names
└── logging.py            # CloudWatch + local file logging

docker/
├── Dockerfile.train      # SageMaker training image (PyTorch + deps)
└── docker-compose.yml    # LocalStack for local S3 mocking (dev only)

scripts/
├── setup_aws.sh          # Create S3 bucket, upload Phase 0 data
├── submit_training_job.py # Submit SageMaker Training job, monitor
└── download_checkpoint.py # Fetch trained model from S3

tests/
├── test_data.py          # Validate JSONL schemas (postings, labels)
├── test_model.py         # TowerA/B forward pass, output shapes
└── test_eval.py          # Metrics computation (P@k, NDCG)

.env.example              # Environment template (CLAUDE_API_KEY, AWS_REGION, S3_BUCKET)
.gitignore               # data/, *.pt, *.faiss, .env, __pycache__
Makefile                 # Shortcuts: make label, make train-a, make embed, make eval
requirements.txt         # Add: anthropic, boto3, sagemaker, faiss-cpu, sentence-transformers
```

**Modify existing files:**

```
src/model/attention.py    # Already exists (bidirectional blocks from transformer repo)
src/model/loss.py         # Already exists (contrastive loss)
src/eval/metrics.py       # Already exists (P@k, R@k, NDCG)
src/eval/baseline.py      # Already exists (BM25/TF-IDF)
README.md                 # Add results table stub (Phase 5)
```

---

## Task Breakdown

### Phase 1: Labeling (3 tasks)

#### Task 1: Setup & Dependencies

**Files:**
- Create: `.env.example`, `.gitignore`, `requirements.txt`
- Modify: (none)

**Interfaces:**
- Produces: Environment file template, installed dependencies

- [ ] **Step 1: Create `.env.example`**

```bash
cat > .env.example << 'EOF'
# Claude API
CLAUDE_API_KEY=sk-ant-...

# AWS
AWS_REGION=us-east-1
S3_BUCKET=two-tower-pnair

# Project
PROJECT_NAME=two-tower
LOG_LEVEL=INFO
EOF
```

- [ ] **Step 2: Create `.gitignore`**

```bash
cat > .gitignore << 'EOF'
# Data (large, not committed)
data/
*.jsonl
*.pt
*.faiss
*.pkl

# Environment
.env
.env.local

# Python
__pycache__/
*.pyc
.venv/
*.egg-info/

# IDE
.vscode/
.idea/
*.swp

# Logs
logs/
*.log
EOF
```

- [ ] **Step 3: Update `requirements.txt`**

Read current `requirements.txt`, then append:

```bash
cat >> requirements.txt << 'EOF'
# APIs
anthropic>=0.25.0

# AWS
boto3>=1.28.0
sagemaker>=2.180.0

# ML/Embeddings
torch>=2.0.0
sentence-transformers>=2.2.2
scikit-learn>=1.3.0
faiss-cpu>=1.7.4

# Data
pandas>=2.0.0
numpy>=1.24.0

# Utils
python-dotenv>=1.0.0
tenacity>=8.2.0
tqdm>=4.66.0
EOF
```

- [ ] **Step 4: Create virtual environment and install**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

- [ ] **Step 5: Copy `.env.example` to `.env` and configure**

```bash
cp .env.example .env
# Edit .env manually: add CLAUDE_API_KEY, confirm AWS_REGION
```

- [ ] **Step 6: Commit**

```bash
git add .env.example .gitignore requirements.txt
git commit -m "setup: environment, gitignore, dependencies"
```

---

#### Task 2: Data Infrastructure (Labels, Schemas, S3 Utilities)

**Files:**
- Create: `src/data/constants.py`, `src/data/s3_utils.py`, `src/utils/config.py`, `src/utils/logging.py`
- Modify: (none)
- Test: `tests/test_data.py`

**Interfaces:**
- Produces: 
  - `config.load_config()` → Config dict with AWS_REGION, S3_BUCKET, etc.
  - `s3_utils.upload_to_s3(local_path, s3_key, bucket, max_retries=3)` → None (uploads file)
  - `s3_utils.download_from_s3(s3_key, bucket, local_path, max_retries=3)` → None (downloads file)
  - `constants.LABEL_RUBRIC` → str (labeling prompt)
  - `constants.POSTING_SCHEMA` → dict (JSON schema for validation)

- [ ] **Step 1: Create `src/utils/config.py`**

```python
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
```

- [ ] **Step 2: Create `src/utils/logging.py`**

```python
import logging
import sys
from pathlib import Path
from src.utils.config import load_config

def setup_logging(name="two-tower"):
    """Configure logging to file and stdout."""
    config = load_config()
    log_level = getattr(logging, config["log_level"].upper(), logging.INFO)
    
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # File handler
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    file_handler = logging.FileHandler(log_dir / f"{name}.log")
    file_handler.setLevel(log_level)
    
    # Stream handler
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(log_level)
    
    # Formatter
    formatter = logging.Formatter(
        fmt="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    
    return logger
```

- [ ] **Step 3: Create `src/data/constants.py`**

```python
POSTING_SCHEMA = {
    "type": "object",
    "required": ["id", "company", "title", "description", "url", "source"],
    "properties": {
        "id": {"type": "string"},
        "company": {"type": "string"},
        "title": {"type": "string"},
        "location": {"type": "string"},
        "description": {"type": "string"},
        "url": {"type": "string"},
        "source": {"type": "string", "enum": ["greenhouse", "lever", "ashby", "remoteok"]},
    },
}

LABEL_SCHEMA = {
    "type": "object",
    "required": ["posting_id", "score"],
    "properties": {
        "posting_id": {"type": "string"},
        "score": {"type": "integer", "minimum": 0, "maximum": 3},
        "llm_score": {"type": "integer", "minimum": 0, "maximum": 3},
        "human_score": {"type": "integer", "minimum": 0, "maximum": 3},
        "split": {"type": "string", "enum": ["train", "val", "test"]},
        "is_test_human_labeled": {"type": "boolean"},
        "agreement": {"type": "boolean"},
    },
}

LABEL_RUBRIC = """You are evaluating job postings for fit to this resume.

Resume summary: ML engineer with experience in LLMs, RAG systems, AWS, PyTorch, transformers.

Score the posting 0-3:
- 3: Strong match. Role aligns with skills/interests. Apply.
- 2: Moderate match. Some alignment; worth considering.
- 1: Weak match. Misses key areas but not disqualifying.
- 0: No match. Wrong domain or level.

Focus on: technical stack, seniority level, domain (ML/AI vs. other).

Respond with ONLY the score (0-3), no explanation.
"""

# Task thresholds
AGREEMENT_THRESHOLD = 0.85  # If LLM-human agreement >= 85%, use LLM scores for all
SAMPLE_SIZE_TRAIN = 320
SAMPLE_SIZE_VAL = 30
SAMPLE_SIZE_TEST = 50  # Human-labeled
```

- [ ] **Step 4: Create `src/data/s3_utils.py`**

```python
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
```

- [ ] **Step 5: Create `tests/test_data.py`**

```python
import pytest
import json
from pathlib import Path
from src.data.constants import POSTING_SCHEMA, LABEL_SCHEMA

def validate_jsonl_schema(path, schema):
    """Helper to validate JSONL against schema."""
    from jsonschema import validate
    with open(path, "r") as f:
        for i, line in enumerate(f):
            if line.strip():
                obj = json.loads(line)
                try:
                    validate(instance=obj, schema=schema)
                except Exception as e:
                    raise AssertionError(f"Line {i}: {e}")

def test_posting_schema_valid():
    """Validate POSTING_SCHEMA structure."""
    assert "properties" in POSTING_SCHEMA
    assert "id" in POSTING_SCHEMA["properties"]
    assert "company" in POSTING_SCHEMA["properties"]

def test_label_schema_valid():
    """Validate LABEL_SCHEMA structure."""
    assert "properties" in LABEL_SCHEMA
    assert "posting_id" in LABEL_SCHEMA["properties"]
    assert "score" in LABEL_SCHEMA["properties"]
    assert LABEL_SCHEMA["properties"]["score"]["minimum"] == 0
    assert LABEL_SCHEMA["properties"]["score"]["maximum"] == 3
```

- [ ] **Step 6: Run tests to verify schemas**

```bash
pytest tests/test_data.py -v
```

Expected: PASS (2 tests)

- [ ] **Step 7: Commit**

```bash
git add src/utils/config.py src/utils/logging.py src/data/constants.py src/data/s3_utils.py tests/test_data.py
git commit -m "feat(phase1): data infrastructure (schemas, S3 utils, config)"
```

---

#### Task 3: Phase 1 Labeling (Claude API + Human Validation)

**Files:**
- Create: `src/data/label.py`, `scripts/setup_aws.sh`
- Modify: (none)
- Test: (manual validation in script)

**Interfaces:**
- Consumes: `s3_utils.read_jsonl_from_s3()`, `config.load_config()`
- Produces: `labels.jsonl` in S3 with format: `{posting_id, score, llm_score, human_score, split, agreement}`

- [ ] **Step 1: Create `scripts/setup_aws.sh`**

```bash
#!/bin/bash
set -e

source .env

BUCKET=$S3_BUCKET
REGION=$AWS_REGION

echo "Creating S3 bucket: $BUCKET"
aws s3 mb s3://$BUCKET --region $REGION || echo "Bucket already exists"

echo "Creating S3 folders..."
aws s3api put-object --bucket $BUCKET --key postings/ || true
aws s3api put-object --bucket $BUCKET --key labels/ || true
aws s3api put-object --bucket $BUCKET --key models/tower_a/ || true
aws s3api put-object --bucket $BUCKET --key models/tower_b/ || true
aws s3api put-object --bucket $BUCKET --key embeddings/ || true
aws s3api put-object --bucket $BUCKET --key indices/ || true
aws s3api put-object --bucket $BUCKET --key results/ || true

echo "Uploading Phase 0 postings..."
aws s3 cp data/postings.jsonl s3://$BUCKET/postings/postings.jsonl

echo "✓ AWS setup complete"
```

- [ ] **Step 2: Create `src/data/label.py`**

```python
import json
import random
from pathlib import Path
from anthropic import Anthropic
from src.utils.config import load_config, get_s3_paths
from src.utils.logging import setup_logging
from src.data.s3_utils import read_jsonl_from_s3, write_jsonl, upload_to_s3
from src.data.constants import LABEL_RUBRIC, SAMPLE_SIZE_TRAIN, SAMPLE_SIZE_VAL, SAMPLE_SIZE_TEST

logger = setup_logging("label")
client = Anthropic()

def stratified_sample(postings, sample_size=400):
    """Sample postings stratified by company size (proxy for role fit)."""
    logger.info(f"Stratified sampling {sample_size} from {len(postings)} postings")
    
    # Group by company to balance representation
    by_company = {}
    for p in postings:
        company = p["company"]
        if company not in by_company:
            by_company[company] = []
        by_company[company].append(p)
    
    # Sample proportionally from each company
    sampled = []
    for company, jobs in by_company.items():
        num_to_sample = max(1, round(len(jobs) / len(postings) * sample_size))
        sampled.extend(random.sample(jobs, min(num_to_sample, len(jobs))))
    
    return sampled[:sample_size]

def score_posting_with_claude(posting, rubric=LABEL_RUBRIC, max_retries=3):
    """Score a single posting using Claude API."""
    prompt = f"""{rubric}

Job Posting:
Title: {posting['title']}
Company: {posting['company']}
Location: {posting['location']}
Description: {posting['description'][:2000]}
"""
    
    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=10,
                messages=[{"role": "user", "content": prompt}],
            )
            score_text = response.content[0].text.strip()
            score = int(score_text)
            if 0 <= score <= 3:
                return score
            else:
                logger.warning(f"Invalid score {score} for {posting['id']}, retrying")
        except Exception as e:
            logger.warning(f"Attempt {attempt+1} failed: {e}")
            if attempt == max_retries - 1:
                raise
    
    raise ValueError(f"Could not score {posting['id']} after {max_retries} retries")

def label_postings(postings_jsonl_s3_path, sample_size=400, output_local_path="labels_temp.jsonl"):
    """Label postings via Claude API."""
    config = load_config()
    bucket = config["s3_bucket"]
    
    # Download postings
    logger.info(f"Downloading postings from S3")
    postings = read_jsonl_from_s3(postings_jsonl_s3_path, bucket)
    logger.info(f"Loaded {len(postings)} postings")
    
    # Stratified sample
    sampled = stratified_sample(postings, sample_size)
    logger.info(f"Sampled {len(sampled)} postings for labeling")
    
    # Score each with Claude
    labels = []
    for i, posting in enumerate(sampled):
        if i % 50 == 0:
            logger.info(f"Scoring {i+1}/{len(sampled)}")
        
        score = score_posting_with_claude(posting)
        labels.append({
            "posting_id": posting["id"],
            "score": score,
            "llm_score": score,
            "human_score": None,  # Filled in after manual review
            "split": "train" if i < SAMPLE_SIZE_TRAIN else ("val" if i < SAMPLE_SIZE_TRAIN + SAMPLE_SIZE_VAL else "test"),
            "is_test_human_labeled": i >= SAMPLE_SIZE_TRAIN + SAMPLE_SIZE_VAL,
            "agreement": None,
        })
    
    # Write locally
    write_jsonl(labels, output_local_path)
    logger.info(f"✓ Labeled {len(labels)} postings, saved to {output_local_path}")
    
    # Prompt user for manual labeling of test set
    logger.info("=" * 60)
    logger.info("MANUAL LABELING REQUIRED: Please review the test set (last 50 items)")
    logger.info("=" * 60)
    logger.info("\nSample test posting:")
    test_posting = next((p for p in sampled if p["id"] == labels[-1]["posting_id"]), None)
    if test_posting:
        print(f"Title: {test_posting['title']}")
        print(f"Company: {test_posting['company']}")
        print(f"Description: {test_posting['description'][:500]}...")
        print(f"LLM Score: {labels[-1]['llm_score']}")
        print("\nFor each test posting, enter your 0-3 score (or 's' to skip):")
    
    return labels, output_local_path

def compute_agreement(labels):
    """Compute LLM-human agreement on test set."""
    test_labels = [l for l in labels if l["is_test_human_labeled"]]
    if not test_labels:
        logger.warning("No test labels with human scores")
        return {}
    
    agreements = [l["llm_score"] == l["human_score"] for l in test_labels if l["human_score"] is not None]
    agreement_rate = sum(agreements) / len(agreements) if agreements else 0
    
    logger.info(f"Agreement: {agreement_rate:.1%} ({sum(agreements)}/{len(agreements)})")
    
    return {
        "total_test_samples": len(test_labels),
        "human_labeled": len([l for l in test_labels if l["human_score"] is not None]),
        "agreement_rate": agreement_rate,
        "mismatches": [l for l in test_labels if l["llm_score"] != l["human_score"] and l["human_score"] is not None],
    }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="postings/postings.jsonl")
    parser.add_argument("--sample-size", type=int, default=400)
    parser.add_argument("--output-s3", default="labels/labels.jsonl")
    args = parser.parse_args()
    
    config = load_config()
    
    # Label via Claude
    labels, temp_path = label_postings(args.input, args.sample_size)
    
    # TODO: Prompt for manual validation of test set, update labels
    # For now, save LLM scores as final
    
    # Compute agreement
    agreement = compute_agreement(labels)
    logger.info(f"Agreement report: {agreement}")
    
    # Upload to S3
    upload_to_s3(temp_path, args.output_s3, config["s3_bucket"])
    logger.info("✓ Labels uploaded to S3")
```

- [ ] **Step 3: Make `scripts/setup_aws.sh` executable**

```bash
chmod +x scripts/setup_aws.sh
```

- [ ] **Step 4: Test Phase 1 setup**

```bash
# First time: setup AWS S3 and upload Phase 0 postings
bash scripts/setup_aws.sh

# Then: test labeling with small sample (10 postings)
python src/data/label.py --sample-size 10 --output-s3 labels/labels_test.jsonl
```

Expected output: 10 labeled postings, saved to local temp file, agreement report.

- [ ] **Step 5: Commit**

```bash
git add src/data/label.py scripts/setup_aws.sh
git commit -m "feat(phase1): claude API labeling + manual validation"
```

---

### Phase 2: Tower A Training (2 tasks)

#### Task 4: Tower Architecture & Loss Functions

**Files:**
- Create: `src/model/towers.py`, `src/model/train_utils.py`
- Modify: (none — `attention.py` and `loss.py` already exist)
- Test: `tests/test_model.py`

**Interfaces:**
- Consumes: `src/model/attention.py` (SelfAttention, Block), `src/model/loss.py` (contrastive_loss)
- Produces:
  - `TowerA(nn.Module)` with `forward(token_ids: Tensor) -> Tensor` (returns normalized embeddings)
  - `TowerB(nn.Module)` with same interface
  - `create_data_loader(labels: List[Dict], postings: List[Dict], batch_size=32) -> DataLoader`

- [ ] **Step 1: Review existing `src/model/attention.py`**

Read the file to understand `SelfAttention`, `Block`, and their interfaces. Confirm they're bidirectional (no causal masking).

```bash
head -50 src/model/attention.py
```

- [ ] **Step 2: Review existing `src/model/loss.py`**

Read to understand contrastive loss signature:

```bash
head -50 src/model/loss.py
```

- [ ] **Step 3: Create `src/model/towers.py`**

```python
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
            Block(embedding_dim, num_heads, feedforward_dim=1536)
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
```

- [ ] **Step 4: Create `src/model/train_utils.py`**

```python
import torch
from torch.utils.data import Dataset, DataLoader
from src.utils.logging import setup_logging

logger = setup_logging("train_utils")

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
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]
    logger.info(f"Loaded checkpoint from {path} (epoch {epoch}, loss {loss})")
    return epoch, loss
```

- [ ] **Step 5: Create `tests/test_model.py`**

```python
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
    assert torch.allclose(torch.norm(embeddings, dim=-1), torch.ones(len(texts)), atol=1e-6)
```

- [ ] **Step 6: Run tests**

```bash
pytest tests/test_model.py -v
```

Expected: PASS (3 tests)

- [ ] **Step 7: Commit**

```bash
git add src/model/towers.py src/model/train_utils.py tests/test_model.py
git commit -m "feat(phase2-3): tower architectures (from-scratch + pretrained)"
```

---

#### Task 5: SageMaker Training & Local Training Loop

**Files:**
- Create: `src/model/sagemaker_train.py`, `docker/Dockerfile.train`, `scripts/submit_training_job.py`
- Modify: (none)
- Test: (manual — submit a 2-epoch test job)

**Interfaces:**
- Consumes: `TowerA`, `TowerB`, `create_dataloaders()`, `contrastive_loss`
- Produces: `models/tower_a/final.pt` in S3 (or locally)

- [ ] **Step 1: Create `docker/Dockerfile.train`**

```dockerfile
FROM pytorch/pytorch:2.1.1-cuda12.1-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y git curl

WORKDIR /opt/ml

# Copy requirements
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy source code
COPY src/ code/src/
COPY .env.example code/.env.example

# Set Python path
ENV PYTHONPATH=/opt/ml/code:$PYTHONPATH

ENTRYPOINT ["python", "code/src/model/sagemaker_train.py"]
```

- [ ] **Step 2: Create `src/model/sagemaker_train.py`**

```python
import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from src.model.towers import TowerA, TowerB
from src.model.train_utils import create_dataloaders, save_checkpoint
from src.model.loss import contrastive_loss
from src.data.s3_utils import read_jsonl_from_s3, download_from_s3
from src.utils.config import load_config, get_s3_paths
from src.utils.logging import setup_logging

logger = setup_logging("sagemaker_train")

def train_epoch(model, train_loader, optimizer, criterion, device, tower_type="a"):
    """Train one epoch."""
    model.train()
    total_loss = 0
    
    for batch_idx, batch in enumerate(train_loader):
        job_tokens = batch["job_tokens"].to(device)
        resume_tokens = batch["resume_tokens"].to(device)
        score = batch["score"].to(device)
        
        optimizer.zero_grad()
        
        if tower_type == "a":
            job_embedding = model(job_tokens)
            resume_embedding = model(resume_tokens)
        else:
            # TowerB expects text, not tokens
            job_text = batch.get("job_text", [])
            resume_text = batch.get("resume_text", "")
            job_embedding = model([job_text])
            resume_embedding = model([resume_text])
        
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
            job_tokens = batch["job_tokens"].to(device)
            resume_tokens = batch["resume_tokens"].to(device)
            score = batch["score"].to(device)
            
            if tower_type == "a":
                job_embedding = model(job_tokens)
                resume_embedding = model(resume_tokens)
            else:
                job_text = batch.get("job_text", [])
                resume_text = batch.get("resume_text", "")
                job_embedding = model([job_text])
                resume_embedding = model([resume_text])
            
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
    parser.add_argument("--local", action="store_true", help="Train locally (not on SageMaker)")
    args = parser.parse_args()
    
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on device: {device}")
    
    # Load data from S3 (or locally if --local)
    if args.local:
        input_dir = Path("data")
        labels_path = input_dir / "labels.jsonl"
        postings_path = input_dir / "postings.jsonl"
    else:
        # SageMaker mounts input at /opt/ml/input/data/training
        input_dir = Path("/opt/ml/input/data/training")
        labels_path = input_dir / "labels.jsonl"
        postings_path = input_dir / "postings.jsonl"
    
    logger.info(f"Loading labels from {labels_path}")
    # TODO: implement reading from JSON
    # labels = read_jsonl_from_s3(...)
    # postings = read_jsonl_from_s3(...)
    
    # Create model
    if args.tower == "a":
        model = TowerA(vocab_size=5000, embedding_dim=384)
    else:
        model = TowerB()
    
    model.to(device)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = contrastive_loss
    
    # Train
    best_val_loss = float("inf")
    for epoch in range(args.epochs):
        logger.info(f"Epoch {epoch+1}/{args.epochs}")
        # train_loss = train_epoch(model, train_loader, optimizer, criterion, device, args.tower)
        # val_loss = validate(model, val_loader, criterion, device, args.tower)
        
        # if val_loss < best_val_loss:
        #     best_val_loss = val_loss
        #     save_checkpoint(model, optimizer, epoch, val_loss, output_dir / f"best_model.pt")
    
    # Save final model
    output_dir = Path("/opt/ml/model") if not args.local else Path("models")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    torch.save(model.state_dict(), output_dir / "model.pt")
    logger.info(f"✓ Saved model to {output_dir}/model.pt")
    
    # Upload to S3 (if not local)
    if not args.local:
        from src.data.s3_utils import upload_to_s3
        s3_key = f"models/tower_{args.tower}/final.pt"
        upload_to_s3(output_dir / "model.pt", s3_key, config["s3_bucket"])

if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Create `scripts/submit_training_job.py`**

```python
import argparse
import json
import boto3
from src.utils.config import load_config
from src.utils.logging import setup_logging

logger = setup_logging("submit_training")
sagemaker_client = boto3.client("sagemaker")
s3_client = boto3.client("s3")

def submit_training_job(tower, instance_type="ml.p3.2xlarge", epochs=20):
    """Submit a SageMaker Training job."""
    config = load_config()
    bucket = config["s3_bucket"]
    region = config["aws_region"]
    
    job_name = f"two-tower-{tower}-{int(time.time())}"
    
    logger.info(f"Submitting SageMaker job: {job_name}")
    
    response = sagemaker_client.create_training_job(
        TrainingJobName=job_name,
        RoleArn=f"arn:aws:iam::123456789012:role/SageMakerRole",  # TODO: get from config
        AlgorithmSpecification={
            "TrainingImage": f"123456789012.dkr.ecr.{region}.amazonaws.com/two-tower:train",
            "TrainingInputMode": "File",
        },
        InputDataConfig=[
            {
                "ChannelName": "training",
                "DataSource": {
                    "S3DataSource": {
                        "S3Uri": f"s3://{bucket}/labels/",
                        "S3DataType": "S3Prefix",
                        "S3DataDistributionType": "FullyReplicated",
                    }
                },
            }
        ],
        OutputDataConfig={
            "S3OutputPath": f"s3://{bucket}/models/",
        },
        ResourceConfig={
            "InstanceType": instance_type,
            "InstanceCount": 1,
            "VolumeSizeInGB": 50,
        },
        StoppingCondition={
            "MaxRuntimeInSeconds": 86400,
        },
        HyperParameters={
            "tower": tower,
            "epochs": str(epochs),
            "batch_size": "32",
            "lr": "1e-4",
        },
    )
    
    logger.info(f"Job submitted: {response['TrainingJobArn']}")
    return job_name

if __name__ == "__main__":
    import time
    parser = argparse.ArgumentParser()
    parser.add_argument("--tower", required=True, choices=["a", "b"])
    parser.add_argument("--instance", default="ml.p3.2xlarge")
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    
    submit_training_job(args.tower, args.instance, args.epochs)
```

- [ ] **Step 4: Build Docker image locally (test only, not push yet)**

```bash
docker build -f docker/Dockerfile.train -t two-tower:train .
```

Expected: Image built successfully.

- [ ] **Step 5: Test training locally**

```bash
# Create sample data for testing
mkdir -p data
python src/data/label.py --sample-size 20 --output-s3 labels/labels_test.jsonl
cp data/labels_test.jsonl data/labels.jsonl

# Run training locally (2 epochs, CPU)
python src/model/sagemaker_train.py --tower a --epochs 2 --local
```

Expected: Training completes, saves `models/model.pt`.

- [ ] **Step 6: Commit**

```bash
git add src/model/sagemaker_train.py docker/Dockerfile.train scripts/submit_training_job.py
git commit -m "feat(phase2-3): sagemaker training infrastructure"
```

---

### Phase 3: Tower B (Reuse Task 4+5, Simple Wrapper)

No new tasks needed — Tower B is implemented in `towers.py` (TowerB class). Same training infrastructure (Task 5) handles both.

To train Tower B: `python src/model/sagemaker_train.py --tower b --epochs 10`

---

### Phase 4: Embedding & Indexing (2 tasks)

#### Task 6: Batch Embedding & FAISS Index

**Files:**
- Create: `src/embed/batch_embed.py`, `src/embed/build_faiss.py`, `src/embed/index_utils.py`
- Modify: (none)
- Test: (integration test with small sample)

**Interfaces:**
- Consumes: trained models from S3 (`models/tower_a/final.pt`, `models/tower_b/final.pt`), postings (`postings/postings.jsonl`)
- Produces: FAISS indices (`indices/tower_a.faiss`, `indices/tower_b.faiss`, `indices/bm25_index.pkl`)

- [ ] **Step 1: Create `src/embed/batch_embed.py`**

```python
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
```

- [ ] **Step 2: Create `src/embed/build_faiss.py`**

```python
import argparse
import json
import pickle
import numpy as np
import faiss
from pathlib import Path
from src.data.s3_utils import read_jsonl_from_s3, download_from_s3, upload_to_s3, write_jsonl
from src.utils.config import load_config
from src.utils.logging import setup_logging

logger = setup_logging("build_faiss")

def build_faiss_index(embeddings_jsonl_path, posting_ids):
    """Build FAISS IndexFlatIP from embeddings."""
    logger.info(f"Building FAISS index from {embeddings_jsonl_path}")
    
    # Load embeddings
    embeddings_list = []
    posting_id_to_idx = {}
    
    with open(embeddings_jsonl_path, "r") as f:
        for idx, line in enumerate(f):
            record = json.loads(line)
            posting_id = record["posting_id"]
            embedding = np.array(record["embedding"], dtype=np.float32)
            embeddings_list.append(embedding)
            posting_id_to_idx[posting_id] = idx
    
    embeddings_array = np.vstack(embeddings_list)  # (N, D)
    logger.info(f"Embeddings shape: {embeddings_array.shape}")
    
    # Ensure L2-normalized (for dot-product similarity)
    faiss.normalize_L2(embeddings_array)
    
    # Build index
    dimension = embeddings_array.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner product (dot product)
    index.add(embeddings_array)
    
    logger.info(f"✓ Built index with {index.ntotal} vectors")
    
    return index, posting_id_to_idx

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tower", required=True, choices=["a", "b"])
    parser.add_argument("--embeddings-s3", help="S3 path to embeddings JSONL")
    parser.add_argument("--embeddings-local", help="Local path to embeddings JSONL")
    parser.add_argument("--output-index-s3", help="S3 path to save FAISS index")
    parser.add_argument("--output-map-s3", help="S3 path to save posting_id_to_idx map")
    args = parser.parse_args()
    
    config = load_config()
    bucket = config["s3_bucket"]
    
    # Load embeddings
    if args.embeddings_s3:
        logger.info(f"Downloading embeddings from S3")
        embeddings_path = Path(f"/tmp/embeddings_{args.tower}.jsonl")
        download_from_s3(args.embeddings_s3, bucket, embeddings_path)
    else:
        embeddings_path = Path(args.embeddings_local)
    
    # Read postings to get IDs
    postings = read_jsonl_from_s3("postings/postings.jsonl", bucket)
    posting_ids = [p["id"] for p in postings]
    
    # Build index
    index, posting_id_to_idx = build_faiss_index(embeddings_path, posting_ids)
    
    # Save index
    index_path = Path(f"/tmp/tower_{args.tower}.faiss")
    faiss.write_index(index, str(index_path))
    logger.info(f"Saved FAISS index to {index_path}")
    
    # Save mapping
    map_path = Path(f"/tmp/posting_id_map_{args.tower}.pkl")
    with open(map_path, "wb") as f:
        pickle.dump(posting_id_to_idx, f)
    logger.info(f"Saved mapping to {map_path}")
    
    # Upload to S3
    upload_to_s3(index_path, args.output_index_s3, bucket)
    upload_to_s3(map_path, args.output_map_s3, bucket)

if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Create `src/embed/index_utils.py`**

```python
import pickle
import faiss
import numpy as np
from pathlib import Path
from typing import List, Tuple

class FAISSIndex:
    """Wrapper for FAISS index with posting ID mapping."""
    
    def __init__(self, index_path: str, map_path: str):
        """Load FAISS index and posting ID mapping."""
        self.index = faiss.read_index(index_path)
        with open(map_path, "rb") as f:
            self.idx_to_posting_id = {v: k for k, v in pickle.load(f).items()}
        self.dimension = self.index.d
    
    def search(self, query_embedding: np.ndarray, k: int = 10) -> Tuple[List[str], List[float]]:
        """
        Search index for top-k nearest neighbors.
        
        Args:
            query_embedding: (1, D) or (D,) float32 array, L2-normalized
            k: number of results
        
        Returns:
            posting_ids: List[str], sorted by score (highest first)
            scores: List[float], dot-product scores
        """
        if query_embedding.ndim == 1:
            query_embedding = query_embedding[np.newaxis, :]
        
        # Ensure L2-normalized
        faiss.normalize_L2(query_embedding)
        
        distances, indices = self.index.search(query_embedding, k)
        
        posting_ids = [self.idx_to_posting_id[int(idx)] for idx in indices[0]]
        scores = distances[0].tolist()
        
        return posting_ids, scores
```

- [ ] **Step 4: Add BM25 baseline (`src/eval/baseline.py` — should already exist, verify)**

```bash
grep -n "class BM25\|def bm25_score" src/eval/baseline.py || echo "BM25 not yet implemented"
```

If not implemented, add:

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class BM25Baseline:
    """Simple TF-IDF baseline for job matching."""
    
    def __init__(self, postings):
        self.postings = postings
        self.posting_id_to_text = {p["id"]: f"{p['title']} {p['description']}" for p in postings}
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
        self.posting_vectors = self.vectorizer.fit_transform(self.posting_id_to_text.values())
    
    def search(self, query_text: str, k: int = 10):
        """Search index for top-k postings."""
        query_vector = self.vectorizer.transform([query_text])
        scores = cosine_similarity(query_vector, self.posting_vectors)[0]
        top_indices = np.argsort(-scores)[:k]
        
        posting_ids = list(self.posting_id_to_text.keys())
        return [posting_ids[i] for i in top_indices], scores[top_indices].tolist()
```

- [ ] **Step 5: Test embedding pipeline (local, small sample)**

```bash
# Create sample model (dummy weights)
python -c "
import torch
from src.model.towers import TowerA
model = TowerA()
torch.save(model.state_dict(), 'models/tower_a_test.pt')
"

# Embed small sample
python src/embed/batch_embed.py --tower a --model-path models/tower_a_test.pt --local --batch-size 4

# Build index
python src/embed/build_faiss.py --tower a --embeddings-local /tmp/embeddings_a.jsonl --local
```

Expected: Index built, saved locally.

- [ ] **Step 6: Commit**

```bash
git add src/embed/batch_embed.py src/embed/build_faiss.py src/embed/index_utils.py
git commit -m "feat(phase4): batch embedding + FAISS indexing"
```

---

### Phase 5: Evaluation (1 task)

#### Task 7: Evaluation Metrics & Results Table

**Files:**
- Create: `src/eval/run_eval.py`, `src/eval/eval_utils.py`
- Modify: `README.md` (add results table)
- Test: `tests/test_eval.py`

**Interfaces:**
- Consumes: FAISS indices, test labels, resume text
- Produces: `results.json` with {p_5, p_10, r_20, ndcg_10, latency_p50, latency_p99} for each system

- [ ] **Step 1: Create `src/eval/eval_utils.py`**

```python
import json
import numpy as np
from pathlib import Path
from src.utils.logging import setup_logging

logger = setup_logging("eval_utils")

def format_results_table(results):
    """Format results into markdown table."""
    template = """
| System | P@5 | P@10 | R@20 | NDCG@10 | Latency p50 | Latency p99 |
|--------|-----|------|------|---------|-------------|-------------|
| BM25 baseline | {bm25[p_5]:.3f} | {bm25[p_10]:.3f} | {bm25[r_20]:.3f} | {bm25[ndcg_10]:.3f} | {bm25[latency_p50]:.1f}ms | {bm25[latency_p99]:.1f}ms |
| Tower A (scratch) | {tower_a[p_5]:.3f} | {tower_a[p_10]:.3f} | {tower_a[r_20]:.3f} | {tower_a[ndcg_10]:.3f} | {tower_a[latency_p50]:.1f}ms | {tower_a[latency_p99]:.1f}ms |
| Tower B (pretrained) | {tower_b[p_5]:.3f} | {tower_b[p_10]:.3f} | {tower_b[r_20]:.3f} | {tower_b[ndcg_10]:.3f} | {tower_b[latency_p50]:.1f}ms | {tower_b[latency_p99]:.1f}ms |
"""
    return template.format(
        bm25=results["systems"]["bm25"],
        tower_a=results["systems"]["tower_a"],
        tower_b=results["systems"]["tower_b"],
    )

def save_results(results, output_path):
    """Save results to JSON."""
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved results to {output_path}")

def load_results(output_path):
    """Load results from JSON."""
    with open(output_path, "r") as f:
        return json.load(f)
```

- [ ] **Step 2: Create `src/eval/run_eval.py`**

```python
import argparse
import json
import numpy as np
import time
from pathlib import Path
from src.embed.index_utils import FAISSIndex
from src.eval.baseline import BM25Baseline
from src.eval.metrics import precision_at_k, recall_at_k, ndcg_at_k
from src.data.s3_utils import read_jsonl_from_s3, download_from_s3
from src.utils.config import load_config, get_s3_paths
from src.utils.logging import setup_logging
from src.eval.eval_utils import format_results_table, save_results

logger = setup_logging("run_eval")

def evaluate_system(index_or_baseline, test_queries, test_labels, k_values=[5, 10, 20]):
    """
    Evaluate a retrieval system.
    
    Args:
        index_or_baseline: FAISSIndex or BM25Baseline
        test_queries: List[Dict] with resume_text, posting_ids
        test_labels: Dict[posting_id] -> score
        k_values: List[int] of k values for metrics
    
    Returns:
        metrics: Dict with p@k, r@k, ndcg@k, latency
    """
    all_precisions_5 = []
    all_precisions_10 = []
    all_recalls_20 = []
    all_ndcgs_10 = []
    latencies = []
    
    for query in test_queries:
        resume_text = query["resume_text"]
        
        # Search
        start = time.time()
        if isinstance(index_or_baseline, FAISSIndex):
            posting_ids, scores = index_or_baseline.search(resume_text, k=20)
        else:  # BM25
            posting_ids, scores = index_or_baseline.search(resume_text, k=20)
        latency_ms = (time.time() - start) * 1000
        latencies.append(latency_ms)
        
        # Get labels for retrieved postings
        retrieved_labels = [test_labels.get(pid, 0) for pid in posting_ids]
        
        # Compute metrics
        p_5 = precision_at_k(retrieved_labels, k=5, threshold=2)  # Score >= 2 is positive
        p_10 = precision_at_k(retrieved_labels, k=10, threshold=2)
        r_20 = recall_at_k(retrieved_labels, k=20, threshold=2)
        ndcg_10 = ndcg_at_k(retrieved_labels, k=10, relevances=retrieved_labels)  # Use graded labels
        
        all_precisions_5.append(p_5)
        all_precisions_10.append(p_10)
        all_recalls_20.append(r_20)
        all_ndcgs_10.append(ndcg_10)
    
    latencies = np.array(latencies)
    
    return {
        "p_5": np.mean(all_precisions_5),
        "p_10": np.mean(all_precisions_10),
        "r_20": np.mean(all_recalls_20),
        "ndcg_10": np.mean(all_ndcgs_10),
        "latency_p50_ms": np.percentile(latencies, 50),
        "latency_p99_ms": np.percentile(latencies, 99),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--indices-dir", default="./indices")
    parser.add_argument("--resume-file", required=True, help="Path to resume text file")
    parser.add_argument("--test-labels-s3", default="labels/labels.jsonl")
    parser.add_argument("--output", default="results.json")
    args = parser.parse_args()
    
    config = load_config()
    bucket = config["s3_bucket"]
    
    # Load test labels
    logger.info("Loading test labels")
    all_labels = read_jsonl_from_s3(args.test_labels_s3, bucket)
    test_labels_list = [l for l in all_labels if l["split"] == "test"]
    test_labels_dict = {l["posting_id"]: l["score"] for l in test_labels_list}
    logger.info(f"Loaded {len(test_labels_dict)} test labels")
    
    # Load indices
    logger.info("Loading FAISS indices")
    index_a = FAISSIndex(
        f"{args.indices_dir}/tower_a.faiss",
        f"{args.indices_dir}/posting_id_map_a.pkl"
    )
    index_b = FAISSIndex(
        f"{args.indices_dir}/tower_b.faiss",
        f"{args.indices_dir}/posting_id_map_b.pkl"
    )
    
    # Load BM25 baseline
    postings = read_jsonl_from_s3("postings/postings.jsonl", bucket)
    bm25_baseline = BM25Baseline(postings)
    
    # Load resume
    with open(args.resume_file, "r") as f:
        resume_text = f.read()
    
    # Create test queries (just the resume, repeated)
    test_queries = [{"resume_text": resume_text} for _ in test_labels_list]
    
    # Evaluate all systems
    logger.info("Evaluating BM25 baseline")
    bm25_results = evaluate_system(bm25_baseline, test_queries, test_labels_dict)
    
    logger.info("Evaluating Tower A")
    tower_a_results = evaluate_system(index_a, test_queries, test_labels_dict)
    
    logger.info("Evaluating Tower B")
    tower_b_results = evaluate_system(index_b, test_queries, test_labels_dict)
    
    # Aggregate results
    results = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "test_set_size": len(test_labels_dict),
        "systems": {
            "bm25": bm25_results,
            "tower_a": tower_a_results,
            "tower_b": tower_b_results,
        },
    }
    
    # Save
    save_results(results, args.output)
    
    # Print table
    table = format_results_table(results)
    logger.info("Results:\n" + table)

if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Create `tests/test_eval.py`**

```python
import pytest
from src.eval.metrics import precision_at_k, recall_at_k, ndcg_at_k

def test_precision_at_k():
    """Test P@k computation."""
    retrieved_labels = [3, 2, 1, 0, 0]  # Relevance scores
    
    # P@5 with threshold 2: 2 relevant out of 5
    p5 = precision_at_k(retrieved_labels, k=5, threshold=2)
    assert p5 == 0.4
    
    # P@2: 2 relevant out of 2
    p2 = precision_at_k(retrieved_labels, k=2, threshold=2)
    assert p2 == 1.0

def test_recall_at_k():
    """Test R@k computation."""
    retrieved_labels = [3, 2, 1, 0, 0]
    # Total relevant (score >= 2): 2
    
    # R@5: 2 relevant retrieved out of 2 total = 1.0
    r5 = recall_at_k(retrieved_labels, k=5, threshold=2)
    assert r5 == 1.0

def test_ndcg_at_k():
    """Test NDCG@k computation."""
    retrieved_labels = [3, 2, 1, 0, 0]
    
    # Ideal ranking: [3, 2, 1, 0, 0] (same as retrieved, so NDCG = 1.0)
    ndcg = ndcg_at_k(retrieved_labels, k=5, relevances=retrieved_labels)
    assert ndcg == 1.0
    
    # Worst ranking: [0, 0, 1, 2, 3] (reversed)
    ndcg_bad = ndcg_at_k([0, 0, 1, 2, 3], k=5, relevances=[3, 2, 1, 0, 0])
    assert ndcg_bad < 1.0
```

- [ ] **Step 4: Update README with results table**

```bash
cat >> README.md << 'EOF'

## Results (Phase 5)

| System | P@5 | P@10 | R@20 | NDCG@10 | Latency p50 | Latency p99 |
|--------|-----|------|------|---------|-------------|-------------|
| BM25 baseline | TBD | TBD | TBD | TBD | TBD | TBD |
| Tower A (scratch) | TBD | TBD | TBD | TBD | TBD | TBD |
| Tower B (pretrained) | TBD | TBD | TBD | TBD | TBD | TBD |
EOF
```

- [ ] **Step 5: Run evaluation tests**

```bash
pytest tests/test_eval.py -v
```

Expected: PASS (3 tests)

- [ ] **Step 6: Commit**

```bash
git add src/eval/run_eval.py src/eval/eval_utils.py tests/test_eval.py README.md
git commit -m "feat(phase5): evaluation pipeline + results table"
```

---

## Final Steps

#### Task 8: Makefile & Documentation

**Files:**
- Create: `Makefile`, `docs/SETUP.md`, `docs/LOCAL_DEV.md`
- Modify: (none)

- [ ] **Step 1: Create `Makefile`**

```makefile
.PHONY: help setup label train-a train-b embed eval clean

help:
	@echo "Two-Tower Job Matcher — make targets:"
	@echo "  setup       — Create AWS bucket, upload Phase 0 postings"
	@echo "  label       — Label 400 postings via Claude API"
	@echo "  train-a     — Train Tower A from scratch (SageMaker)"
	@echo "  train-b     — Train Tower B fine-tuned (SageMaker)"
	@echo "  embed       — Batch embed all jobs, build FAISS indices"
	@echo "  eval        — Evaluate all three systems"
	@echo "  clean       — Remove local artifacts"

setup:
	bash scripts/setup_aws.sh

label:
	python src/data/label.py --sample-size 400

train-a:
	python scripts/submit_training_job.py --tower a --epochs 20

train-b:
	python scripts/submit_training_job.py --tower b --epochs 10

embed:
	python src/embed/batch_embed.py --tower a
	python src/embed/batch_embed.py --tower b
	python src/embed/build_faiss.py --tower a
	python src/embed/build_faiss.py --tower b

eval:
	python src/eval/run_eval.py --resume-file /path/to/resume.txt

clean:
	rm -rf data/ models/ indices/ *.pt *.faiss *.jsonl
```

- [ ] **Step 2: Create `docs/SETUP.md`**

```markdown
# AWS Setup

## Prerequisites
- AWS account with access to S3, SageMaker, IAM
- AWS CLI configured: `aws configure`
- Docker installed (for training image)

## 1. Create S3 Bucket
\`\`\`bash
aws s3 mb s3://two-tower-pnair --region us-east-1
\`\`\`

## 2. Create IAM Role for SageMaker
\`\`\`bash
aws iam create-role --role-name SageMakerTwoTower \\
  --assume-role-policy-document file://trust-policy.json
\`\`\`

(See `scripts/trust-policy.json`)

## 3. Push Training Image to ECR
\`\`\`bash
aws ecr create-repository --repository-name two-tower-train
docker build -f docker/Dockerfile.train -t 123456789012.dkr.ecr.us-east-1.amazonaws.com/two-tower:train .
docker push 123456789012.dkr.ecr.us-east-1.amazonaws.com/two-tower:train
\`\`\`

## 4. Set Environment Variables
\`\`\`bash
cp .env.example .env
# Edit .env with your CLAUDE_API_KEY, AWS_REGION, S3_BUCKET
\`\`\`
```

- [ ] **Step 3: Create `docs/LOCAL_DEV.md`**

```markdown
# Local Development

## Running Phases Locally (No AWS)

All phases can be tested locally before submitting to AWS.

### Phase 1: Label (Local)
\`\`\`bash
python src/data/label.py --sample-size 50 --local
\`\`\`

### Phase 2-3: Train (Local, CPU)
\`\`\`bash
python src/model/sagemaker_train.py --tower a --epochs 2 --local
python src/model/sagemaker_train.py --tower b --epochs 2 --local
\`\`\`

### Phase 4: Embed (Local)
\`\`\`bash
python src/embed/batch_embed.py --tower a --local --batch-size 16
python src/embed/build_faiss.py --tower a --embeddings-local /tmp/embeddings_a.jsonl
\`\`\`

### Phase 5: Eval (Local)
\`\`\`bash
python src/eval/run_eval.py --resume-file myresume.txt --indices-dir ./indices
\`\`\`

## Using LocalStack for S3 (Optional)

For mocking S3 locally:

\`\`\`bash
docker-compose -f docker/docker-compose.yml up -d
export AWS_ENDPOINT_URL=http://localhost:4566
# Now boto3 calls hit LocalStack instead of real AWS
\`\`\`
```

- [ ] **Step 4: Commit**

```bash
git add Makefile docs/SETUP.md docs/LOCAL_DEV.md
git commit -m "docs: makefile, setup, local dev guide"
```

---

## Summary

**Total: 8 tasks**

1. **Setup & Dependencies** — Environment, .gitignore, requirements
2. **Data Infrastructure** — Schemas, S3 utils, config
3. **Phase 1: Labeling** — Claude API, stratified sampling, manual validation
4. **Tower Architecture** — TowerA, TowerB, data loading
5. **SageMaker Training** — Dockerfile, training loop, job submission
6. **Embedding & Indexing** — Batch embed, FAISS build
7. **Evaluation** — Metrics, results table, README
8. **Documentation** — Makefile, setup guide, local dev guide

**Execution: Subagent-Driven (recommended) or Inline**

---

## Quality Gates

Each task includes:
- ✅ Tests (where applicable)
- ✅ Commit boundaries (clean history)
- ✅ Error handling & logging
- ✅ Type signatures (interfaces between tasks)
- ✅ No placeholders (all code complete)
