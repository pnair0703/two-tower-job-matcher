# Two-Tower Job Matcher — Infrastructure Design

**Date:** 2026-08-16  
**Status:** Design (awaiting implementation)  
**Author:** Claude Code (Pranav's assistant)  
**Phases Covered:** 1–5  

---

## Executive Summary

Infrastructure for a five-phase job-matching system: collect → label → train two encoders → embed/index → evaluate. All phases run on AWS (SageMaker, S3, EC2) with local Docker development. Data lives in S3; computation is cloud-first to avoid stressing local CPU. Evaluation (the deliverable) runs locally against downloaded FAISS indices.

---

## 1. Architecture Overview

### System Topology

```
Phase 0 (complete)
├─ Postings JSONL (13K jobs from 4 ATS sources) → S3
└─ Normalized schema: {id, company, title, location, description, url, source}

Phase 1: Label
├─ Sample 400 postings (stratified: good/bad/mid)
├─ Score each 0-3 via Claude API with fixed rubric
├─ Hand-label 50 for validation, compute agreement
└─ Labels JSONL → S3

Phase 2-3: Train Towers
├─ Tower A: from-scratch transformer encoder (yours)
├─ Tower B: fine-tuned sentence-transformer
├─ Both trained contrastively with in-batch negatives
├─ SageMaker Training jobs (no local GPU stress)
└─ Model checkpoints → S3

Phase 4: Index
├─ Batch embed all 13K jobs with trained tower
├─ EC2 spot instance or Lambda (precompute everything)
├─ Build FAISS indices (one per tower + BM25 baseline)
└─ Indices → S3

Phase 5: Evaluate
├─ Download indices locally
├─ Embed your resume (at query time)
├─ Query all three systems, compute metrics (P@5, P@10, R@20, NDCG)
└─ Update README table with results
```

### Key Design Decisions

**1. Why S3 + batch, not SageMaker Endpoints?**
- Endpoints auto-scale (cost) and are overkill for a batch job
- S3 + precomputed embeddings keeps iteration fast and cheap
- Same interface: download index, query locally
- Interview answer: "precompute embeddings offline, FAISS for sub-millisecond queries"

**2. Why SageMaker Training for Phases 2-3?**
- No local GPU means slow training loops
- SageMaker handles checkpointing, logs, resource management
- You submit a job, wait for results, download from S3
- Clean separation: your code (versioned in git) vs. artifacts (in S3)

**3. Why Docker from the start?**
- Same code runs locally and in SageMaker/EC2
- Iterates locally with mocked S3 (LocalStack) before pushing to AWS
- Production deployment is trivial: push image to ECR, run as task

**4. Why simple labeling?**
- Phase 1 is not the research question; Phases 2-5 are
- Claude API + fixed rubric + 50 human labels is enough validation
- Keeps iteration fast, lets you move to the interesting part (encoders)

---

## 2. Data Schemas (All JSONL in S3)

### `postings.jsonl` (13K records, Phase 0 output)
```json
{
  "id": "greenhouse-elastic-mle-senior-01",
  "company": "elastic",
  "title": "Senior Machine Learning Engineer",
  "location": "remote",
  "description": "We're hiring a senior ML engineer to...",
  "url": "https://elastic.greenhouse.io/...",
  "source": "greenhouse"
}
```

### `labels.jsonl` (450 records: 400 train/val + 50 test, Phase 1 output)
```json
{
  "posting_id": "greenhouse-elastic-mle-senior-01",
  "score": 2,
  "llm_score": 2,
  "human_score": 2,
  "rubric_reasoning": "Technical fit (score 2): requires Kubernetes, you have it; wants 5y ML experience, you have 2y. Cultural fit high.",
  "split": "train",
  "is_test_human_labeled": false,
  "agreement": true
}
```

**Score meaning:**
- 3: Strong match (apply)
- 2: Moderate match (consider)
- 1: Weak match (maybe not)
- 0: No match (pass)

**Splits:** ~320 train, ~30 val, ~50 test (held-out human-labeled).

### `embeddings.jsonl` (13K records, Phase 4 output)
```json
{
  "posting_id": "greenhouse-elastic-mle-senior-01",
  "embedding_a": [0.123, -0.456, 0.789, ...],
  "embedding_b": [-0.234, 0.567, ...],
  "bm25_score": 0.75
}
```

**Note:** Embeddings stored as lists (not Base64). FAISS indices stored as binary `.faiss` files in S3.

### `results.json` (Phase 5 output, single file)
```json
{
  "timestamp": "2026-08-20T14:30:00Z",
  "test_set_size": 50,
  "systems": {
    "bm25": {
      "p_5": 0.68,
      "p_10": 0.62,
      "r_20": 0.84,
      "ndcg_10": 0.71,
      "latency_p50_ms": 2.3,
      "latency_p99_ms": 5.1
    },
    "tower_a": {
      "p_5": 0.52,
      "p_10": 0.48,
      "r_20": 0.76,
      "ndcg_10": 0.58,
      "latency_p50_ms": 1.2,
      "latency_p99_ms": 3.8
    },
    "tower_b": {
      "p_5": 0.80,
      "p_10": 0.75,
      "r_20": 0.92,
      "ndcg_10": 0.82,
      "latency_p50_ms": 1.1,
      "latency_p99_ms": 3.5
    }
  }
}
```

---

## 3. S3 Bucket Organization

**Bucket:** `two-tower-{user-id}` (or `two-tower-pnair` for simplicity)

```
two-tower-pnair/
├── postings/
│   └── postings.jsonl              # 13K jobs, Phase 0 output (already exists)
├── labels/
│   ├── sampled_400.jsonl           # sampled subset for labeling
│   ├── labels.jsonl                # scored + human-validated, Phase 1 output
│   └── agreement_report.json       # LLM vs human agreement stats
├── models/
│   ├── tower_a/
│   │   ├── config.json             # architecture + hyperparams
│   │   ├── checkpoint_epoch_10.pt  # intermediate checkpoints
│   │   └── final.pt                # best model, Phase 2 output
│   └── tower_b/
│       ├── config.json
│       ├── checkpoint_epoch_5.pt
│       └── final.pt                # Phase 3 output
├── embeddings/
│   ├── jobs_tower_a.jsonl          # all 13K jobs embedded with tower A
│   ├── jobs_tower_b.jsonl          # all 13K jobs embedded with tower B
│   └── meta.json                   # {total_count, dimensions, avg_norm}
├── indices/
│   ├── tower_a.faiss               # built from tower A embeddings
│   ├── tower_b.faiss               # built from tower B embeddings
│   ├── bm25_index.pkl              # scikit-learn TfidfVectorizer
│   └── posting_id_map.pkl          # FAISS index → posting_id mapping
├── results/
│   ├── 2026-08-20_eval_results.json  # Phase 5 metrics
│   └── eval_history.jsonl          # appended results for each run
└── logs/
    ├── training_tower_a.log        # SageMaker job logs (also in CloudWatch)
    ├── training_tower_b.log
    └── embedding_job.log
```

**Versioning:** Enable S3 versioning on `models/` and `indices/` to track iterations.

**Lifecycle:** Set expiration on logs (30 days) to keep costs low.

---

## 4. Phase-by-Phase Infrastructure

### Phase 1: Label (AWS Lambda + Claude API)

**Infrastructure:**
- Local: `src/data/label.py` reads postings JSONL, calls Claude API, writes labels
- AWS: Lambda function (optional, for scale) or just run locally—labeling is IO-bound, not CPU-bound
- Dependencies: `anthropic` SDK, `pandas`, `boto3`

**Workflow:**
```bash
python src/data/label.py \
  --input s3://two-tower-pnair/postings/postings.jsonl \
  --sample-size 400 \
  --output s3://two-tower-pnair/labels/labels.jsonl
```

**What it does:**
1. Load postings from S3
2. Stratified sample: 100 high-skill roles, 100 non-tech roles, 200 ambiguous
3. For each posting, call Claude with rubric prompt → score 0-3
4. Save labels to S3
5. Download 50 labeled pairs, prompt user to hand-label
6. Compute agreement: LLM vs human, report in `agreement_report.json`

**Error handling:**
- Retry Claude API calls (exponential backoff)
- Cache responses to disk during development (don't re-query)
- Validation: ensure all scores are 0-3, no nulls

---

### Phase 2: Train Tower A (SageMaker Training)

**Infrastructure:**
- Docker image built locally, pushed to ECR
- SageMaker Training job: `ml.p3.2xlarge` (1 GPU, $3.06/hr, terminates after training)
- Input data: `postings.jsonl` + `labels.jsonl` from S3
- Output: model checkpoint to S3

**Dockerfile (`docker/Dockerfile.train`):**
```dockerfile
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04
RUN apt-get update && apt-get install -y python3.11 python3-pip
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY src/ /opt/ml/code/src/
WORKDIR /opt/ml/code
ENTRYPOINT ["python", "src/model/sagemaker_train.py"]
```

**Entry point (`src/model/sagemaker_train.py`):**
```python
import argparse
import boto3
import torch
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--tower", default="a", help="tower a or b")
parser.add_argument("--s3-bucket", default="two-tower-pnair")
parser.add_argument("--epochs", type=int, default=20)
args = parser.parse_args()

# SageMaker mounts input data at /opt/ml/input/data/
# and expects output at /opt/ml/model/
s3 = boto3.client("s3")
local_data = Path("/opt/ml/input/data/training")
output_dir = Path("/opt/ml/model")

# Download training data
s3.download_file(args.s3_bucket, "labels/labels.jsonl", "labels.jsonl")

# Train tower
tower = train_tower(tower=args.tower, epochs=args.epochs)

# Upload checkpoint
torch.save(tower.state_dict(), output_dir / "model.pt")
```

**Invocation (from your laptop):**
```bash
python scripts/submit_training_job.py --tower a --instance ml.p3.2xlarge --epochs 20
# Waits for job to finish, downloads checkpoint from S3
```

**What it does:**
1. Read training labels from S3
2. Create DataLoader with positive/negative pairs
3. Train tower with contrastive loss (in-batch negatives)
4. Save checkpoint every epoch (versioned in S3)
5. Log metrics to CloudWatch (loss, val accuracy, learning rate)
6. On completion, notify via CloudWatch

**Error handling:**
- Checkpointing: save every epoch so interrupted jobs don't lose progress
- Validation split: track val loss, save best model (early stopping optional)
- Data validation: check label distributions before training starts

---

### Phase 3: Train Tower B (SageMaker Training)

**Infrastructure:** Identical to Phase 2, except:
- Uses `sentence-transformers` library (pretrained `all-MiniLM-L6-v2`)
- Smaller model, faster training (~30 min vs. hours for scratch Tower A)
- Same SageMaker job template

**Entry point difference:**
```python
from sentence_transformers import SentenceTransformer, models

# Load pretrained
model = SentenceTransformer("all-MiniLM-L6-v2")
# Wrap in custom tower interface (so Tower A and B are swappable)
tower = TowerB(model)
# Train with same contrastive loss
```

---

### Phase 4: Embed & Index (EC2 Spot or Local)

**Infrastructure:**
- Option A (cheap): EC2 spot instance (`g4dn.xlarge`, ~$0.35/hr, 1 GPU, 30 min runtime)
- Option B (simplest): Run locally in Docker, takes 1-2 hours on CPU
- Input: model checkpoints from S3, postings JSONL
- Output: embeddings JSONL, FAISS indices → S3

**Script (`src/embed/batch_embed.py`):**
```bash
python src/embed/batch_embed.py \
  --model-a s3://two-tower-pnair/models/tower_a/final.pt \
  --model-b s3://two-tower-pnair/models/tower_b/final.pt \
  --postings s3://two-tower-pnair/postings/postings.jsonl \
  --batch-size 64 \
  --output-dir s3://two-tower-pnair/embeddings/
```

**What it does:**
1. Load both trained towers
2. Load all 13K postings
3. Embed in batches (batch_size=64) with both towers
4. Stream embeddings to S3 (don't hold all in RAM)
5. Build FAISS index from embeddings: `IndexFlatIP` (dot-product similarity, L2-normalized)
6. Save `.faiss` files + `posting_id_map.pkl` (FAISS index ↔ posting ID)
7. Time it: report p50/p99 latency per query

**Error handling:**
- Disk space: stream embeddings directly to S3, don't cache
- Memory: load models one at a time, embed in chunks
- Validation: sanity-check embedding dimensions, norms

---

### Phase 5: Evaluate (Local Python)

**Infrastructure:**
- Run locally (no GPU needed, only CPU-based FAISS queries)
- Input: FAISS indices + test labels (both from S3)
- Output: `results.json`, update README table

**Script (`src/eval/run_eval.py`):**
```bash
python src/eval/run_eval.py \
  --indices-dir ./indices/ \
  --test-labels s3://two-tower-pnair/labels/labels.jsonl \
  --resume path/to/resume.txt \
  --output results.json
```

**What it does:**
1. Download indices + test labels from S3
2. Embed your resume with both towers
3. Query all three systems (BM25, Tower A, Tower B) for top-20 jobs
4. For each system, compute:
   - Precision@5, @10
   - Recall@20
   - NDCG@10 (using 0-3 labels as grades)
   - Latency (p50, p99) over 100 queries
5. Save results to `results.json`
6. Update README results table

**Error handling:**
- Index mismatch: validate posting_id_map length matches FAISS index
- Missing labels: only evaluate on test set (50 human-labeled)
- Resume edge case: if resume embedding is all-zeros, catch and debug

---

## 5. Code Layout & Structure

```
two_tower/
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── fetch.py          (Phase 0, complete)
│   │   ├── normalize.py       (Phase 0, complete)
│   │   ├── label.py           (Phase 1, NEW)
│   │   ├── s3_utils.py        (shared, NEW)
│   │   └── constants.py       (schemas, constants, NEW)
│   ├── model/
│   │   ├── __init__.py
│   │   ├── attention.py       (bidirectional blocks, from transformer repo)
│   │   ├── towers.py          (Tower A, Tower B, NEW)
│   │   ├── loss.py            (contrastive loss, complete)
│   │   ├── sagemaker_train.py (SageMaker entry point, NEW)
│   │   └── train_utils.py     (checkpointing, validation, NEW)
│   ├── embed/
│   │   ├── __init__.py
│   │   ├── batch_embed.py     (embed all jobs, NEW)
│   │   ├── build_faiss.py     (FAISS index construction, NEW)
│   │   └── index_utils.py     (FAISS query wrapper, NEW)
│   ├── eval/
│   │   ├── __init__.py
│   │   ├── metrics.py         (P@k, R@k, NDCG, complete)
│   │   ├── baseline.py        (BM25/TF-IDF, complete)
│   │   ├── run_eval.py        (main evaluation loop, NEW)
│   │   └── eval_utils.py      (result formatting, NEW)
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── config.py          (read .env, AWS config, NEW)
│   │   ├── logging.py         (CloudWatch + local logging, NEW)
│   │   └── io_utils.py        (S3 download/upload, NEW)
│   └── index.py               (legacy? or keep for reference?)
├── docker/
│   ├── Dockerfile.train       (SageMaker training image, NEW)
│   ├── Dockerfile.embed       (batch embedding image, NEW)
│   └── docker-compose.yml     (local dev with LocalStack, NEW)
├── scripts/
│   ├── submit_training_job.py (submit SageMaker job, NEW)
│   ├── monitor_training.py    (tail job logs, NEW)
│   ├── download_checkpoint.py (fetch model from S3, NEW)
│   └── run_locally.sh         (dev workflow, NEW)
├── notebooks/
│   ├── 01_eda.ipynb           (explore postings + labels, NEW)
│   └── 02_results.ipynb       (plot Phase 5 metrics, NEW)
├── data/
│   └── raw/                   (Phase 0 postings, existing)
├── docs/
│   ├── SETUP.md               (AWS account setup, S3, IAM, NEW)
│   ├── LOCAL_DEV.md           (Docker, LocalStack, iterating locally, NEW)
│   └── superpowers/
│       └── specs/
│           └── 2026-08-16-two-tower-infrastructure-design.md
├── tests/
│   ├── test_data.py           (validate JSONL schemas, NEW)
│   ├── test_model.py          (Tower A/B forward pass, NEW)
│   ├── test_eval.py           (metrics correctness, NEW)
│   └── conftest.py            (pytest fixtures, S3 mocks, NEW)
├── .env.example               (CLAUDE_API_KEY, AWS_REGION, S3_BUCKET, NEW)
├── .gitignore                 (data/, *.pt, *.faiss, .env, NEW)
├── Makefile                   (shortcuts for common tasks, NEW)
├── requirements.txt           (updated with boto3, sagemaker, torch, etc.)
├── README.md                  (updated with results table)
└── BUILD_TWO_TOWER.md         (existing guide, complementary)
```

---

## 6. Local Development Workflow

### Setup

```bash
# 1. Clone repo, create venv
cd two_tower
python3 -m venv .venv && source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure AWS credentials
aws configure
# OR set env vars: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION

# 4. Create .env
cp .env.example .env
# Edit: CLAUDE_API_KEY, S3_BUCKET=two-tower-pnair

# 5. Create S3 bucket and folder structure
aws s3 mb s3://two-tower-pnair --region us-east-1
```

### Iteration Loop (Example: Phase 1 Labeling)

```bash
# 1. Test locally with mock S3 (LocalStack)
docker-compose -f docker/docker-compose.yml up -d
export AWS_ENDPOINT_URL=http://localhost:4566

# 2. Upload sample postings to local S3
python scripts/setup_local_s3.py

# 3. Run labeling script against local S3
python src/data/label.py \
  --input s3://two-tower-pnair/postings/sample_100.jsonl \
  --sample-size 10 \
  --output /tmp/labels_test.jsonl \
  --local

# 4. Inspect output, validate schema
python -c "import json; [print(json.loads(l)) for l in open('/tmp/labels_test.jsonl')]"

# 5. Once satisfied, run against real AWS S3
unset AWS_ENDPOINT_URL
python src/data/label.py \
  --input s3://two-tower-pnair/postings/postings.jsonl \
  --sample-size 400 \
  --output s3://two-tower-pnair/labels/labels.jsonl

# 6. Download results, hand-validate 50 labels
```

### CI/CD (GitHub Actions, optional)

```yaml
# .github/workflows/test.yml
name: test
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - run: pip install -r requirements.txt
      - run: pytest tests/ -v
      - run: python -m black --check src/
      - run: python -m pylint src/
```

---

## 7. AWS Configuration & Permissions

### S3 Bucket Policy

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "AWS": "arn:aws:iam::123456789012:root"
      },
      "Action": "s3:*",
      "Resource": [
        "arn:aws:s3:::two-tower-pnair",
        "arn:aws:s3:::two-tower-pnair/*"
      ]
    }
  ]
}
```

### IAM Policy for SageMaker

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "sagemaker:CreateTrainingJob",
        "sagemaker:DescribeTrainingJob",
        "sagemaker:StopTrainingJob"
      ],
      "Resource": "*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:DeleteObject"
      ],
      "Resource": "arn:aws:s3:::two-tower-pnair/*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "ecr:GetDownloadUrlForLayer",
        "ecr:BatchGetImage",
        "ecr:BatchCheckLayerAvailability"
      ],
      "Resource": "*"
    }
  ]
}
```

### CloudWatch Logs

- SageMaker jobs automatically log to `/aws/sagemaker/TrainingJobs/{job_name}`
- Local code logs to stdout + file (`logs/train.log`)
- Configure logging in `src/utils/logging.py` to send to CloudWatch

---

## 8. Monitoring & Error Handling

### What to Monitor

| Component | Metric | Alert Threshold |
|-----------|--------|-----------------|
| Claude API (Phase 1) | Rate limit hits | > 5 retries |
| SageMaker job (Phase 2-3) | Training loss plateau | no improvement for 5 epochs |
| SageMaker job | Out of memory | any OOM error |
| Embedding job (Phase 4) | Embedding dimension mismatch | embedding != 384 or 768 |
| FAISS index | Index corruption | index size != 13K |
| Eval (Phase 5) | Missing test labels | < 50 labels |

### Retry Logic

**Claude API (Phase 1):**
```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2))
def score_posting_with_claude(posting, rubric):
    response = client.messages.create(...)
    return parse_score(response)
```

**S3 uploads (all phases):**
```python
def upload_to_s3(local_path, s3_key, bucket, max_retries=3):
    for attempt in range(max_retries):
        try:
            s3.upload_file(local_path, bucket, s3_key)
            return
        except ClientError as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)
```

---

## 9. Cost Estimation

| Component | Unit | Quantity | Cost |
|-----------|------|----------|------|
| Claude API (Phase 1) | tokens | ~500K (400 jobs × ~1.2K tokens) | $2–3 |
| SageMaker Training (Phase 2) | ml.p3.2xlarge-hours | ~4 | $12 |
| SageMaker Training (Phase 3) | ml.p3.2xlarge-hours | ~1 | $3 |
| EC2 spot (Phase 4) | g4dn.xlarge-hours | ~0.5 | $0.18 |
| S3 storage (indices, embeddings, models) | GB-months | ~2 | $0.05–0.10 |
| **Total** | | | ~$17–18 |

**Note:** Costs are one-time. Subsequent iterations (re-training) reuse the same infrastructure.

---

## 10. Known Limitations & Trade-offs

| Trade-off | Choice | Why |
|-----------|--------|-----|
| Local vs. cloud training | Cloud (SageMaker) | Avoids CPU stress; scales easily |
| Simple vs. sophisticated labeling | Simple (Claude API) | Iteration speed; Phase 1 not the focus |
| FAISS index type | `IndexFlatIP` (flat) | 13K jobs fit in memory; linear search is fast enough |
| Model serving | Precompute + download | Simpler than endpoints; faster iteration than containerizing |
| Evaluation split | 50 human-labeled test | Small, but honest; manually reviewed |

---

## 11. Success Criteria

By end of Phase 5, you should have:

- ✅ `labels.jsonl` with 400 scores + 50 human-validated samples
- ✅ `models/tower_a/final.pt` and `models/tower_b/final.pt` trained and saved
- ✅ `indices/tower_a.faiss`, `tower_b.faiss`, and BM25 index built and queried locally
- ✅ `results.json` with P@5, P@10, R@20, NDCG@10 for all three systems
- ✅ README results table populated (likely: BM25 > Tower A, Tower B > both)
- ✅ All code in git, reproducible (pull repo → make eval → get results)

---

## 12. Next Steps

1. **Implement Phase 1 labeling** — Claude API loop, hand-label 50, compute agreement
2. **Dockerize & test training** — build image, test SageMaker job submission
3. **Train towers** — submit jobs, monitor logs, download checkpoints
4. **Implement embedding & FAISS** — batch embed, build indices
5. **Run evaluation** — download indices, compute metrics, update README

Each phase is independent; parallel work is possible (e.g., write Phase 2-3 code while Phase 1 labeling is in progress).

---

**Design Approved:** (awaiting user sign-off)  
**Implementation Plan:** (follows after design review)
