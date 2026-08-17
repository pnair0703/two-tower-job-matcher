# Local Development

## Running Phases Locally (No AWS)

All phases can be tested locally before submitting to AWS.

### Phase 1: Label (Local)
```bash
python src/data/label.py --sample-size 50 --local
```

### Phase 2-3: Train (Local, CPU)
```bash
python src/model/sagemaker_train.py --tower a --epochs 2 --local
python src/model/sagemaker_train.py --tower b --epochs 2 --local
```

### Phase 4: Embed (Local)
```bash
python src/embed/batch_embed.py --tower a --local --batch-size 16
python src/embed/build_faiss.py --tower a --embeddings-local /tmp/embeddings_a.jsonl
```

### Phase 5: Eval (Local)
```bash
python src/eval/run_eval.py --resume-file myresume.txt --indices-dir ./indices
```

## Using LocalStack for S3 (Optional)

For mocking S3 locally:

```bash
docker-compose -f docker/docker-compose.yml up -d
export AWS_ENDPOINT_URL=http://localhost:4566
# Now boto3 calls hit LocalStack instead of real AWS
```
