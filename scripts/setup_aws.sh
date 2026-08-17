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
