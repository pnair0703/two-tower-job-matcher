# AWS Setup

## Prerequisites
- AWS account with access to S3, SageMaker, IAM
- AWS CLI configured: `aws configure`
- Docker installed (for training image)

## 1. Create S3 Bucket
```bash
aws s3 mb s3://two-tower-pnair --region us-east-1
```

## 2. Create IAM Role for SageMaker
```bash
aws iam create-role --role-name SageMakerTwoTower \
  --assume-role-policy-document file://trust-policy.json
```

(See `scripts/trust-policy.json`)

## 3. Push Training Image to ECR
```bash
aws ecr create-repository --repository-name two-tower-train
docker build -f docker/Dockerfile.train -t 123456789012.dkr.ecr.us-east-1.amazonaws.com/two-tower:train .
docker push 123456789012.dkr.ecr.us-east-1.amazonaws.com/two-tower:train
```

## 4. Set Environment Variables
```bash
cp .env.example .env
# Edit .env with your CLAUDE_API_KEY, AWS_REGION, S3_BUCKET
```
