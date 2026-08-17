import argparse
import json
import time
import boto3
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

sagemaker_client = boto3.client("sagemaker")
s3_client = boto3.client("s3")


def submit_training_job(tower, instance_type="ml.p3.2xlarge", epochs=20):
    """Submit a SageMaker Training job."""
    try:
        from src.utils.config import load_config
        config = load_config()
    except Exception as e:
        logger.warning(f"Could not load config: {e}")
        config = {
            "s3_bucket": "two-tower-pnair",
            "aws_region": "us-east-1",
        }

    bucket = config.get("s3_bucket", "two-tower-pnair")
    region = config.get("aws_region", "us-east-1")

    job_name = f"two-tower-{tower}-{int(time.time())}"

    logger.info(f"Submitting SageMaker job: {job_name}")

    # Note: This is a template. In practice, you need to:
    # 1. Have the Docker image pushed to ECR
    # 2. Have proper IAM role configured
    # 3. Update the image URI below
    response = sagemaker_client.create_training_job(
        TrainingJobName=job_name,
        RoleArn="arn:aws:iam::123456789012:role/SageMakerRole",  # TODO: get from config
        AlgorithmSpecification={
            "TrainingImage": "123456789012.dkr.ecr.us-east-1.amazonaws.com/two-tower:train",
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--tower", required=True, choices=["a", "b"])
    parser.add_argument("--instance", default="ml.p3.2xlarge")
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()

    submit_training_job(args.tower, args.instance, args.epochs)
