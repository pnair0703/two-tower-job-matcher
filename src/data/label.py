import json
import random
from pathlib import Path
from anthropic import Anthropic
from src.utils.config import load_config, get_s3_paths
from src.utils.logging import setup_logging
from src.data.s3_utils import read_jsonl_from_s3, write_jsonl, upload_to_s3
from src.data.constants import LABEL_RUBRIC, SAMPLE_SIZE_TRAIN, SAMPLE_SIZE_VAL, SAMPLE_SIZE_TEST

logger = setup_logging("label")
client = Anthropic(api_key=load_config()["claude_api_key"])

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
Location: {posting.get('location', 'N/A')}
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
