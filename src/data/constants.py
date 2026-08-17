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
