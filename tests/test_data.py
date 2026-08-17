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
