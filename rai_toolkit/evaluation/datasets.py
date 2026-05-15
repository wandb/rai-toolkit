"""Dataset loading and creation utilities."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class DatasetLoader:
    """Load evaluation datasets from various formats.

    Supports CSV, JSON, and JSONL files. Datasets are returned as
    lists of dicts with standardized keys.

    Example::

        loader = DatasetLoader()

        # From file
        dataset = loader.from_file("datasets/rag_qa_dataset.csv")

        # From list of dicts
        dataset = loader.from_list([
            {"input": "What is X?", "context": "X is...", "expected": "X is Y"},
        ])
    """

    @staticmethod
    def from_file(path: str | Path) -> list[dict[str, Any]]:
        """Load a dataset from a file.

        Args:
            path: Path to CSV, JSON, or JSONL file.

        Returns:
            List of dicts with string values.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")

        suffix = path.suffix.lower()
        if suffix == ".csv":
            return DatasetLoader._load_csv(path)
        elif suffix == ".json":
            return DatasetLoader._load_json(path)
        elif suffix == ".jsonl":
            return DatasetLoader._load_jsonl(path)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

    @staticmethod
    def from_list(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Normalize a list of dicts to standard dataset format.

        Ensures all values are strings and maps common column name variations.
        """
        normalized = []
        for item in items:
            row = DatasetLoader._normalize_keys(item)
            normalized.append(_stringify_dataset_scalars(row))
        return normalized

    @staticmethod
    def _load_csv(path: Path) -> list[dict[str, Any]]:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = [DatasetLoader._normalize_keys(row) for row in reader]
        logger.info("Loaded %d rows from %s", len(rows), path)
        return rows

    @staticmethod
    def _load_json(path: Path) -> list[dict[str, Any]]:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            rows = [DatasetLoader._normalize_keys(item) for item in data]
        elif isinstance(data, dict) and "data" in data:
            rows = [DatasetLoader._normalize_keys(item) for item in data["data"]]
        else:
            raise ValueError("JSON must be a list of objects or {data: [...]}")
        logger.info("Loaded %d rows from %s", len(rows), path)
        return rows

    @staticmethod
    def _load_jsonl(path: Path) -> list[dict[str, Any]]:
        rows = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(DatasetLoader._normalize_keys(json.loads(line)))
        logger.info("Loaded %d rows from %s", len(rows), path)
        return rows

    @staticmethod
    def _normalize_keys(row: dict[str, Any]) -> dict[str, Any]:
        """Map common column name variations to standard keys."""
        key_mapping = {
            "question": "input",
            "query": "input",
            "prompt": "input",
            "user_input": "input",
            "answer": "expected",
            "ground_truth": "expected",
            "reference": "expected",
            "expected_output": "expected",
            "retrieved_context": "context",
            "source": "context",
            "documents": "context",
        }
        normalized: dict[str, Any] = {}
        for key, value in row.items():
            mapped_key = key_mapping.get(key.lower(), key.lower())
            normalized[mapped_key] = value if mapped_key == "policy_expectations" else (
                str(value) if value is not None else ""
            )
        return normalized


def _stringify_dataset_scalars(row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in row.items():
        if key == "policy_expectations":
            out[key] = value
        else:
            out[key] = str(value) if value is not None else ""
    return out
