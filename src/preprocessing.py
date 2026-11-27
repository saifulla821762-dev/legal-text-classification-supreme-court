"""Dataset loading, cleaning, and reproducible train/test preparation."""
from __future__ import annotations

import re
from collections import Counter

from sklearn.model_selection import train_test_split
from textacy.datasets import SupremeCourt


def clean_text(text: str) -> str:
    """Lowercase, normalize whitespace, and remove punctuation."""
    text = text.lower()
    text = re.sub(r"\\s+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\\s]", "", text)
    return text.strip()


def load_records() -> tuple[list[str], list[int]]:
    """Load Supreme Court texts and discard unusable issue-area labels."""
    dataset = SupremeCourt()
    dataset.download()
    texts, labels = [], []
    records = list(dataset.records())
    for text, metadata in records:
        issue_area = metadata.get("issue_area")
        issue = metadata.get("issue")
        if issue_area in (None, -1) or issue in (None, "none"):
            continue
        texts.append(clean_text(text))
        labels.append(int(issue_area))
    print(f"Records loaded: {len(records)}")
    print(f"Records retained: {len(texts)}")
    print(f"Class distribution: {dict(Counter(labels))}")
    return texts, labels


def split_data(texts: list[str], labels: list[int], test_size: float = 0.20,
               random_state: int = 42):
    """Create a shared reproducible split; stratify only if all classes allow it."""
    counts = Counter(labels)
    stratify = labels if counts and min(counts.values()) >= 2 else None
    return train_test_split(
        texts, labels, test_size=test_size, random_state=random_state,
        stratify=stratify,
    )
