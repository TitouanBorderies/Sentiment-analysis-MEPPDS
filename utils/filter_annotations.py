import json
from collections import defaultdict, Counter
import os

# Définir les chemins d'annotation
ANNOTATION_PATH = os.environ.get("ANNOTATION_PATH", "annotations/annotations.jsonl")
CLEAN_PATH = os.environ.get("CLEAN_ANNOTATION_PATH", "annotations/annotations_clean.jsonl")


def load_annotations(path):
    """Load annotations from a JSONL file and return as a list."""
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def filter_majority_annotations(annotations):
    """Filter annotations by majority label per unique text."""
    grouped = defaultdict(list)

    for ann in annotations:
        grouped[ann["text"]].append(ann["label"])

    filtered = []
    for text, labels in grouped.items():
        label_counts = Counter(labels)
        total_votes = sum(label_counts.values())
        most_common_label, votes = label_counts.most_common(1)[0]

        # Keep only those with a majority vote
        if votes > total_votes / 2:
            filtered.append({"text": text, "label": most_common_label})

    return filtered


def save_annotations(path, annotations):
    """Save filtered annotations to a JSONL file."""
    with open(path, "w", encoding="utf-8") as f:
        for ann in annotations:
            f.write(json.dumps(ann, ensure_ascii=False) + "\n")


def filter_and_save_clean_annotations():
    """Load, filter, and save cleaned annotations."""
    annotations = load_annotations(ANNOTATION_PATH)
    filtered = filter_majority_annotations(annotations)
    save_annotations(CLEAN_PATH, filtered)
