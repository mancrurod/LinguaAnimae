"""
Unit tests for theme labeling utilities in Lingua Animae.

This module tests the core logic for zero-shot theme classification
and batch processing of Bible verse texts, ensuring robust labeling 
and data integrity before analysis or downstream modeling.

Covers:
- Batch theme classification logic (thresholding and label assignment).
- Handling of multiple themes, thresholds, and edge cases.
- DataFrame and file output structure (for integration tests).

Usage:
pytest tests/test_theme_labeling.py
"""

import sys
import os
import pytest

# Import the function as in your merge test: src.modeling.theme_labeling
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from modeling.theme_labeling import classify_batch

# === MOCKS ===

def mock_classifier(texts, candidate_labels, multi_label):
    """
    Simulate the output of a HuggingFace zero-shot classifier for different test cases.
    """
    if isinstance(texts, str):
        texts = [texts]
    outputs = []
    for text in texts:
        if text == "love only":
            outputs.append({
                "labels": ["love", "faith", "hope"],
                "scores": [0.8, 0.2, 0.1]
            })
        elif text == "multi label":
            outputs.append({
                "labels": ["love", "faith", "hope"],
                "scores": [0.8, 0.75, 0.1]
            })
        elif text == "none":
            outputs.append({
                "labels": ["love", "faith", "hope"],
                "scores": [0.5, 0.6, 0.2]
            })
        elif text == "edge":
            outputs.append({
                "labels": ["love", "faith", "hope"],
                "scores": [0.7, 0.69, 0.7]
            })
    return outputs if len(outputs) > 1 else outputs[0]

# === TESTS ===

def test_classify_batch_single_label():
    result = classify_batch(
        ["love only"],
        classifier=mock_classifier,
        labels=["love", "faith", "hope"],
        threshold=0.7
    )
    assert result == ["love"]

def test_classify_batch_multi_label():
    result = classify_batch(
        ["multi label"],
        classifier=mock_classifier,
        labels=["love", "faith", "hope"],
        threshold=0.7
    )
    assert result == ["love;faith"]

def test_classify_batch_none_pass():
    result = classify_batch(
        ["none"],
        classifier=mock_classifier,
        labels=["love", "faith", "hope"],
        threshold=0.7
    )
    assert result == [""]

def test_classify_batch_edge_threshold():
    result = classify_batch(
        ["edge"],
        classifier=mock_classifier,
        labels=["love", "faith", "hope"],
        threshold=0.7
    )
    assert result == ["love;hope"]

def test_classify_batch_multiple_texts():
    result = classify_batch(
        ["love only", "multi label", "none"],
        classifier=mock_classifier,
        labels=["love", "faith", "hope"],
        threshold=0.7
    )
    assert result == ["love", "love;faith", ""]
