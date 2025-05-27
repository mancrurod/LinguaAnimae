"""
Unit tests for labeling pipeline utilities in Lingua Animae.

This module tests the core logic for batch emotion and theme classification,
ensuring that emotion/theme assignment, multi-label thresholds, and error handling 
work as expected before full pipeline execution.

Covers:
- Batch emotion classification (classify_emotion_batch)
- Batch theme classification (classify_theme_batch)
- Handling of error cases and empty batches

Usage:
pytest tests/test_labeling_pipeline.py
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from modeling.labeling_pipeline import classify_emotion_batch, classify_theme_batch

class DummyLogger:
    def error(self, *args, **kwargs): pass

# === Emotion classifier mocks ===
def mock_emotion_classifier(texts):
    outputs = []
    for text in texts:
        if text == "joy":
            outputs.append([
                {"label": "joy", "score": 0.9},
                {"label": "sadness", "score": 0.1}
            ])
        elif text == "anger":
            outputs.append([
                {"label": "anger", "score": 0.8},
                {"label": "joy", "score": 0.2}
            ])
        else:
            outputs.append([
                {"label": "neutral", "score": 0.7}
            ])
    return outputs

def test_classify_emotion_batch_majority():
    result = classify_emotion_batch(["joy", "anger"], classifier=mock_emotion_classifier, logger=DummyLogger())
    assert result == ["joy", "anger"]

def test_classify_emotion_batch_empty():
    result = classify_emotion_batch([], classifier=mock_emotion_classifier, logger=DummyLogger())
    assert result == []

def test_classify_emotion_batch_error():
    def broken_classifier(texts):
        raise Exception("Classifier crashed")
    result = classify_emotion_batch(["fail", "fail"], classifier=broken_classifier, logger=DummyLogger())
    assert result == ["error", "error"]

# === Theme classifier mocks ===
def mock_theme_classifier(texts, candidate_labels, multi_label):
    outputs = []
    for text in texts:
        if text == "multi":
            outputs.append({
                "labels": candidate_labels,
                "scores": [0.8, 0.75, 0.1, 0.2, 0.9]  # Assumes 5 labels
            })
        elif text == "none":
            outputs.append({
                "labels": candidate_labels,
                "scores": [0.5, 0.6, 0.2, 0.1, 0.2]
            })
        elif text == "edge":
            outputs.append({
                "labels": candidate_labels,
                "scores": [0.7, 0.69, 0.7, 0.2, 0.5]  # Two at threshold
            })
    return outputs if len(outputs) > 1 else outputs[0]

def test_classify_theme_batch_multi():
    result = classify_theme_batch(
        ["multi"],
        classifier=mock_theme_classifier,
        labels=["love", "faith", "hope", "forgiveness", "fear"],
        threshold=0.7,
        logger=DummyLogger()
    )
    assert result == ["love;faith;fear"]

def test_classify_theme_batch_none():
    result = classify_theme_batch(
        ["none"],
        classifier=mock_theme_classifier,
        labels=["love", "faith", "hope", "forgiveness", "fear"],
        threshold=0.7,
        logger=DummyLogger()
    )
    assert result == [""]

def test_classify_theme_batch_edge():
    result = classify_theme_batch(
        ["edge"],
        classifier=mock_theme_classifier,
        labels=["love", "faith", "hope", "forgiveness", "fear"],
        threshold=0.7,
        logger=DummyLogger()
    )
    assert result == ["love;hope"]

def test_classify_theme_batch_error():
    def broken_classifier(texts, candidate_labels, multi_label):
        raise Exception("Classifier crashed")
    result = classify_theme_batch(
        ["fail", "fail"],
        classifier=broken_classifier,
        labels=["love", "faith"],
        threshold=0.7,
        logger=DummyLogger()
    )
    assert result == ["error", "error"]
