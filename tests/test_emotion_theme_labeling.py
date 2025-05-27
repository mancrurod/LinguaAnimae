"""
Unit tests for emotion labeling utilities in Lingua Animae.

This module tests the core logic for batch emotion classification
in the emotion_theme_labeling script, ensuring robust and predictable
label assignment for downstream processing.

Covers:
- Batch emotion classification (classify_batch)
- Handling of error cases and empty batches

Usage:
pytest tests/test_emotion_theme_labeling.py
"""

import sys
import os
import pytest

# Importar la función desde src/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from modeling.emotion_theme_labeling import classify_batch

class DummyLogger:
    def error(self, *args, **kwargs): pass

# Mock del classifier: simula salida tipo HuggingFace
def mock_classifier(texts):
    # Simula clasificación de emociones en lote
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
        elif text == "error":
            raise Exception("Simulated error")
        else:
            outputs.append([
                {"label": "neutral", "score": 0.5}
            ])
    return outputs

def test_classify_batch_labels_majority():
    result = classify_batch(["joy", "anger"], classifier=mock_classifier, logger=DummyLogger())
    assert result == ["joy", "anger"]

def test_classify_batch_default_label():
    result = classify_batch(["unknown"], classifier=mock_classifier, logger=DummyLogger())
    assert result == ["neutral"]

def test_classify_batch_empty_input():
    result = classify_batch([], classifier=mock_classifier, logger=DummyLogger())
    assert result == []

def test_classify_batch_handles_error():
    # Cuando el classifier falla, la función debe devolver "error" para cada texto
    def broken_classifier(texts):
        raise Exception("Broken!")
    result = classify_batch(["a", "b"], classifier=broken_classifier, logger=DummyLogger())
    assert result == ["error", "error"]
