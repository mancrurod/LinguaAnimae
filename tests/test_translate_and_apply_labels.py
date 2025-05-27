"""
Unit tests for translation helpers in translate_and_apply_labels.py.

Usage:
pytest tests/test_translate_and_apply_labels.py
"""

import numpy as np
import pandas as pd
import sys
import os

# Path when running from tests/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/preprocessing')))
from translate_and_apply_labels import translate_theme, translate_emotion

# --- Tests for translate_theme ---

def test_translate_theme_single_known():
    """Test translation of a single known theme."""
    assert translate_theme("love") == "amor"
    assert translate_theme("faith") == "fe"
    assert translate_theme("FORGIVENESS") == "perdón"   # Case-insensitive

def test_translate_theme_multiple_known():
    """Test translation of multiple known themes, separated by semicolon."""
    assert translate_theme("love;hope") == "amor;esperanza"

def test_translate_theme_mixed():
    """Test mixed known and unknown themes."""
    assert translate_theme("love;peace") == "amor;peace"
    assert translate_theme("fear;custom") == "miedo;custom"

def test_translate_theme_spaces_and_case():
    """Test handling of spaces and case insensitivity."""
    assert translate_theme("  Love ;  Faith  ") == "amor;fe"

def test_translate_theme_unknown():
    """Test themes not present in THEME_MAP are left unchanged."""
    assert translate_theme("peace") == "peace"

def test_translate_theme_empty_and_nan():
    """Test empty strings and NaN/NA handling."""
    assert translate_theme("") == ""
    assert translate_theme(np.nan) == ""
    assert translate_theme(pd.NA) == ""

# --- Tests for translate_emotion ---

def test_translate_emotion_known():
    """Test translation of known emotion."""
    assert translate_emotion("joy") == "Alegría"
    assert translate_emotion("FEAR") == "Miedo"  # Case-insensitive

def test_translate_emotion_unknown():
    """Test unknown emotion is left unchanged."""
    assert translate_emotion("ecstasy") == "ecstasy"

def test_translate_emotion_spaces_and_case():
    """Test spaces and case-insensitivity for emotion translation."""
    assert translate_emotion("   sadness ") == "Tristeza"

def test_translate_emotion_empty_and_nan():
    """Test empty and NaN/NA emotion."""
    assert translate_emotion("") == ""
    assert translate_emotion(np.nan) == ""
    assert translate_emotion(pd.NA) == ""
