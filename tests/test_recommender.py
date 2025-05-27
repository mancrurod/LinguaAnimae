"""
Unit tests for recommender module in Lingua Animae.

This module tests core normalization and recommendation logic for Bible verse suggestions,
ensuring correct matching and filtering by emotion and theme.

Covers:
- Text normalization (normalize)
- Verse recommendation by emotion/theme (recommend_verses)
- Edge cases: empty corpus, no matches, robust filtering

Usage:
pytest tests/test_recommender.py
"""

import sys
import os
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from src.interface.recommender import normalize, recommend_verses

class DummyLogger:
    def warning(self, *args, **kwargs): pass
    def info(self, *args, **kwargs): pass
    def error(self, *args, **kwargs): pass

def test_normalize_basic():
    assert normalize(" Jesús ") == "jesus"
    assert normalize("Éxodo") == "exodo"
    assert normalize("FAITH") == "faith"
    assert normalize("SeÑor") == "senor"

def test_recommend_verses_basic():
    df = pd.DataFrame({
        "book": ["John", "Genesis"],
        "emotion": ["joy", "joy"],
        "theme": ["love;faith", "hope"],
        "verse_id": ["John_3_16", "Genesis_1_1"],
        "text": ["For God so loved the world...", "In the beginning..."]
    })
    # Debe devolver solo las filas que coinciden con emotion="joy" y theme="love"
    recs = recommend_verses(
        df, emotion="joy", theme="love", lang="en", max_results=2, logger=DummyLogger()
    )
    assert not recs.empty
    # Solo la fila con theme "love;faith" debe aparecer
    assert (recs["verse_id"] == "John_3_16").any()

def test_recommend_verses_no_match():
    df = pd.DataFrame({
        "book": ["John"],
        "emotion": ["joy"],
        "theme": ["hope"],
        "verse_id": ["John_3_16"],
        "text": ["For God so loved the world..."]
    })
    recs = recommend_verses(
        df, emotion="sadness", theme="love", lang="en", max_results=2, logger=DummyLogger()
    )
    # No debe haber recomendaciones
    assert recs.empty

def test_recommend_verses_empty_df():
    df = pd.DataFrame(columns=["book", "emotion", "theme", "verse_id", "text"])
    recs = recommend_verses(
        df, emotion="joy", theme="love", lang="en", max_results=2, logger=DummyLogger()
    )
    assert recs.empty

def test_recommend_verses_spanish_mapping(monkeypatch):
    # Simulate EMOTION_MAP y THEME_MAP for Spanish
    import src.interface.recommender as recommender_mod
    monkeypatch.setattr(recommender_mod, "EMOTION_MAP", {"alegria": "joy"})
    monkeypatch.setattr(recommender_mod, "THEME_MAP", {"amor": "love"})
    df = pd.DataFrame({
        "book": ["Juan"],
        "emotion": ["joy"],
        "theme": ["love"],
        "verse_id": ["Juan_3_16"],
        "text": ["Tanto amó Dios al mundo..."]
    })
    recs = recommender_mod.recommend_verses(
        df, emotion="alegria", theme="amor", lang="es", max_results=1, logger=DummyLogger()
    )
    assert not recs.empty
    assert (recs["verse_id"] == "Juan_3_16").any()

