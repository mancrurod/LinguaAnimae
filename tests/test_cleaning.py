"""
Unit tests for text cleaning utilities in Lingua Animae.

This module tests the core text cleaning logic used in the annotation pipeline.
Usage:
pytest tests/test_cleaning.py

"""

import sys
import os
import pandas as pd
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from preprocessing.cleaning import clean_text, normalize_unicode, validate_row, generate_id



def test_clean_text_basic():
    """
    Test that clean_text trims whitespace, collapses internal spaces, and preserves line breaks—including blank lines.
    """
    # Example with leading/trailing whitespace, tabs, multiple spaces, and blank lines
    raw_text = "   Jesús  dijo:    \t ¡Hola!  \n\n  En verdad   te digo.\t "
    expected = "Jesús dijo: ¡Hola!\n\nEn verdad te digo."
    cleaned = clean_text(raw_text)
    assert cleaned == expected, f"Expected: {expected!r}, got: {cleaned!r}"

def test_clean_text_handles_empty_string():
    """
    Test that clean_text returns an empty string when input is empty or whitespace.
    """
    assert clean_text("") == ""
    assert clean_text("   \t  ") == ""

def test_clean_text_handles_nan():
    """
    Test that clean_text returns an empty string if input is NaN (pandas missing value).
    """
    import numpy as np
    import pandas as pd
    assert clean_text(np.nan) == ""
    assert clean_text(pd.NA) == ""

def test_clean_text_preserves_single_line():
    """
    Test that a simple line is cleaned but preserved.
    """
    input_line = "  Dios es amor. "
    expected = "Dios es amor."
    assert clean_text(input_line) == expected

def test_normalize_unicode_basic():
    """
    Test that unicode quotation marks and dashes are normalized to ASCII equivalents.
    """
    raw = '“Blessed are the ‘meek’ – they shall inherit the earth.”'
    expected = '"Blessed are the \'meek\' - they shall inherit the earth."'
    assert normalize_unicode(raw) == expected

def test_normalize_unicode_mixed():
    """
    Test normalization on a text with multiple types of unicode punctuation.
    """
    raw = '—“Hello”—said ‘Juan’—'
    expected = '-"Hello"-said \'Juan\'-'
    assert normalize_unicode(raw) == expected

def test_normalize_unicode_ascii_unchanged():
    """
    Test that ASCII punctuation is not altered.
    """
    raw = '"Standard" -- punctuation!'
    expected = '"Standard" -- punctuation!'
    assert normalize_unicode(raw) == expected

def test_normalize_unicode_empty_string():
    """
    Test that an empty string returns an empty string.
    """
    assert normalize_unicode("") == ""

def test_normalize_unicode_nan():
    """
    Test that NaN or missing values return an empty string.
    """
    import numpy as np
    import pandas as pd
    assert normalize_unicode(np.nan) == ""
    assert normalize_unicode(pd.NA) == ""

# --- Tests for validate_row ---

def test_validate_row_valid():
    row = pd.Series({"book": "John", "chapter": 3, "verse": 16})
    assert validate_row(row) is True

def test_validate_row_book_empty():
    row = pd.Series({"book": "   ", "chapter": 1, "verse": 1})
    assert validate_row(row) is False

def test_validate_row_chapter_not_numeric():
    row = pd.Series({"book": "John", "chapter": "abc", "verse": 5})
    assert validate_row(row) is False

def test_validate_row_verse_zero():
    row = pd.Series({"book": "John", "chapter": 1, "verse": 0})
    assert validate_row(row) is False

def test_validate_row_negative():
    row = pd.Series({"book": "John", "chapter": -1, "verse": 5})
    assert validate_row(row) is False

def test_validate_row_strings():
    row = pd.Series({"book": "John", "chapter": "2", "verse": "5"})
    assert validate_row(row) is True

def test_validate_row_nan():
    row = pd.Series({"book": np.nan, "chapter": np.nan, "verse": np.nan})
    assert validate_row(row) is False

# --- Tests for generate_id ---

def test_generate_id_valid():
    row = pd.Series({"book": "Genesis", "chapter": 1, "verse": 1})
    assert generate_id(row) == "Genesis_1_1"

def test_generate_id_spaces_and_case():
    row = pd.Series({"book": "  song of solomon  ", "chapter": 3, "verse": 4})
    assert generate_id(row) == "Song_Of_Solomon_3_4"

def test_generate_id_zero_or_negative():
    row = pd.Series({"book": "John", "chapter": 0, "verse": 5})
    assert generate_id(row) == "INVALID_ID"
    row = pd.Series({"book": "John", "chapter": 2, "verse": -2})
    assert generate_id(row) == "INVALID_ID"

def test_generate_id_non_numeric():
    row = pd.Series({"book": "John", "chapter": "abc", "verse": 5})
    assert generate_id(row) == "INVALID_ID"

def test_generate_id_missing():
    row = pd.Series({"book": None, "chapter": None, "verse": None})
    assert generate_id(row) == "INVALID_ID"