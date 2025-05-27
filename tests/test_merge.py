"""
Unit tests for the CSV merging utility in Lingua Animae.

This module tests the combine_cleaned_csvs function used to merge cleaned CSV files
from a processed subfolder into a single ordered file for downstream modeling and analysis.

Covers:
- Correct file order and concatenation based on filename prefixes.
- Removal of original 'id' columns and creation of a new sequential 'id'.
- Preservation of all required columns, especially 'verse_id'.
- Handling of empty or malformed CSV files.

Usage:
pytest tests/test_merge.py
"""

import pandas as pd
import pytest
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from preprocessing.merge import combine_cleaned_csvs

def test_combine_cleaned_csvs_merges_correctly(tmp_path):
    """
    Integration test for combine_cleaned_csvs:
    - Combines multiple cleaned CSVs in the correct order
    - Removes original 'id' columns
    - Generates a new sequential 'id' column
    - Preserves 'verse_id'
    """
    # Simulate 'processed_subdir' with fake CSVs
    processed_subdir = tmp_path / "bible_test"
    processed_subdir.mkdir(parents=True)

    # CSV 1 (with numeric prefix for order)
    df1 = pd.DataFrame({
        "id": [1, 2],
        "book": ["Genesis", "Genesis"],
        "chapter": [1, 1],
        "verse": [1, 2],
        "verse_id": ["Genesis_1_1", "Genesis_1_2"],
        "text": ["In the beginning", "And the earth was..."],
        "theme": ["love", "hope"],
        "emotion": ["joy", "sadness"]
    })
    df1.to_csv(processed_subdir / "1_genesis_cleaned.csv", index=False)

    # CSV 2
    df2 = pd.DataFrame({
        "id": [1],
        "book": ["Exodus"],
        "chapter": [1],
        "verse": [1],
        "verse_id": ["Exodus_1_1"],
        "text": ["Now these are the names..."],
        "theme": ["faith"],
        "emotion": ["anger"]
    })
    df2.to_csv(processed_subdir / "2_exodus_cleaned.csv", index=False)

    # CSV 3 (empty, should be skipped)
    df3 = pd.DataFrame(columns=df1.columns)
    df3.to_csv(processed_subdir / "3_leviticus_cleaned.csv", index=False)

    # Setup a dummy logger
    class DummyLogger:
        def info(self, *args, **kwargs): pass
        def error(self, *args, **kwargs): pass
        def warning(self, *args, **kwargs): pass

    logger = DummyLogger()

    # Run merging
    combine_cleaned_csvs(processed_subdir, logger)

    # Check output CSV
    output_csv = processed_subdir / "bible_test.csv"
    assert output_csv.exists()

    merged = pd.read_csv(output_csv)
    # Should have 3 rows (2 from Genesis, 1 from Exodus)
    assert len(merged) == 3

    # Check 'id' is sequential and starts at 1
    assert list(merged['id']) == [1, 2, 3]

    # Check 'verse_id' values are preserved in order
    assert list(merged['verse_id']) == [
        "Genesis_1_1",
        "Genesis_1_2",
        "Exodus_1_1"
    ]

    # Check columns (id is first, verse_id is present, original id is gone)
    assert 'id' in merged.columns
    assert 'verse_id' in merged.columns
    assert 'book' in merged.columns
    assert 'text' in merged.columns
    assert 'emotion' in merged.columns
    assert 'theme' in merged.columns
    assert 'id' not in merged.columns[1:]  # 'id' solo la primera posición

    # Check that there are no empty rows
    assert not merged.isnull().all(axis=1).any()
