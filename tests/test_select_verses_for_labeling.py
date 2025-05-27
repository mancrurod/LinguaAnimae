"""
Unit tests for verse selection utilities in Lingua Animae.

This module tests the logic for random selection of unique verses,
ensuring de-duplication, correct exclusion of already labeled verses,
and robust output formatting for manual annotation.

Covers:
- Reading and combining input CSVs
- Deduplication by verse_id
- Exclusion of already labeled verses
- Sampling correct number of unique verses

Usage:
pytest tests/test_select_verses_for_labeling.py
"""

import sys
import os
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from fine_tuning.select_verses_for_labeling import main as select_main, setup_logger

def create_csv(path, df):
    df.to_csv(path, index=False)

class DummyLogger:
    def info(self, *args, **kwargs): pass
    def warning(self, *args, **kwargs): pass
    def error(self, *args, **kwargs): pass

def test_select_verses_dedup_and_exclude(tmp_path):
    # Prepare input directory and CSVs
    input_dir = tmp_path / "input"
    input_dir.mkdir()

    # Two files with overlapping verses
    df1 = pd.DataFrame({
        "verse_id": ["Genesis_1_1", "Genesis_1_2"],
        "text": ["In the beginning...", "And the earth was..."]
    })
    df2 = pd.DataFrame({
        "verse_id": ["Genesis_1_2", "Genesis_1_3"],
        "text": ["And the earth was...", "And God said..."]
    })
    create_csv(input_dir / "a.csv", df1)
    create_csv(input_dir / "b.csv", df2)

    # Already labeled file (exclude Genesis_1_1)
    labeled_dir = tmp_path / "eval"
    labeled_dir.mkdir()
    already_labeled = pd.DataFrame({"verse_id": ["Genesis_1_1"]})
    create_csv(labeled_dir / "emotion_verses_labeled_combined.csv", already_labeled)

    # Output location
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    output_file = output_dir / "verses_sample.csv"
    log_file = tmp_path / "test_select_verses.log"

    logger = setup_logger(log_file)

    # Run main function
    select_main(
        input_dir=input_dir,
        output_file=output_file,
        n_samples=2,
        existing_labels_path=labeled_dir / "emotion_verses_labeled_combined.csv",
        logger=logger
    )

    # Check output file
    result = pd.read_csv(output_file)
    # Only Genesis_1_2 and Genesis_1_3 should remain (Genesis_1_1 excluded, Genesis_1_2 deduped)
    assert set(result["verse_id"]) == {"Genesis_1_2", "Genesis_1_3"}
    # Output columns should match expected
    assert list(result.columns) == ["id", "verse_id", "verse"]
    # IDs should be sequential and unique
    assert set(result["id"]) == set(range(len(result)))

def test_select_verses_empty_input(tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_file = tmp_path / "out.csv"
    log_file = tmp_path / "test_empty.log"
    logger = setup_logger(log_file)
    # Should not fail, but output file should not exist or be empty
    select_main(input_dir=input_dir, output_file=output_file, n_samples=5, logger=logger)
    assert not output_file.exists() or pd.read_csv(output_file).empty
