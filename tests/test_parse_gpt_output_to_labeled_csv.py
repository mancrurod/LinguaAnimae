"""
Unit test for parse_gpt_output_to_labeled_csv.py in Lingua Animae.

This test ensures that sample verses and GPT output are merged correctly,
handling unmatched IDs/verse_ids, and producing the expected output CSV.

Usage:
pytest tests/test_parse_gpt_output_to_labeled_csv.py
"""

import sys
import os
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from fine_tuning.parse_gpt_output_to_labeled_csv import main, setup_logger

def test_parse_gpt_output_merges_and_saves(tmp_path):
    # --- Prepare dummy sample verses ---
    df_samples = pd.DataFrame({
        "id": ["1", "2", "3"],
        "verse_id": ["Gen_1_1", "Gen_1_2", "Gen_1_3"],
        "verse": ["In the beginning...", "And the earth was...", "Let there be light."]
    })
    samples_file = tmp_path / "samples.csv"
    df_samples.to_csv(samples_file, index=False)

    # --- Prepare dummy GPT output ---
    df_gpt = pd.DataFrame({
        "id": ["1", "3", "99"],  # 99 no debería aparecer en output
        "verse_id": ["Gen_1_1", "Gen_1_3", "No_match"],
        "label": ["love", "hope", "error"]
    })
    gpt_output_file = tmp_path / "gpt_output.csv"
    df_gpt.to_csv(gpt_output_file, index=False, header=False)

    # --- Output file ---
    output_file = tmp_path / "labeled.csv"
    log_file = tmp_path / "test_parse.log"
    logger = setup_logger(log_file)

    # --- Run the main function ---
    main(
        samples_file=samples_file,
        gpt_output_file=gpt_output_file,
        output_file=output_file,
        logger=logger
    )

    # --- Check output ---
    assert output_file.exists()
    df_result = pd.read_csv(output_file)
    # Only id 1 and 3 should remain, merged on both id and verse_id
    assert set(df_result["id"].astype(str)) == {"1", "3"}
    # The output columns must be exactly as expected
    assert list(df_result.columns) == ["id", "verse_id", "verse", "label"]
    # The labels must match the correct IDs (both sides as str)
    labels = dict(zip(df_result["id"].astype(str), df_result["label"]))
    assert labels["1"] == "love"
    assert labels["3"] == "hope"

