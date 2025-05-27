"""
Unit tests for parse_osis_kjv.py in Lingua Animae.

These tests cover the OSIS XML parsing and CSV generation logic,
including handling of edge cases, book mapping, and text cleaning.

Covers:
- Parsing minimal OSIS XML with valid and invalid book codes
- Handling of missing or malformed osisID attributes
- Output dictionary format and book mapping

Usage:
pytest tests/test_parse_osis_kjv.py
"""

import sys
import os
import pytest
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from scraping.parse_osis_kjv import parse_osis_verses, setup_logger

class DummyLogger:
    def __init__(self):
        self.errors = []
        self.warnings = []
    def error(self, msg): self.errors.append(msg)
    def warning(self, msg): self.warnings.append(msg)

def create_osis_xml(path, verses):
    """
    Helper to write a minimal OSIS XML file.
    `verses` is a list of tuples: (osisID, text)
    """
    root = ET.Element("osis")
    osis_text = ET.SubElement(root, "osisText")
    div = ET.SubElement(osis_text, "div")
    for osisID, text in verses:
        v_start = ET.SubElement(div, "verse", osisID=osisID)
        v_start.tail = text
        v_end = ET.SubElement(div, "verse", eID=osisID)
    tree = ET.ElementTree(root)
    tree.write(path, encoding="utf-8", xml_declaration=True)

def test_parse_osis_verses_valid_and_unknown(tmp_path):
    # Create XML with one known and one unknown book code
    xml_path = tmp_path / "mini.osis.xml"
    verses = [
        ("Gen.1.1", "In the beginning..."),          # Should be mapped to "genesis"
        ("FakeBook.1.1", "This should not appear")   # Unknown book code
    ]
    create_osis_xml(xml_path, verses)
    logger = DummyLogger()
    result = parse_osis_verses(xml_path, logger)
    # Only "genesis" must be present
    assert "genesis" in result
    assert len(result["genesis"]) == 1
    row = result["genesis"][0]
    assert row["book"] == "genesis"
    assert row["chapter"] == "1"
    assert row["verse"] == "1"
    assert "beginning" in row["text"]
    # Unknown book should trigger a warning
    assert any("Unknown OSIS book code" in w for w in logger.warnings)

def test_parse_osis_verses_invalid_osisid(tmp_path):
    # Create XML with invalid osisID
    xml_path = tmp_path / "bad.osis.xml"
    verses = [("Gen.1", "Short osisID")]
    create_osis_xml(xml_path, verses)
    logger = DummyLogger()
    result = parse_osis_verses(xml_path, logger)
    # No verses should be parsed
    assert not result

def test_parse_osis_verses_missing_file(tmp_path):
    xml_path = tmp_path / "nope.xml"
    logger = DummyLogger()
    result = parse_osis_verses(xml_path, logger)
    assert result == {}
    assert any("File not found" in e for e in logger.errors)
