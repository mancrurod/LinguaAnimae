"""
Parse the KJV Bible from an OSIS XML file with split <verse osisID> and <verse eID> tags.

Each CSV will contain the following columns:
book, chapter, verse, subtitle (empty), text, source_url

Output path: data/raw/bible_kjv/
"""

# ========================
# === IMPORTS & CONFIG ===
# ========================

import xml.etree.ElementTree as ET
import pandas as pd
from pathlib import Path
from typing import Dict, List

# ========================
# === CONSTANTS ==========
# ========================

OSIS_FILE = Path("data/raw/bible_kjv/eng-kjv.osis.xml")
OUTPUT_DIR = Path("data/raw/bible_kjv")
SOURCE_URL = "https://github.com/seven1m/open-bibles"

# Map OSIS codes to full book names in canonical order
BOOK_ORDER = [
    "genesis", "exodus", "leviticus", "numbers", "deuteronomy", "joshua", "judges", "ruth",
    "1_samuel", "2_samuel", "1_kings", "2_kings", "1_chronicles", "2_chronicles", "ezra", "nehemiah",
    "esther", "job", "psalms", "proverbs", "ecclesiastes", "song_of_solomon", "isaiah", "jeremiah",
    "lamentations", "ezekiel", "daniel", "hosea", "joel", "amos", "obadiah", "jonah", "micah", "nahum",
    "habakkuk", "zephaniah", "haggai", "zechariah", "malachi", "matthew", "mark", "luke", "john",
    "acts", "romans", "1_corinthians", "2_corinthians", "galatians", "ephesians", "philippians",
    "colossians", "1_thessalonians", "2_thessalonians", "1_timothy", "2_timothy", "titus", "philemon",
    "hebrews", "james", "1_peter", "2_peter", "1_john", "2_john", "3_john", "jude", "revelation"
]

OSIS_TO_NAME = {
    "Gen": "genesis", "Exod": "exodus", "Lev": "leviticus", "Num": "numbers", "Deut": "deuteronomy",
    "Josh": "joshua", "Judg": "judges", "Ruth": "ruth", "1Sam": "1_samuel", "2Sam": "2_samuel",
    "1Kgs": "1_kings", "2Kgs": "2_kings", "1Chr": "1_chronicles", "2Chr": "2_chronicles",
    "Ezra": "ezra", "Neh": "nehemiah", "Esth": "esther", "Job": "job", "Ps": "psalms",
    "Prov": "proverbs", "Eccl": "ecclesiastes", "Song": "song_of_solomon", "Isa": "isaiah",
    "Jer": "jeremiah", "Lam": "lamentations", "Ezek": "ezekiel", "Dan": "daniel", "Hos": "hosea",
    "Joel": "joel", "Amos": "amos", "Obad": "obadiah", "Jonah": "jonah", "Mic": "micah",
    "Nah": "nahum", "Hab": "habakkuk", "Zeph": "zephaniah", "Hag": "haggai", "Zech": "zechariah",
    "Mal": "malachi", "Matt": "matthew", "Mark": "mark", "Luke": "luke", "John": "john", "Acts": "acts",
    "Rom": "romans", "1Cor": "1_corinthians", "2Cor": "2_corinthians", "Gal": "galatians",
    "Eph": "ephesians", "Phil": "philippians", "Col": "colossians", "1Thess": "1_thessalonians",
    "2Thess": "2_thessalonians", "1Tim": "1_timothy", "2Tim": "2_timothy", "Titus": "titus",
    "Phlm": "philemon", "Heb": "hebrews", "Jas": "james", "1Pet": "1_peter", "2Pet": "2_peter",
    "1John": "1_john", "2John": "2_john", "3John": "3_john", "Jude": "jude", "Rev": "revelation"
}

# ========================
# === MAIN FUNCTION ======
# ========================

def extract_verses() -> None:
    """Parse the OSIS XML and save one CSV per book with full verse content."""
    print("📖 Parsing OSIS XML (split verse mode)...")
    tree = ET.parse(OSIS_FILE)
    root = tree.getroot()

    verses_by_book: Dict[str, List[Dict[str, str]]] = {}

    current_verse_id = None
    current_text = ""
    capturing = False
    book = chapter = verse = ""

    # Traverse all nodes in order
    for elem in root.iter():
        tag = elem.tag.split("}")[-1]  # Remove namespace

        if tag == "verse" and "osisID" in elem.attrib:
            # Start capturing a new verse
            current_verse_id = elem.attrib["osisID"]
            parts = current_verse_id.split(".")
            if len(parts) != 3:
                continue
            osis_book, chapter, verse = parts
            book = OSIS_TO_NAME.get(osis_book)
            if not book:
                continue
            capturing = True
            current_text = (elem.tail or "").strip()

        elif tag == "verse" and "eID" in elem.attrib:
            # End of current verse
            if capturing and current_verse_id:
                row = {
                    "book": book,
                    "chapter": chapter,
                    "verse": verse,
                    "subtitle": "",
                    "text": " ".join(current_text.split()),
                    "source_url": SOURCE_URL
                }
                if book not in verses_by_book:
                    verses_by_book[book] = []
                verses_by_book[book].append(row)
            capturing = False
            current_verse_id = None
            current_text = ""

        elif capturing:
            # Accumulate text and tail while inside a verse
            if elem.text:
                current_text += " " + elem.text
            if elem.tail:
                current_text += " " + elem.tail

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"📁 Saving CSVs to: {OUTPUT_DIR}")

    for i, book in enumerate(BOOK_ORDER, start=1):
        verses = verses_by_book.get(book)
        if not verses:
            continue
        df = pd.DataFrame(verses)
        filename = f"{i}_{book}.csv"
        df.to_csv(OUTPUT_DIR / filename, index=False)
        print(f"✅ Saved {filename} with {len(df)} verses")

    print("🚀 Extraction complete.")

# ========================
# === ENTRY POINT ========
# ========================

if __name__ == "__main__":
    extract_verses()
