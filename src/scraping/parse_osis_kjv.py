"""
Parse the KJV Bible from an OSIS XML file with split <verse osisID> and <verse eID> tags.

Each CSV will contain the following columns:
book, chapter, verse, subtitle (empty), text, source_url

Output path: data/raw/bible_kjv/

Usage example:
    python parse_osis_kjv.py --input data/raw/bible_kjv/eng-kjv.osis.xml --output data/raw/bible_kjv/
"""


import xml.etree.ElementTree as ET  # For parsing XML files
import pandas as pd  # For handling data and saving to CSV
from pathlib import Path  # For handling file paths
from typing import Dict, List, Optional  # For type annotations
import argparse  # For command-line argument parsing
import logging  # For logging errors and warnings

def setup_logger(log_path: Path) -> logging.Logger:
    """
    Set up a logger to record errors and warnings.

    Args:
        log_path (Path): Path to the log file.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger("OSISParser")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.FileHandler(log_path, encoding="utf-8")
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


# === CONFIG ===
SOURCE_URL = "https://github.com/seven1m/open-bibles"  # Source URL for the data
DEFAULT_OSIS_PATH = Path("data/raw/bible_kjv/eng-kjv.osis.xml")  # Default input OSIS XML file path
DEFAULT_OUTPUT_DIR = Path("data/raw/bible_kjv")  # Default output directory for CSV files
DEFAULT_LOG_PATH = Path("logs/parse_osis_kjv.log")  # Default log file path


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

# Map OSIS book codes to human-readable book names
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

# === HELPERS ===

# Clean and normalize text by removing extra whitespace
def clean_text(text: Optional[str]) -> str:
    """
    Clean and normalize a string by removing extra whitespace.

    Args:
        text (Optional[str]): The text to clean.

    Returns:
        str: Cleaned text with normalized spaces.
    """
    return " ".join((text or "").split())


# Parse OSIS XML file and extract verses grouped by book
def parse_osis_verses(osis_path: Path, logger: logging.Logger) -> Dict[str, List[Dict[str, str]]]:
    """
    Parse an OSIS XML file and extract all verses grouped by book.

    Args:
        osis_path (Path): Path to the OSIS XML file.
        logger (logging.Logger): Logger for error and warning reporting.

    Returns:
        Dict[str, List[Dict[str, str]]]: Dictionary mapping book names to lists of verses.
    """
    print(f"📖 Parsing: {osis_path}")
    try:
        tree = ET.parse(osis_path)  # May raise ParseError or FileNotFoundError
        root = tree.getroot()
    except FileNotFoundError:
        logger.error(f"File not found: {osis_path}")
        print(f"❌ File not found: {osis_path}")
        return {}
    except ET.ParseError as e:
        logger.error(f"XML parsing error in {osis_path}: {e}")
        print(f"❌ XML parsing error in {osis_path}: {e}")
        return {}

    verses_by_book: Dict[str, List[Dict[str, str]]] = {}
    current_text, current_verse_id = "", None
    book = chapter = verse = ""
    capturing = False
    unknown_books = set()

    # Iterate through all elements in the XML tree
    for elem in root.iter():
        tag = elem.tag.split("}")[-1]  # Extract the tag name without namespace

        # Start of a verse with osisID
        if tag == "verse" and "osisID" in elem.attrib:
            current_verse_id = elem.attrib["osisID"]
            parts = current_verse_id.split(".")
            if len(parts) != 3:
                logger.warning(f"Invalid osisID format: {current_verse_id}")
                continue
            osis_book, chapter, verse = parts
            book = OSIS_TO_NAME.get(osis_book)
            if not book:
                unknown_books.add(osis_book)
                logger.warning(f"Unknown OSIS book code: {osis_book}")
                continue
            capturing = True
            current_text = clean_text(elem.tail)

        # End of a verse with eID
        elif tag == "verse" and "eID" in elem.attrib:
            if capturing and book:
                row = {
                    "book": book,
                    "chapter": chapter,
                    "verse": verse,
                    "subtitle": "",
                    "text": current_text.strip(),
                    "source_url": SOURCE_URL
                }
                verses_by_book.setdefault(book, []).append(row)
            capturing = False
            current_text = ""

        # Capture text within a verse
        elif capturing:
            current_text += " " + clean_text(elem.text)
            current_text += " " + clean_text(elem.tail)

    # Warn about unknown OSIS book codes
    if unknown_books:
        logger.warning(f"Unknown OSIS book codes encountered: {', '.join(sorted(unknown_books))}")
        print(f"⚠️ Unknown OSIS book codes encountered: {', '.join(sorted(unknown_books))}")

    return verses_by_book

def save_verses_to_csvs(verses_by_book: Dict[str, List[Dict[str, str]]], output_dir: Path, logger: logging.Logger) -> None:
    """
    Save verses grouped by book to individual CSV files.

    Args:
        verses_by_book (Dict[str, List[Dict[str, str]]]): Verses grouped by book.
        output_dir (Path): Directory to save the CSV files.
        logger (logging.Logger): Logger for error and warning reporting.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Saving to: {output_dir}")

    missing_books = []
    for i, book in enumerate(BOOK_ORDER, start=1):
        verses = verses_by_book.get(book)
        if not verses:
            missing_books.append(book)
            logger.warning(f"No verses found for book: {book}")
            continue
        df = pd.DataFrame(verses)
        filename = f"{i}_{book}.csv"
        try:
            df.to_csv(output_dir / filename, index=False)
            print(f"✅ {filename}: {len(df)} verses")
        except Exception as e:
            logger.error(f"Failed to save {filename}: {e}")
            print(f"❌ Failed to save {filename}: {e}")

    if missing_books:
        print(f"⚠️ The following books had no verses and were skipped: {', '.join(missing_books)}")


# === MAIN ===

def main(osis_path: Path, output_dir: Path, log_path: Path):
    """
    Main function to parse OSIS file and save verses to CSVs.

    Args:
        osis_path (Path): Path to the OSIS XML file.
        output_dir (Path): Directory to save CSVs.
        log_path (Path): Path to the log file.
    """
    logger = setup_logger(log_path)
    try:
        verses_by_book = parse_osis_verses(osis_path, logger)
        if not verses_by_book:
            print("❌ No verses parsed. Check the log for details.")
            return
        save_verses_to_csvs(verses_by_book, output_dir, logger)
        print("🚀 Done.")
    except Exception as e:
        logger.error(f"Unhandled exception in main: {e}")
        print(f"❌ Unhandled error: {e}")

# === CLI ===

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="📘 OSIS KJV Parser to CSV")
    parser.add_argument("--input", type=Path, default=DEFAULT_OSIS_PATH, help="Path to OSIS XML file")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory to save CSVs")
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG_PATH, help="Path to save the log file")
    args = parser.parse_args()

    main(args.input, args.output, args.log)
