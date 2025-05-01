"""
Parse the KJV Bible from an OSIS XML file with split <verse osisID> and <verse eID> tags.

Each CSV will contain the following columns:
book, chapter, verse, subtitle (empty), text, source_url

Output path: data/raw/bible_kjv/
"""

import xml.etree.ElementTree as ET  # For parsing XML files
import pandas as pd  # For handling data and saving to CSV
from pathlib import Path  # For handling file paths
from typing import Dict, List, Optional  # For type annotations
import argparse  # For command-line argument parsing

# === CONFIG ===
SOURCE_URL = "https://github.com/seven1m/open-bibles"  # Source URL for the data

# === PATHS ===
DEFAULT_OSIS_PATH = Path("data/raw/bible_kjv/eng-kjv.osis.xml")  # Default input OSIS XML file path
DEFAULT_OUTPUT_DIR = Path("data/raw/bible_kjv")  # Default output directory for CSV files

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
    return " ".join((text or "").split())

# Parse OSIS XML file and extract verses grouped by book
def parse_osis_verses(osis_path: Path) -> Dict[str, List[Dict[str, str]]]:
    print(f"📖 Parsing: {osis_path}")
    tree = ET.parse(osis_path)  # Parse the XML file
    root = tree.getroot()  # Get the root element of the XML

    verses_by_book: Dict[str, List[Dict[str, str]]] = {}  # Dictionary to store verses by book
    current_text, current_verse_id = "", None  # Initialize variables for verse text and ID
    book = chapter = verse = ""  # Initialize book, chapter, and verse
    capturing = False  # Flag to indicate if we are capturing verse text
    unknown_books = set()  # Set to track unknown OSIS book codes

    # Iterate through all elements in the XML tree
    for elem in root.iter():
        tag = elem.tag.split("}")[-1]  # Extract the tag name without namespace

        # Start of a verse with osisID
        if tag == "verse" and "osisID" in elem.attrib:
            current_verse_id = elem.attrib["osisID"]  # Get the osisID attribute
            parts = current_verse_id.split(".")  # Split osisID into parts
            if len(parts) != 3:  # Skip if the osisID format is invalid
                continue
            osis_book, chapter, verse = parts  # Extract book, chapter, and verse
            book = OSIS_TO_NAME.get(osis_book)  # Map OSIS book code to book name
            if not book:  # Skip if the book is unknown
                unknown_books.add(osis_book)
                continue
            capturing = True  # Start capturing verse text
            current_text = clean_text(elem.tail)  # Initialize verse text

        # End of a verse with eID
        elif tag == "verse" and "eID" in elem.attrib:
            if capturing and book:  # If capturing and book is valid
                row = {
                    "book": book,
                    "chapter": chapter,
                    "verse": verse,
                    "subtitle": "",
                    "text": current_text.strip(),
                    "source_url": SOURCE_URL
                }
                verses_by_book.setdefault(book, []).append(row)  # Add verse to the book
            capturing = False  # Stop capturing
            current_text = ""  # Reset verse text

        # Capture text within a verse
        elif capturing:
            current_text += " " + clean_text(elem.text)  # Add element text
            current_text += " " + clean_text(elem.tail)  # Add element tail text

    # Warn about unknown OSIS book codes
    if unknown_books:
        print(f"⚠️ Unknown OSIS book codes encountered: {', '.join(sorted(unknown_books))}")

    return verses_by_book

# Save verses grouped by book to individual CSV files
def save_verses_to_csvs(verses_by_book: Dict[str, List[Dict[str, str]]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)  # Create output directory if it doesn't exist
    print(f"📁 Saving to: {output_dir}")

    # Iterate through books in canonical order
    for i, book in enumerate(BOOK_ORDER, start=1):
        verses = verses_by_book.get(book)  # Get verses for the current book
        if not verses:  # Skip if no verses for the book
            continue
        df = pd.DataFrame(verses)  # Convert verses to a DataFrame
        filename = f"{i}_{book}.csv"  # Generate filename
        df.to_csv(output_dir / filename, index=False)  # Save DataFrame to CSV
        print(f"✅ {filename}: {len(df)} verses")  # Print success message

# === MAIN ===

# Main function to parse OSIS file and save verses to CSVs
def main(osis_path: Path, output_dir: Path):
    verses_by_book = parse_osis_verses(osis_path)  # Parse OSIS file
    save_verses_to_csvs(verses_by_book, output_dir)  # Save verses to CSVs
    print("🚀 Done.")  # Print completion message

# === CLI ===

# Command-line interface for the script
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="📘 OSIS KJV Parser to CSV")  # CLI description
    parser.add_argument("--input", type=Path, default=DEFAULT_OSIS_PATH, help="Path to OSIS XML file")  # Input file argument
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory to save CSVs")  # Output directory argument
    args = parser.parse_args()  # Parse command-line arguments

    main(args.input, args.output)  # Run the main function with parsed arguments