"""
Clean all CSVs in the 'data/raw' folder and prepare them for annotation.

Steps per file:
- Strip leading/trailing spaces from 'text'.
- Replace multiple spaces with a single space.
- Normalize unicode characters in text.
- Remove invalid rows based on structure.
- Add empty columns for 'theme' and 'emotion'.
- Add unique ID for each verse.
- Save cleaned file in parallel structure under 'data/processed'.
- Record a log of changes for each processed file.

Args:
    None

Returns:
    None
"""

# =====================
# === IMPORTS =========
# =====================

import pandas as pd
from pathlib import Path
import re
from datetime import datetime

# ========================
# === CONSTANTS ==========
# ========================

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
LOG_DIR = Path("logs/cleaning_logs")

# =============================
# === CLEANING UTILITIES ======
# =============================

def clean_text(text: str) -> str:
    """
    Clean verse text by trimming and normalizing whitespace.

    Args:
        text (str): The original verse text.

    Returns:
        str: Cleaned text.
    """
    if pd.isna(text):
        return text
    return re.sub(r'\s+', ' ', text.strip())


def normalize_unicode(text: str) -> str:
    """
    Normalize unicode punctuation characters to standard ASCII.

    Args:
        text (str): Text containing potentially non-standard unicode punctuation.

    Returns:
        str: Normalized text.
    """
    if pd.isna(text):
        return text
    return (
        text.replace("“", '"')
            .replace("”", '"')
            .replace("’", "'")
            .replace("‘", "'")
            .replace("–", "-")
            .replace("—", "-")
    )


def validate_row(row: pd.Series) -> bool:
    """
    Validate that required fields exist and contain appropriate types/values.

    Args:
        row (pd.Series): A row of the DataFrame.

    Returns:
        bool: True if the row is valid, False otherwise.
    """
    return (
        isinstance(row.get("book"), str) and
        isinstance(row.get("chapter"), (int, float)) and row["chapter"] > 0 and
        isinstance(row.get("verse"), (int, float)) and row["verse"] > 0
    )


def generate_id(row: pd.Series) -> str:
    """
    Generate a unique ID string from book, chapter, and verse.

    Args:
        row (pd.Series): A row of the DataFrame.

    Returns:
        str: Unique ID for the verse.
    """
    return f"{row['book']}_{int(row['chapter'])}_{int(row['verse'])}"

# ========================
# === MAIN PROCESS =======
# ========================

def clean_and_prepare_csvs() -> None:
    """
    Traverse all CSVs under RAW_DIR and generate cleaned versions in PROCESSED_DIR.
    Generates a log of actions performed.

    Args:
        None

    Returns:
        None
    """
    csv_files = list(RAW_DIR.rglob("*.csv"))
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    if not csv_files:
        print("❌ No CSV files found under 'data/raw/'.")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"cleaning_log_{timestamp}.txt"

    with open(log_file, "w", encoding="utf-8") as log:
        log.write(f"Cleaning Session: {timestamp}\n")
        log.write(f"Total CSV files found: {len(csv_files)}\n\n")

        print(f"📁 Found {len(csv_files)} CSV files to process.\n")

        for file_path in csv_files:
            try:
                relative_file = file_path.relative_to(RAW_DIR)
                print(f"🔍 Processing: {relative_file}")

                df = pd.read_csv(file_path)
                log.write(f"=== File: {relative_file} ===\n")

                required_cols = {"book", "chapter", "verse", "text"}
                if not required_cols.issubset(df.columns):
                    warning = f"⚠️ Skipped (missing required columns)\n\n"
                    log.write(warning)
                    print(warning)
                    continue

                original_rows = len(df)

                # Clean and normalize text
                df["text"] = df["text"].apply(clean_text).apply(normalize_unicode)

                # Drop invalid rows
                df = df[df.apply(validate_row, axis=1)]
                valid_rows = len(df)

                # Add empty columns for 'theme' and 'emotion'
                df["theme"] = ""
                df["emotion"] = ""

                df["verse_id"] = df.apply(generate_id, axis=1)
                df.insert(0, "id", range(1, len(df) + 1))

                # Reorganize: 'verse_id' after 'text'
                text_idx = df.columns.get_loc("text")
                cols = list(df.columns)
                cols.remove("verse_id")
                cols.insert(text_idx + 1, "verse_id")
                df = df[cols]

                # Saving .csv
                relative_path = file_path.relative_to(RAW_DIR)
                new_name = relative_path.stem + "_cleaned.csv"
                output_path = PROCESSED_DIR / relative_path.parent / new_name
                output_path.parent.mkdir(parents=True, exist_ok=True)

                df.to_csv(output_path, index=False)

                # === Logging ===
                log.write(f"Original rows: {original_rows}\n")
                log.write(f"Rows after cleaning: {valid_rows}\n")
                log.write(f"Rows removed: {original_rows - valid_rows}\n")
                log.write(f"Saved to: {output_path.relative_to(PROCESSED_DIR)}\n\n")

                print(f"✅ Saved cleaned file: {output_path.relative_to(PROCESSED_DIR)}\n")

            except Exception as e:
                error_message = f"❌ Error processing {file_path}: {e}\n"
                log.write(error_message)
                print(error_message)

    print(f"📝 Cleaning log saved at: {log_file}")

# ========================
# === ENTRY POINT ========
# ========================

if __name__ == "__main__":
    clean_and_prepare_csvs()
