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

RAW_DIR = Path("data/raw")  # Directory containing raw CSV files
PROCESSED_DIR = Path("data/processed")  # Directory to save processed CSV files
LOG_DIR = Path("logs/cleaning_logs")  # Directory to save cleaning logs

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
    if pd.isna(text):  # Handle NaN values
        return text
    return re.sub(r'\s+', ' ', text.strip())  # Replace multiple spaces with a single space


def normalize_unicode(text: str) -> str:
    """
    Normalize unicode punctuation characters to standard ASCII.

    Args:
        text (str): Text containing potentially non-standard unicode punctuation.

    Returns:
        str: Normalized text.
    """
    if pd.isna(text):  # Handle NaN values
        return text
    return (
        text.replace("“", '"')  # Replace left double quotes
            .replace("”", '"')  # Replace right double quotes
            .replace("’", "'")  # Replace right single quotes
            .replace("‘", "'")  # Replace left single quotes
            .replace("–", "-")  # Replace en dash
            .replace("—", "-")  # Replace em dash
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
        isinstance(row.get("book"), str) and  # 'book' must be a string
        isinstance(row.get("chapter"), (int, float)) and row["chapter"] > 0 and  # 'chapter' must be a positive number
        isinstance(row.get("verse"), (int, float)) and row["verse"] > 0  # 'verse' must be a positive number
    )


def generate_id(row: pd.Series) -> str:
    """
    Generate a unique ID string from book, chapter, and verse.

    Args:
        row (pd.Series): A row of the DataFrame.

    Returns:
        str: Unique ID for the verse.
    """
    return f"{row['book']}_{int(row['chapter'])}_{int(row['verse'])}"  # Combine book, chapter, and verse into a unique ID

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
    csv_files = list(RAW_DIR.rglob("*.csv"))  # Find all CSV files recursively in RAW_DIR
    LOG_DIR.mkdir(parents=True, exist_ok=True)  # Ensure the log directory exists

    if not csv_files:  # If no CSV files are found, exit early
        print("❌ No CSV files found under 'data/raw/'.")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # Generate a timestamp for the log file
    log_file = LOG_DIR / f"cleaning_log_{timestamp}.txt"  # Define the log file path

    with open(log_file, "w", encoding="utf-8") as log:
        log.write(f"Cleaning Session: {timestamp}\n")
        log.write(f"Total CSV files found: {len(csv_files)}\n\n")

        print(f"📁 Found {len(csv_files)} CSV files to process.\n")

        for file_path in csv_files:
            try:
                relative_file = file_path.relative_to(RAW_DIR)  # Get the relative path of the file
                print(f"🔍 Processing: {relative_file}")

                df = pd.read_csv(file_path)  # Read the CSV file into a DataFrame
                log.write(f"=== File: {relative_file} ===\n")

                required_cols = {"book", "chapter", "verse", "text"}  # Required columns for processing
                if not required_cols.issubset(df.columns):  # Skip files missing required columns
                    warning = f"⚠️ Skipped (missing required columns)\n\n"
                    log.write(warning)
                    print(warning)
                    continue

                original_rows = len(df)  # Record the original number of rows

                # Clean and normalize text
                df["text"] = df["text"].apply(clean_text).apply(normalize_unicode)

                # Drop invalid rows
                df = df[df.apply(validate_row, axis=1)]
                valid_rows = len(df)  # Record the number of valid rows

                # Add empty columns for 'theme' and 'emotion'
                df["theme"] = ""
                df["emotion"] = ""

                # Generate unique IDs for each verse
                df["verse_id"] = df.apply(generate_id, axis=1)
                df.insert(0, "id", range(1, len(df) + 1))  # Add a sequential ID column

                # Reorganize: 'verse_id' after 'text'
                text_idx = df.columns.get_loc("text")  # Get the index of the 'text' column
                cols = list(df.columns)
                cols.remove("verse_id")
                cols.insert(text_idx + 1, "verse_id")  # Insert 'verse_id' after 'text'
                df = df[cols]

                # Saving .csv
                relative_path = file_path.relative_to(RAW_DIR)  # Get the relative path for saving
                new_name = relative_path.stem + "_cleaned.csv"  # Append '_cleaned' to the file name
                output_path = PROCESSED_DIR / relative_path.parent / new_name  # Define the output path
                output_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure the output directory exists

                df.to_csv(output_path, index=False)  # Save the cleaned DataFrame to a new CSV file

                # === Logging ===
                log.write(f"Original rows: {original_rows}\n")
                log.write(f"Rows after cleaning: {valid_rows}\n")
                log.write(f"Rows removed: {original_rows - valid_rows}\n")
                log.write(f"Saved to: {output_path.relative_to(PROCESSED_DIR)}\n\n")

                print(f"✅ Saved cleaned file: {output_path.relative_to(PROCESSED_DIR)}\n")

            except Exception as e:  # Handle any errors during processing
                error_message = f"❌ Error processing {file_path}: {e}\n"
                log.write(error_message)
                print(error_message)

    print(f"📝 Cleaning log saved at: {log_file}")  # Notify the user of the log file location

# ========================
# === ENTRY POINT ========
# ========================

if __name__ == "__main__":
    clean_and_prepare_csvs()  # Run the main cleaning process
