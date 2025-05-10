"""
Clean all CSV files in the 'data/raw' directory and prepare them for annotation.

Each file undergoes the following steps:
- Trim and normalize whitespace in the 'text' column.
- Normalize unicode punctuation to ASCII.
- Remove invalid rows based on structural rules.
- Add 'theme' and 'emotion' columns (empty).
- Generate a unique verse identifier ('verse_id').
- Save the cleaned data under 'data/processed' in a mirrored folder structure.
- Log changes made during the process in 'logs/cleaning_logs'.

This module is intended to be executed as a standalone script.
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

# Define directories for raw data, processed data, and logs
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
LOG_DIR = Path("logs/cleaning_logs")

# =============================
# === CLEANING UTILITIES ======
# =============================

def clean_text(text: str) -> str:
    """
    Clean verse text by trimming each line and collapsing internal whitespace,
    but preserving poetic line breaks (\n) for formatted verses.

    Args:
        text (str): Raw verse text.

    Returns:
        str: Cleaned text with preserved line breaks.
    """
    if pd.isna(text):
        return text

    # Normalize each line individually
    lines = text.splitlines()
    cleaned_lines = [re.sub(r'\s+', ' ', line.strip()) for line in lines]
    return "\n".join(cleaned_lines).strip()


def normalize_unicode(text: str) -> str:
    """
    Normalize unicode punctuation characters to standard ASCII equivalents.

    Args:
        text (str): Text potentially containing unicode punctuation.

    Returns:
        str: Text with standardized punctuation.
    """
    if pd.isna(text):  # Check if the text is NaN
        return text
    return (
        text.replace("“", '"')  # Replace left double quotes with standard double quotes
            .replace("”", '"')  # Replace right double quotes with standard double quotes
            .replace("’", "'")  # Replace right single quotes with standard single quotes
            .replace("‘", "'")  # Replace left single quotes with standard single quotes
            .replace("–", "-")  # Replace en dash with hyphen
            .replace("—", "-")  # Replace em dash with hyphen
    )


def validate_row(row: pd.Series) -> bool:
    """
    Check whether a DataFrame row contains valid 'book', 'chapter', and 'verse' fields.

    Args:
        row (pd.Series): A row from the DataFrame.

    Returns:
        bool: True if the row passes all validation rules, False otherwise.
    """
    return (
        isinstance(row.get("book"), str) and  # Ensure 'book' is a string
        isinstance(row.get("chapter"), (int, float)) and row["chapter"] > 0 and  # Ensure 'chapter' is positive
        isinstance(row.get("verse"), (int, float)) and row["verse"] > 0  # Ensure 'verse' is positive
    )


def generate_id(row: pd.Series) -> str:
    """
    Create a unique verse identifier from the 'book', 'chapter', and 'verse'.

    Args:
        row (pd.Series): A row from the DataFrame.

    Returns:
        str: Unique ID string, e.g., "Genesis_1_1".
    """
    return f"{row['book']}_{int(row['chapter'])}_{int(row['verse'])}"  # Combine fields into a unique identifier

# ========================
# === MAIN PROCESS =======
# ========================

def clean_and_prepare_csvs() -> None:
    """
    Process all CSV files under RAW_DIR, clean the data, and save outputs to PROCESSED_DIR.

    Performs:
    - Validation and sanitization of text.
    - Row-level structural validation.
    - Logging of all transformations per file.
    - Export of cleaned CSVs and session log.
    """
    # Get a list of all CSV files in the raw data directory
    csv_files = list(RAW_DIR.rglob("*.csv"))
    LOG_DIR.mkdir(parents=True, exist_ok=True)  # Ensure the log directory exists

    if not csv_files:  # Check if no CSV files are found
        print("❌ No CSV files found under 'data/raw/'.")
        return

    # Generate a timestamp for the log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"cleaning_log_{timestamp}.txt"

    # Open the log file for writing
    with open(log_file, "w", encoding="utf-8") as log:
        log.write(f"Cleaning Session: {timestamp}\n")
        log.write(f"Total CSV files found: {len(csv_files)}\n\n")

        print(f"📁 Found {len(csv_files)} CSV files to process.\n")

        # Process each CSV file
        for file_path in csv_files:
            try:
                # Get the relative path of the file for logging
                relative_file = file_path.relative_to(RAW_DIR)
                print(f"🔍 Processing: {relative_file}")

                # Read the CSV file into a DataFrame
                df = pd.read_csv(file_path)
                log.write(f"=== File: {relative_file} ===\n")

                # Check if required columns are present
                required_cols = {"book", "chapter", "verse", "text"}
                if not required_cols.issubset(df.columns):
                    warning = f"⚠️ Skipped (missing required columns)\n\n"
                    log.write(warning)
                    print(warning)
                    continue

                original_rows = len(df)  # Record the original number of rows

                # Clean and normalize the 'text' column
                df["text"] = df["text"].apply(clean_text).apply(normalize_unicode)
                # Filter rows based on validation rules
                df = df[df.apply(validate_row, axis=1)]
                valid_rows = len(df)  # Record the number of valid rows

                # Add empty 'theme' and 'emotion' columns
                df["theme"] = ""
                df["emotion"] = ""
                # Generate unique verse IDs
                df["verse_id"] = df.apply(generate_id, axis=1)
                # Add an 'id' column with sequential numbers
                df.insert(0, "id", range(1, len(df) + 1))

                # Reorder columns to place 'verse_id' after 'text'
                text_idx = df.columns.get_loc("text")
                cols = list(df.columns)
                cols.remove("verse_id")
                cols.insert(text_idx + 1, "verse_id")
                df = df[cols]

                # Determine the output path for the cleaned file
                relative_path = file_path.relative_to(RAW_DIR)
                new_name = relative_path.stem + "_cleaned.csv"
                output_path = PROCESSED_DIR / relative_path.parent / new_name
                output_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure the output directory exists

                # Save the cleaned DataFrame to a new CSV file
                df.to_csv(
                    output_path,
                    index=False,
                    encoding="utf-8",
                    lineterminator="\n",
                    quoting=1  # csv.QUOTE_ALL — forces quoting of all fields
                )

                # Log the results of the cleaning process
                log.write(f"Original rows: {original_rows}\n")
                log.write(f"Rows after cleaning: {valid_rows}\n")
                log.write(f"Rows removed: {original_rows - valid_rows}\n")
                log.write(f"Saved to: {output_path.relative_to(PROCESSED_DIR)}\n\n")

                print(f"✅ Saved cleaned file: {output_path.relative_to(PROCESSED_DIR)}\n")

            except Exception as e:
                # Log and print any errors encountered during processing
                error_message = f"❌ Error processing {file_path}: {e}\n"
                log.write(error_message)
                print(error_message)

    print(f"📝 Cleaning log saved at: {log_file}")

# ========================
# === ENTRY POINT ========
# ========================

if __name__ == "__main__":
    clean_and_prepare_csvs()  # Execute the main cleaning process
