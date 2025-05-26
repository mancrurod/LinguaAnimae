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

Usage:
    python cleaning.py
"""


# =====================
# === IMPORTS =========
# =====================

import pandas as pd
from pathlib import Path
import re
from datetime import datetime
import logging


# ========================
# === CONSTANTS ==========
# ========================

# Define directories for raw data, processed data, and logs
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
LOG_DIR = Path("logs/cleaning_logs")

for folder in [RAW_DIR, LOG_DIR]:
    if not folder.exists():
        print(f"❌ Required directory does not exist: {folder}")

def setup_logger(
    log_path: Path,
    level: int = logging.INFO,
    console: bool = True,
    log_name: str = "cleaning_logger"
) -> logging.Logger:
    """
    Set up a logger that logs to both a file and optionally the console.

    Args:
        log_path (Path): Path to the log file.
        level (int): Logging level (e.g., logging.INFO, logging.ERROR).
        console (bool): Whether to log also to the console.
        log_name (str): Name of the logger instance.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(log_name)
    logger.setLevel(level)
    if not logger.handlers:
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setLevel(level)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s")
        fh.setFormatter(file_formatter)
        logger.addHandler(fh)

        if console:
            ch = logging.StreamHandler()
            ch.setLevel(level)
            console_formatter = logging.Formatter(
                "%(levelname)s: %(message)s")
            ch.setFormatter(console_formatter)
            logger.addHandler(ch)
    return logger


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
        return ""
    text = str(text)

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
        return ""
    text = str(text)
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

    Returns True if:
    - 'book' is a non-empty string
    - 'chapter' and 'verse' can be converted to positive integers
    """
    book = row.get("book", "")
    chapter = row.get("chapter", None)
    verse = row.get("verse", None)

    if not isinstance(book, str) or not book.strip():
        return False
    try:
        chapter_num = int(float(chapter))
        verse_num = int(float(verse))
        return chapter_num > 0 and verse_num > 0
    except (ValueError, TypeError):
        return False

def generate_id(row: pd.Series) -> str:
    """
    Create a unique verse identifier from the 'book', 'chapter', and 'verse'.

    Args:
        row (pd.Series): A row from the DataFrame.

    Returns:
        str: Unique ID string, e.g., "Genesis_1_1". If values are missing or invalid,
             returns 'INVALID_ID'.
    """
    try:
        book = str(row.get('book', '')).strip().replace(' ', '_').title()
        chapter = int(float(row.get('chapter', 0)))
        verse = int(float(row.get('verse', 0)))
        if not book or chapter <= 0 or verse <= 0:
            return 'INVALID_ID'
        return f"{book}_{chapter}_{verse}"
    except Exception as e:
        # Optional: log the error somewhere if running in a wider pipeline
        return 'INVALID_ID'


# ========================
# === MAIN PROCESS =======
# ========================

def clean_and_prepare_csvs(logger: logging.Logger) -> None:
    """
    Process all CSV files under RAW_DIR, clean the data, and save outputs to PROCESSED_DIR.
    """
    csv_files = list(RAW_DIR.rglob("*.csv"))
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    if not csv_files:
        logger.error("No CSV files found under 'data/raw/'.")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"cleaning_log_{timestamp}.txt"

    processed_count = 0
    failed_count = 0

    with open(log_file, "w", encoding="utf-8") as log:
        log.write(f"Cleaning Session: {timestamp}\n")
        log.write(f"Total CSV files found: {len(csv_files)}\n\n")

        logger.info(f"Found {len(csv_files)} CSV files to process.")

        for file_path in csv_files:
            try:
                relative_file = file_path.relative_to(RAW_DIR)
                logger.info(f"Processing: {relative_file}")

                try:
                    df = pd.read_csv(file_path)
                except Exception as e:
                    warning = f"Error reading file {relative_file}: {e}\n"
                    log.write(f"❌ {warning}")
                    logger.error(warning)
                    failed_count += 1
                    continue

                log.write(f"=== File: {relative_file} ===\n")

                required_cols = {"book", "chapter", "verse", "text"}
                if not required_cols.issubset(df.columns):
                    missing = required_cols - set(df.columns)
                    warning = f"Skipped (missing required columns: {missing})\n"
                    log.write(f"⚠️ {warning}")
                    logger.warning(warning)
                    failed_count += 1
                    continue

                original_rows = len(df)
                df["text"] = df["text"].apply(clean_text).apply(normalize_unicode)

                for col in ["chapter", "verse"]:
                    try:
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                    except Exception as e:
                        log.write(f"⚠️ Could not convert {col} to numeric: {e}\n")
                        logger.warning(f"Could not convert {col} to numeric: {e}")

                valid_mask = df.apply(validate_row, axis=1)
                removed_rows = (~valid_mask).sum()
                df = df[valid_mask]
                valid_rows = len(df)

                if valid_rows == 0:
                    log.write("⚠️ No valid rows after cleaning. Skipped file.\n")
                    logger.warning(f"No valid rows after cleaning for file {relative_file}. Skipped file.")
                    failed_count += 1
                    continue

                df["theme"] = ""
                df["emotion"] = ""
                df["verse_id"] = df.apply(generate_id, axis=1)
                df.insert(0, "id", range(1, len(df) + 1))

                text_idx = df.columns.get_loc("text")
                cols = list(df.columns)
                cols.remove("verse_id")
                cols.insert(text_idx + 1, "verse_id")
                df = df[cols]

                relative_path = file_path.relative_to(RAW_DIR)
                new_name = relative_path.stem + "_cleaned.csv"
                output_path = PROCESSED_DIR / relative_path.parent / new_name
                output_path.parent.mkdir(parents=True, exist_ok=True)

                try:
                    df.to_csv(
                        output_path,
                        index=False,
                        encoding="utf-8",
                        lineterminator="\n",
                        quoting=1
                    )
                except Exception as e:
                    error_message = f"Error saving {output_path}: {e}"
                    log.write(f"❌ {error_message}\n")
                    logger.error(error_message)
                    failed_count += 1
                    continue

                log.write(f"Original rows: {original_rows}\n")
                log.write(f"Rows after cleaning: {valid_rows}\n")
                log.write(f"Rows removed: {removed_rows}\n")
                log.write(f"Saved to: {output_path.relative_to(PROCESSED_DIR)}\n\n")

                logger.info(f"Saved cleaned file: {output_path.relative_to(PROCESSED_DIR)}")
                processed_count += 1

            except Exception as e:
                error_message = f"Error processing {file_path}: {e}"
                log.write(f"❌ {error_message}\n")
                logger.error(error_message)
                failed_count += 1

        log.write(f"=== Session Summary ===\n")
        log.write(f"Successfully processed: {processed_count}\n")
        log.write(f"Files with errors: {failed_count}\n")
        logger.info(f"=== Cleaning summary: {processed_count} processed, {failed_count} with errors ===")

    logger.info(f"Cleaning log saved at: {log_file}")


# ========================
# === ENTRY POINT ========
# ========================

if __name__ == "__main__":
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"cleaning_{timestamp}.log"
    logger = setup_logger(log_path, level=logging.INFO, console=True)
    clean_and_prepare_csvs(logger)

