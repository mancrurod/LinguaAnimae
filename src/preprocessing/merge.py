"""
Merge all cleaned CSVs inside a specified subfolder of 'data/processed/' into a single CSV file.

The merged file will:
- Remove the original 'id' columns temporarily to avoid duplication.
- Preserve the 'verse_id' field.
- Generate a new sequential numeric 'id' as the first column.
- Merge CSVs following the correct biblical order based on file naming (e.g., 1_genesis_cleaned.csv, 2_exodo_cleaned.csv).
- Save the result inside the same folder, named after the subfolder (e.g., bible_rv60.csv).

Usage:
    python merge.py
    # or
    # Adapt target_folder in the script for a different subfolder
"""


# =====================
# === IMPORTS =========
# =====================

import pandas as pd
from pathlib import Path
import argparse
import logging
from datetime import datetime

# ========================
# === CONSTANTS ==========
# ========================

PROCESSED_DIR = Path("data/processed")

def setup_logger(
    log_path: Path,
    level: int = logging.INFO,
    console: bool = True,
    log_name: str = "merge_logger"
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

    # Avoid adding handlers multiple times in interactive sessions
    if not logger.handlers:
        # File handler
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setLevel(level)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s")
        fh.setFormatter(file_formatter)
        logger.addHandler(fh)

        # Console handler
        if console:
            ch = logging.StreamHandler()
            ch.setLevel(level)
            console_formatter = logging.Formatter(
                "%(levelname)s: %(message)s")
            ch.setFormatter(console_formatter)
            logger.addHandler(ch)

    return logger

# =============================
# === MERGING FUNCTION ========
# =============================

def combine_cleaned_csvs(processed_subdir: Path, logger: logging.Logger) -> None:
    """
    Combine all cleaned CSV files inside a given subfolder of 'data/processed/' into a single CSV.
    The final file will have a new global 'id' column and will preserve the 'verse_id' field.
    Original 'id' columns from individual files will be removed during merging.

    Args:
        processed_subdir (Path): Path to the subfolder inside 'data/processed/'.
        logger (logging.Logger): Logger instance for logging messages.

    Returns:
        None
    """
    if not processed_subdir.exists():
        logger.error(f"The directory {processed_subdir} does not exist.")
        return

    all_csvs = list(processed_subdir.glob("*_cleaned.csv"))
    if not all_csvs:
        logger.error(f"No cleaned CSV files found in {processed_subdir}")
        return

    # Sort files based on the numeric prefix in the filename
    def extract_prefix(file: Path) -> int:
        try:
            return int(file.stem.split("_")[0])
        except (IndexError, ValueError):
            logger.warning(f"Malformed filename (ignored for order): {file.name}")
            return float('inf')  # Put malformed filenames at the end

    all_csvs = sorted(all_csvs, key=extract_prefix)

    logger.info(f"Found {len(all_csvs)} cleaned CSV files in {processed_subdir}")

    df_list = []
    error_files = []

    for csv_file in all_csvs:
        try:
            df = pd.read_csv(csv_file)
            if "id" in df.columns:
                df = df.drop(columns=["id"])
            if df.empty:
                logger.warning(f"File {csv_file.name} is empty and will be skipped.")
                continue
            df_list.append(df)
        except Exception as e:
            logger.error(f"Error reading {csv_file}: {e}")
            error_files.append(csv_file.name)

    if not df_list:
        logger.warning("No valid DataFrames to merge.")
        return

    # Concatenate all DataFrames
    combined_df = pd.concat(df_list, ignore_index=True)

    if combined_df.empty:
        logger.warning("Combined DataFrame is empty after merging. No file will be saved.")
        return

    # Insert new sequential 'id' as the first column
    combined_df.insert(0, "id", range(1, len(combined_df) + 1))

    # Define the output file name based on the folder name
    folder_name = processed_subdir.name
    output_path = processed_subdir / f"{folder_name}.csv"

    try:
        combined_df.to_csv(output_path, index=False, encoding="utf-8", lineterminator="\n")
        logger.info(f"Combined CSV saved to: {output_path.relative_to(PROCESSED_DIR)}")
    except Exception as e:
        logger.error(f"Failed to save combined CSV: {e}")

    if error_files:
        logger.warning(f"The following files could not be read and were skipped: {', '.join(error_files)}")


# ======================
# === ENTRY POINT ======
# ======================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge all cleaned CSVs in a processed subfolder into one CSV.")
    parser.add_argument(
        "--folder",
        type=str,
        default="bible_rv60",
        help="Subfolder inside data/processed/ to merge (default: bible_rv60)"
    )
    parser.add_argument(
        "--logfile",
        type=str,
        default=None,
        help="Path to log file (default: logs/merge_logs/merge_<timestamp>.log)"
    )
    args = parser.parse_args()

    target_folder = PROCESSED_DIR / args.folder

    # === Logging setup ===
    LOG_DIR = Path("logs/merge_logs")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.logfile:
        log_path = Path(args.logfile)
    else:
        log_path = LOG_DIR / f"merge_{args.folder}_{timestamp}.log"
    logger = setup_logger(log_path, level=logging.INFO, console=True)

    # Run the merging function
    combine_cleaned_csvs(target_folder, logger)