"""
Selects N random unique verses from all CSVs in data/processed/bible_kjv/
and saves them as emotion_verses_to_label.csv for annotation, including verse_id.

Columns: id, verse_id, verse

- Skips verses already annotated if present in emotion_verses_labeled_combined.csv.
- Logs all steps, errors and key statistics to logs/verse_selection_logs/select_verses.log.

Usage:
python select_verses_for_labeling.py --n-samples 1000 --output-file my_sample.csv
"""

import pandas as pd
from pathlib import Path
import logging
import argparse

# === Logger setup ===
def setup_logger(
    log_path: Path,
    level: int = logging.INFO,
    log_name: str = "select_verses_logger"
) -> logging.Logger:
    """
    Set up a logger for the verse selection script.

    Args:
        log_path (Path): Path to the log file.
        level (int): Logging level.
        log_name (str): Logger name.

    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger(log_name)
    logger.setLevel(level)
    if not logger.handlers:
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(fh)
    return logger

# === Paths and constants ===
BASE = Path(__file__).parent.parent.parent
INPUT_DIR = BASE / "data" / "processed" / "bible_kjv"
OUTPUT_FILE = BASE / "data" / "evaluation" / "verses_to_label" / "emotion_verses_to_label_6.csv" # Change as needed
LOG_DIR = BASE / "logs" / "verse_selection_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
logger = setup_logger(LOG_DIR / "select_verses.log")

N_SAMPLES = 6000

def main(
    input_dir=INPUT_DIR,
    output_file=OUTPUT_FILE,
    n_samples=N_SAMPLES,
    existing_labels_path=BASE / "data" / "evaluation" / "emotion_verses_labeled_combined.csv",
    logger=logger
):
    logger.info("🔎 Searching for CSV files...")
    all_csvs = list(INPUT_DIR.glob("*.csv"))
    logger.info(f"Found {len(all_csvs)} files in {INPUT_DIR}")

    records = []
    for csv in all_csvs:
        try:
            df = pd.read_csv(csv)
        except Exception as e:
            logger.error(f"Error reading file {csv.name}: {e}")
            continue
        if 'verse_id' in df.columns and 'text' in df.columns:
            temp = df[['verse_id', 'text']].dropna()
            temp = temp.rename(columns={'text': 'verse'})
            records.append(temp)
            logger.info(f"Loaded {len(temp)} verses from {csv.name}")
        else:
            logger.warning(f"Skipping {csv.name}: missing 'verse_id' or 'text' column.")

    if not records:
        logger.error("❌ No suitable data found! Exiting.")
        return

    df_all = pd.concat(records, ignore_index=True)
    logger.info(f"Total verses loaded (all files): {len(df_all)}")

    # Remove duplicates by verse_id (safer than by text only)
    before_dedup = len(df_all)
    df_all = df_all.drop_duplicates(subset=['verse_id'])
    logger.info(f"Removed {before_dedup - len(df_all)} duplicate verses by verse_id. Unique verses now: {len(df_all)}")

    # (Opcional) Exclude already labeled verses if file exists
    EXISTING_LABELS = BASE / "data" / "evaluation" / "emotion_verses_labeled_combined.csv"
    if EXISTING_LABELS.exists():
        try:
            df_labeled = pd.read_csv(EXISTING_LABELS)
            labeled_ids = set(df_labeled['verse_id'])
            before_filter = len(df_all)
            df_all = df_all[~df_all['verse_id'].isin(labeled_ids)]
            logger.info(f"Filtered out {before_filter - len(df_all)} verses already labeled. {len(df_all)} left after filtering.")
        except Exception as e:
            logger.error(f"Error reading existing labels file {EXISTING_LABELS.name}: {e}")
    
    n = min(N_SAMPLES, len(df_all))
    if n == 0:
        logger.error("❌ No verses left to sample after filtering! Exiting.")
        return
    if n < N_SAMPLES:
        logger.warning(f"Only {n} verses available to sample, less than requested {N_SAMPLES}.")

    try:
        df_sample = df_all.sample(n=n, random_state=16).reset_index(drop=True)
        logger.info(f"Sampled {n} verses.")
    except Exception as e:
        logger.error(f"Error during sampling: {e}")
        return

    df_sample.insert(0, 'id', range(n))
    df_sample = df_sample[['id', 'verse_id', 'verse']]

    # Ensure output directory exists
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        df_sample.to_csv(OUTPUT_FILE, index=False)
        logger.info(f"✅ Sample of {n} verses saved to {OUTPUT_FILE.absolute()}")
    except Exception as e:
        logger.error(f"❌ Error saving output file {OUTPUT_FILE}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Select random unique Bible verses for annotation.")
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR,
                        help="Directory with processed verse CSVs (default: %(default)s)")
    parser.add_argument("--output-file", type=Path, default=OUTPUT_FILE,
                        help="Output file for the sample (default: %(default)s)")
    parser.add_argument("--n-samples", type=int, default=N_SAMPLES,
                        help="Number of unique verses to sample (default: %(default)s)")
    parser.add_argument("--existing-labels", type=Path,
                        default=BASE / "data" / "evaluation" / "emotion_verses_labeled_combined.csv",
                        help="CSV file with already labeled verse_ids to exclude (default: %(default)s)")
    parser.add_argument("--log-file", type=Path,
                        default=LOG_DIR / "select_verses.log",
                        help="Path to the log file (default: %(default)s)")

    args = parser.parse_args()

    # If you change paths in CLI, create logger with new path
    logger = setup_logger(args.log_file)

    main(
        input_dir=args.input_dir,
        output_file=args.output_file,
        n_samples=args.n_samples,
        existing_labels_path=args.existing_labels,
        logger=logger
    )

