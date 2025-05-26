"""
Emotion labeling script for Bible corpora.

- Labels each verse in one or more CSV files with an emotion using a HuggingFace model.
- Works in batch for efficiency.
- Outputs results and summary logs for analysis.
- Can be used via CLI for flexible integration in ETL pipelines.

Usage:
    python emotion_theme_labeling.py --model "j-hartmann/emotion-english-distilroberta-base" --bible bible_kjv
"""

import pandas as pd
from transformers import pipeline
from pathlib import Path
from typing import List
from datetime import datetime
import argparse
import torch
import logging

# ========================
# === CONFIG DEFAULTS ====
# ========================

DEFAULT_MODEL = "j-hartmann/emotion-english-distilroberta-base"
DEFAULT_BIBLE = "bible_kjv"
DEFAULT_LOG_DIR = Path("logs/emotion_labeling_logs")
BATCH_SIZE = 32

# ========================
# === LOGGER SETUP =======
# ========================

def setup_logger(
    log_path: Path,
    level: int = logging.INFO,
    console: bool = True,
    log_name: str = "emotion_labeling_logger"
) -> logging.Logger:
    """
    Set up a logger that logs to both a file and optionally the console.
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

# ========================
# === MODEL LOADER =======
# ========================

def load_classifier(model_name: str, device: int = 0, logger: logging.Logger = None):
    device_id = device if torch.cuda.is_available() else -1
    msg = f"Loading model: {model_name} (device: {'cuda' if device_id >= 0 else 'cpu'})"
    if logger:
        logger.info(msg)
    else:
        print(f"📦 {msg}")
    try:
        return pipeline("text-classification", model=model_name, device=device_id, top_k=None)
    except Exception as e:
        if logger:
            logger.error(f"Error loading model: {e}")
        else:
            print(f"❌ Error loading model: {e}")
        raise

# ========================
# === EMOTION CLASSIFY ===
# ========================

def classify_batch(texts: List[str], classifier, logger: logging.Logger) -> List[str]:
    try:
        results = classifier(texts)
        return [max(r, key=lambda x: x["score"])["label"] for r in results]
    except Exception as e:
        logger.error(f"Error during batch classification: {e}")
        # Return "error" for all to allow tracking
        return ["error"] * len(texts)

def process_file(
    file: Path,
    classifier,
    output_dir: Path,
    summary_lines: List[str],
    summarize: bool,
    logger: logging.Logger
) -> None:
    logger.info(f"Processing: {file.name}")
    try:
        df = pd.read_csv(file)
    except Exception as e:
        logger.error(f"Error reading file {file.name}: {e}")
        return

    if "text" not in df.columns:
        logger.warning(f"Skipped (missing 'text' column): {file.name}")
        return

    texts = df["text"].astype(str).tolist()
    emotions = []

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        emotions.extend(classify_batch(batch, classifier, logger))

    if len(emotions) != len(df):
        logger.error(f"Mismatch between number of emotions and rows in file {file.name}. Skipping save.")
        return

    df["emotion"] = emotions
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / file.name.replace("_cleaned.csv", "_emotion.csv")
    try:
        df.to_csv(out_file, index=False)
        logger.info(f"Saved: {out_file.name}")
    except Exception as e:
        logger.error(f"Error saving file {out_file.name}: {e}")
        return

    if summarize:
        counts = df["emotion"].value_counts().to_dict()
        summary = f"{out_file.name} — " + ", ".join(f"{k}: {v}" for k, v in counts.items())
        summary_lines.append(summary)

def run(
    model_name: str,
    bible: str,
    dry_run: Path = None,
    summarize: bool = True,
    device: int = 0,
    logger: logging.Logger = None
):
    classifier = load_classifier(model_name, device, logger)
    input_dir = Path("data/processed") / bible
    output_dir = Path("data/labeled") / bible / "emotion"
    log_dir = DEFAULT_LOG_DIR
    summary_lines = []

    files = [dry_run] if dry_run else list(input_dir.glob("*.csv"))
    if not files:
        logger.warning("No input files found.")
        return

    processed_count = 0
    failed_count = 0

    for file in files:
        n_summary = len(summary_lines)
        process_file(file, classifier, output_dir, summary_lines, summarize, logger)
        if len(summary_lines) > n_summary:
            processed_count += 1
        else:
            failed_count += 1

    # Save the summary log if summarization is enabled
    if summarize and summary_lines:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"emotion_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        try:
            log_path.write_text("\n".join(summary_lines), encoding="utf-8")
            logger.info(f"Summary saved to {log_path}")
        except Exception as e:
            logger.error(f"Error saving summary log: {e}")

    logger.info(f"=== Summary: {processed_count} processed, {failed_count} failed ===")

# ========================
# === CLI ================
# ========================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="📘 Emotion Labeling Script")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)  # Model name
    parser.add_argument("--bible", type=str, default=DEFAULT_BIBLE)  # Dataset name
    parser.add_argument("--dry-run", type=Path)  # Process a single file for testing
    parser.add_argument("--device", type=int, default=0)  # Device ID (GPU or CPU)
    parser.add_argument("--no-summary", action="store_true")  # Disable summarization
    args = parser.parse_args()

    # Logging setup
    LOG_DIR = DEFAULT_LOG_DIR
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"emotion_labeling_{timestamp}.log"
    logger = setup_logger(log_path, level=logging.INFO, console=True)

    run(
        model_name=args.model,
        bible=args.bible,
        dry_run=args.dry_run,
        summarize=not args.no_summary,
        device=args.device,
        logger=logger
    )
