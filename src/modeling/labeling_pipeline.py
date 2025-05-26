"""
Labeling pipeline for Bible corpora (emotion + theme) using Hugging Face models.

- Processes all (or a specific) cleaned CSVs for emotion and then theme labeling.
- Emotion: text-classification pipeline.
- Theme: zero-shot-classification pipeline with user-defined labels and threshold.
- Saves per-file logs and a session summary for traceability and auditing.
- CLI configurable.

Usage:
    python labeling_pipeline.py --bible bible_kjv --emotion-model ... --theme-model ... --labels love faith hope
"""

import pandas as pd
from transformers import pipeline
from pathlib import Path
from typing import List
from datetime import datetime
import argparse
import torch
from time import perf_counter
import logging


# ========================
# === CONFIG DEFAULTS ====
# ========================

DEFAULT_BIBLE = "bible_kjv"
DEFAULT_MODEL_EMOTION = "j-hartmann/emotion-english-distilroberta-base"
DEFAULT_MODEL_THEME = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
DEFAULT_THEME_LABELS = ["love", "faith", "hope", "forgiveness", "fear"]
DEFAULT_THRESHOLD = 0.7
DEFAULT_BATCH_SIZE = 32
DEFAULT_LOG_DIR = Path("logs/labeling_logs")

# =========================
# === LOGGER SETUP    =====
# =========================

def setup_logger(
    log_path: Path,
    level: int = logging.INFO,
    console: bool = True,
    log_name: str = "labeling_pipeline_logger"
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
            console_formatter = logging.Formatter("%(levelname)s: %(message)s")
            ch.setFormatter(console_formatter)
            logger.addHandler(ch)
    return logger

# ========================
# === MODEL LOADER =======
# ========================

def load_classifier(model_name: str, device: int = 0, task: str = "text-classification", logger: logging.Logger = None):
    device_id = device if torch.cuda.is_available() else -1
    msg = f"Loading model: {model_name} (task: {task}) on {'cuda' if device_id >= 0 else 'cpu'}"
    if logger:
        logger.info(msg)
    else:
        print(f"📦 {msg}")
    try:
        return pipeline(task, model=model_name, device=device_id, top_k=None if task == "text-classification" else None)
    except Exception as e:
        error_msg = f"Error loading model {model_name} ({task}): {e}"
        if logger:
            logger.error(error_msg)
        else:
            print(f"❌ {error_msg}")
        raise

# ========================
# === EMOTION STAGE ======
# ========================

def classify_emotion_batch(texts: List[str], classifier, logger: logging.Logger) -> List[str]:
    try:
        outputs = classifier(texts)
        return [max(o, key=lambda x: x["score"])["label"] for o in outputs]
    except Exception as e:
        logger.error(f"Error in classify_emotion_batch: {e}")
        return ["error"] * len(texts)

def process_emotion(file: Path, output_dir: Path, classifier, batch_size: int, logger: logging.Logger) -> Path:
    try:
        df = pd.read_csv(file)
    except Exception as e:
        logger.error(f"Error reading file {file.name}: {e}")
        return None
    if "text" not in df.columns:
        logger.warning(f"Skipped (no 'text' column): {file.name}")
        return None

    texts = df["text"].astype(str).tolist()
    emotions = []
    for i in range(0, len(texts), batch_size):
        emotions.extend(classify_emotion_batch(texts[i:i + batch_size], classifier, logger))

    if len(emotions) != len(df):
        logger.error(f"Mismatch: {len(emotions)} predictions for {len(df)} rows in {file.name}")
        return None

    df["emotion"] = emotions
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / file.name.replace("_cleaned.csv", "_emotion.csv")
    try:
        df.to_csv(out_path, index=False)
        logger.info(f"Emotion-labeled file saved: {out_path.name}")
    except Exception as e:
        logger.error(f"Error saving file {out_path}: {e}")
        return None
    return out_path

# ========================
# === THEME STAGE =========
# ========================

def classify_theme_batch(texts: List[str], classifier, labels: List[str], threshold: float, logger: logging.Logger) -> List[str]:
    try:
        results = classifier(texts, candidate_labels=labels, multi_label=True)
        if not isinstance(results, list):
            results = [results]
        return [
            ";".join([l for l, s in zip(r["labels"], r["scores"]) if s >= threshold])
            for r in results
        ]
    except Exception as e:
        logger.error(f"Error in classify_theme_batch: {e}")
        return ["error"] * len(texts)

def process_theme(
    file: Path,
    output_dir: Path,
    classifier,
    labels: List[str],
    threshold: float,
    batch_size: int,
    logger: logging.Logger
) -> Path:
    try:
        df = pd.read_csv(file)
    except Exception as e:
        logger.error(f"Error reading file {file.name}: {e}")
        return None
    if "text" not in df.columns or "emotion" not in df.columns:
        logger.warning(f"Skipped (missing 'text' or 'emotion'): {file.name}")
        return None

    texts = df["text"].astype(str).tolist()
    themes = []
    for i in range(0, len(texts), batch_size):
        themes.extend(classify_theme_batch(texts[i:i + batch_size], classifier, labels, threshold, logger))

    if len(themes) != len(df):
        logger.error(f"Mismatch: {len(themes)} theme predictions for {len(df)} rows in {file.name}")
        return None

    df["theme"] = themes
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / file.name.replace("_emotion.csv", "_emotion_theme.csv")
    try:
        df.to_csv(out_path, index=False)
        logger.info(f"Theme-labeled file saved: {out_path.name}")
    except Exception as e:
        logger.error(f"Error saving file {out_path}: {e}")
        return None
    return out_path

# ========================
# === PIPELINE RUNNER =====
# ========================

# Main function to run the entire pipeline
def run_pipeline(
    bible: str,
    emotion_model: str,
    theme_model: str,
    theme_labels: List[str],
    threshold: float,
    device: int,
    skip_emotion: bool,
    skip_theme: bool,
    dry_run: Path = None,
    logger: logging.Logger = None
):
    # Define paths
    base = Path("data")
    processed_dir = base / "processed" / bible
    labeled_base = base / "labeled" / bible
    emotion_dir = labeled_base / "emotion"
    theme_dir = labeled_base / "emotion_theme"
    log_dir = DEFAULT_LOG_DIR
    log_dir.mkdir(parents=True, exist_ok=True)

    # Input files
    files = [dry_run] if dry_run else list(processed_dir.glob("*.csv"))
    if not files:
        logger.error("No input files found.")
        return

    # Timing & summary logs
    start_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    global_start = perf_counter()
    total_emotion_time = 0.0
    total_theme_time = 0.0
    time_log = log_dir / f"pipeline_timings_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    # Load models (catch and log errors)
    try:
        classifier_emotion = None
        if not skip_emotion:
            classifier_emotion = load_classifier(emotion_model, device, task="text-classification", logger=logger)
        classifier_theme = None
        if not skip_theme:
            classifier_theme = load_classifier(theme_model, device, task="zero-shot-classification", logger=logger)
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        return

    logger.info(f"Pipeline started: {start_time_str}")
    logger.info("Emotion Labeling Stage:")

    # EMOTION LABELING
    for idx, file in enumerate(files, start=1):
        logger.info(f"[{idx}/{len(files)}] {file.name}")
        start = perf_counter()

        emotion_file = emotion_dir / file.name.replace("_cleaned.csv", "_emotion.csv")
        if skip_emotion:
            logger.info("Skipped (flagged)")
            continue
        if emotion_file.exists():
            logger.info("Skipped (already labeled)")
            continue

        out_path = process_emotion(file, emotion_dir, classifier_emotion, DEFAULT_BATCH_SIZE, logger)
        if not out_path:
            logger.warning(f"Emotion processing failed for file: {file.name}")
            continue

        duration = perf_counter() - start
        total_emotion_time += duration
        logger.info(f"Done in {duration:.2f}s")
        with open(time_log, "a", encoding="utf-8") as f:
            f.write(f"{file.name}, emotion, {duration:.2f}s, {datetime.now().isoformat()}\n")

    # THEME LABELING
    logger.info("Theme Labeling Stage:")
    theme_files = list(emotion_dir.glob("*_emotion.csv"))
    for idx, file in enumerate(theme_files, start=1):
        logger.info(f"[{idx}/{len(theme_files)}] {file.name}")

        result_path = theme_dir / file.name.replace("_emotion.csv", "_emotion_theme.csv")
        if skip_theme:
            logger.info("Skipped (flagged)")
            continue
        if result_path.exists():
            logger.info("Skipped (already labeled)")
            continue

        start = perf_counter()
        out_path = process_theme(file, theme_dir, classifier_theme, theme_labels, threshold, DEFAULT_BATCH_SIZE, logger)
        if not out_path:
            logger.warning(f"Theme processing failed for file: {file.name}")
            continue

        duration = perf_counter() - start
        total_theme_time += duration
        logger.info(f"Done in {duration:.2f}s")
        with open(time_log, "a", encoding="utf-8") as f:
            f.write(f"{file.name}, theme, {duration:.2f}s, {datetime.now().isoformat()}\n")

    # Summary
    total_runtime = perf_counter() - global_start
    summary_lines = [
        f"\nPipeline started: {start_time_str}",
        f"Files processed: {len(files)}",
        f"Total emotion time: {total_emotion_time:.2f}s",
        f"Total theme time: {total_theme_time:.2f}s",
        f"Total runtime: {total_runtime:.2f}s"
    ]
    with open(time_log, "a", encoding="utf-8") as f:
        f.write("\n" + "\n".join(summary_lines) + "\n")
    logger.info("\n".join(summary_lines))
    logger.info(f"Summary log saved to: {time_log}")

# ========================
# === CLI INTERFACE ======
# ========================

# Command-line interface for the pipeline
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="📘 Run Emotion + Theme Labeling Pipeline")
    parser.add_argument("--bible", type=str, default=DEFAULT_BIBLE, help="Bible name folder")
    parser.add_argument("--emotion-model", type=str, default=DEFAULT_MODEL_EMOTION)
    parser.add_argument("--theme-model", type=str, default=DEFAULT_MODEL_THEME)
    parser.add_argument("--labels", nargs="+", default=DEFAULT_THEME_LABELS)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--dry-run", type=Path, help="Run on a single file from processed/")
    parser.add_argument("--skip-emotion", action="store_true")
    parser.add_argument("--skip-theme", action="store_true")
    parser.add_argument("--logfile", type=Path, default=None, help="Path to log file (default: logs/labeling_logs/pipeline_<timestamp>.log)")

    args = parser.parse_args()

    # Logging setup
    LOG_DIR = DEFAULT_LOG_DIR
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.logfile:
        log_path = args.logfile
    else:
        log_path = LOG_DIR / f"pipeline_{timestamp}.log"
    logger = setup_logger(log_path, level=logging.INFO, console=True)

    logger.info("==== Starting labeling pipeline ====")
    run_pipeline(
        bible=args.bible,
        emotion_model=args.emotion_model,
        theme_model=args.theme_model,
        theme_labels=args.labels,
        threshold=args.threshold,
        device=args.device,
        skip_emotion=args.skip_emotion,
        skip_theme=args.skip_theme,
        dry_run=args.dry_run,
        logger=logger
    )
    logger.info("==== Pipeline finished ====")
