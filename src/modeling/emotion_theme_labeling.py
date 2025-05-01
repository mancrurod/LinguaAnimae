import pandas as pd
from transformers import pipeline
from pathlib import Path
from typing import List
from datetime import datetime
import argparse
import torch

# ========================
# === CONFIG DEFAULTS ====
# ========================

# Default model to use for emotion classification
DEFAULT_MODEL = "j-hartmann/emotion-english-distilroberta-base"
# Default dataset (bible version) to process
DEFAULT_BIBLE = "bible_kjv"
# Default directory for logs
DEFAULT_LOG_DIR = Path("data/logs")
# Batch size for processing text
BATCH_SIZE = 32

# ========================
# === MODEL LOADER =======
# ========================

def load_classifier(model_name: str, device: int = 0):
    # Determine device (GPU if available, otherwise CPU)
    device_id = device if torch.cuda.is_available() else -1
    print(f"📦 Loading model: {model_name} (device: {'cuda' if device_id >= 0 else 'cpu'})")
    # Load the text classification pipeline with the specified model
    return pipeline("text-classification", model=model_name, device=device_id, top_k=None)

# ========================
# === EMOTION CLASSIFY ===
# ========================

def classify_batch(texts: List[str], classifier) -> List[str]:
    # Classify a batch of texts and return the highest-scoring emotion label for each
    results = classifier(texts)
    return [max(r, key=lambda x: x["score"])["label"] for r in results]

def process_file(file: Path, classifier, output_dir: Path, summary_lines: List[str], summarize: bool) -> None:
    print(f"🔍 Processing: {file.name}")
    # Read the input CSV file
    df = pd.read_csv(file)
    # Skip files that do not have a "text" column
    if "text" not in df.columns:
        print(f"⚠️ Skipped (missing 'text' column): {file.name}")
        return

    # Convert the "text" column to a list of strings
    texts = df["text"].astype(str).tolist()
    emotions = []

    # Process texts in batches
    for i in range(0, len(texts), BATCH_SIZE):
        emotions.extend(classify_batch(texts[i:i + BATCH_SIZE], classifier))

    # Add the predicted emotions as a new column in the DataFrame
    df["emotion"] = emotions

    # Ensure the output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    # Save the processed file with a new name
    out_file = output_dir / file.name.replace("_cleaned.csv", "_emotion.csv")
    df.to_csv(out_file, index=False)
    print(f"✅ Saved: {out_file.name}")

    # If summarization is enabled, generate a summary of emotion counts
    if summarize:
        counts = df["emotion"].value_counts().to_dict()
        summary = f"{out_file.name} — " + ", ".join(f"{k}: {v}" for k, v in counts.items())
        summary_lines.append(summary)

def run(model_name: str, bible: str, dry_run: Path = None, summarize: bool = True, device: int = 0):
    # Load the emotion classification model
    classifier = load_classifier(model_name, device)
    # Define input and output directories
    input_dir = Path("data/processed") / bible
    output_dir = Path("data/labeled") / bible / "emotion"
    log_dir = DEFAULT_LOG_DIR
    summary_lines = []

    # Get the list of files to process (single file for dry-run, or all files in the input directory)
    files = [dry_run] if dry_run else list(input_dir.glob("*.csv"))
    if not files:
        print("📁 No input files found.")
        return

    # Process each file
    for file in files:
        process_file(file, classifier, output_dir, summary_lines, summarize)

    # Save the summary log if summarization is enabled
    if summarize and summary_lines:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"emotion_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        log_path.write_text("\n".join(summary_lines), encoding="utf-8")
        print(f"📝 Summary saved to {log_path}")

# ========================
# === CLI ================
# ========================

if __name__ == "__main__":
    # Define command-line arguments
    parser = argparse.ArgumentParser(description="📘 Emotion Labeling Script")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)  # Model name
    parser.add_argument("--bible", type=str, default=DEFAULT_BIBLE)  # Dataset name
    parser.add_argument("--dry-run", type=Path)  # Process a single file for testing
    parser.add_argument("--device", type=int, default=0)  # Device ID (GPU or CPU)
    parser.add_argument("--no-summary", action="store_true")  # Disable summarization
    args = parser.parse_args()

    # Run the main function with parsed arguments
    run(
        model_name=args.model,
        bible=args.bible,
        dry_run=args.dry_run,
        summarize=not args.no_summary,
        device=args.device
    )
