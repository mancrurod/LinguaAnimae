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

# Default model for zero-shot classification
DEFAULT_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"

# Default dataset (Bible version)
DEFAULT_BIBLE = "bible_kjv"

# Default labels for theme classification
DEFAULT_LABELS = ["love", "faith", "hope", "forgiveness", "fear"]

# Default threshold for classification confidence
DEFAULT_THRESHOLD = 0.7

# Default directory for logs
DEFAULT_LOG_DIR = Path("data/logs")

# Batch size for processing text
BATCH_SIZE = 16

# ========================
# === MODEL LOADER =======
# ========================

def load_classifier(model_name: str, device: int = 0):
    # Determine device (GPU or CPU) for model loading
    device_id = device if torch.cuda.is_available() else -1
    print(f"📦 Loading model: {model_name} (device: {'cuda' if device_id >= 0 else 'cpu'})")
    # Load the zero-shot classification pipeline
    return pipeline("zero-shot-classification", model=model_name, device=device_id)

# ========================
# === THEME CLASSIFY =====
# ========================

def classify_batch(texts: List[str], classifier, labels: List[str], threshold: float) -> List[str]:
    # Perform zero-shot classification on a batch of texts
    results = classifier(texts, candidate_labels=labels, multi_label=True)
    if not isinstance(results, list): results = [results]  # Ensure results are a list
    # Filter labels based on the threshold and return them as a semicolon-separated string
    return [
        ";".join([l for l, s in zip(r["labels"], r["scores"]) if s >= threshold])
        for r in results
    ]

def process_file(file: Path, classifier, output_dir: Path, labels: List[str],
                 threshold: float, summary_lines: List[str], summarize: bool) -> None:
    print(f"🔍 Processing: {file.name}")
    # Read the input CSV file
    df = pd.read_csv(file)
    # Check if required columns are present
    if "text" not in df.columns or "emotion" not in df.columns:
        print(f"⚠️ Skipped (missing 'text' or 'emotion'): {file.name}")
        return

    # Extract text data for classification
    texts = df["text"].astype(str).tolist()
    themes = []

    # Process texts in batches
    for i in range(0, len(texts), BATCH_SIZE):
        themes.extend(classify_batch(texts[i:i + BATCH_SIZE], classifier, labels, threshold))

    # Add the classified themes to the DataFrame
    df["theme"] = themes
    # Ensure the output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    # Save the processed file with themes
    out_file = output_dir / file.name.replace("_emotion.csv", "_emotion_theme.csv")
    df.to_csv(out_file, index=False)
    print(f"✅ Saved: {out_file.name}")

    if summarize:
        # Generate a summary of theme counts
        all_themes = df["theme"].dropna().str.split(";").explode()
        counts = all_themes.value_counts().to_dict()
        summary = f"{out_file.name} — " + ", ".join(f"{k}: {v}" for k, v in counts.items())
        summary_lines.append(summary)

def run(model_name: str, bible: str, labels: List[str], threshold: float,
        dry_run: Path = None, summarize: bool = True, device: int = 0):
    # Load the classifier model
    classifier = load_classifier(model_name, device)
    # Define input and output directories
    input_dir = Path("data/labeled") / bible / "emotion"
    output_dir = Path("data/labeled") / bible / "emotion_theme"
    log_dir = DEFAULT_LOG_DIR
    summary_lines = []

    # Get the list of files to process
    files = [dry_run] if dry_run else list(input_dir.glob("*_emotion.csv"))
    if not files:
        print("📁 No emotion-labeled files found.")
        return

    for file in files:
        # Skip files that are already processed
        result_path = output_dir / file.name.replace("_emotion.csv", "_emotion_theme.csv")
        if result_path.exists():
            print(f"⏩ Skipped (already labeled): {file.name}")
            continue
        # Process each file
        process_file(file, classifier, output_dir, labels, threshold, summary_lines, summarize)

    if summarize and summary_lines:
        # Save the summary log
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"theme_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        log_path.write_text("\n".join(summary_lines), encoding="utf-8")
        print(f"📝 Summary saved to {log_path}")

# ========================
# === CLI INTERFACE ======
# ========================

if __name__ == "__main__":
    # Define command-line arguments
    parser = argparse.ArgumentParser(description="📘 Theme Labeling Script")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)  # Model name
    parser.add_argument("--bible", type=str, default=DEFAULT_BIBLE)  # Dataset name
    parser.add_argument("--labels", nargs="+", default=DEFAULT_LABELS)  # Theme labels
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)  # Confidence threshold
    parser.add_argument("--dry-run", type=Path)  # Single file for testing
    parser.add_argument("--device", type=int, default=0)  # Device ID (GPU/CPU)
    parser.add_argument("--no-summary", action="store_true")  # Disable summary generation
    args = parser.parse_args()

    # Run the theme labeling process
    run(
        model_name=args.model,
        bible=args.bible,
        labels=args.labels,
        threshold=args.threshold,
        dry_run=args.dry_run,
        summarize=not args.no_summary,
        device=args.device
    )
