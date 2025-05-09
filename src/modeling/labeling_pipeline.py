"""
This script performs emotion and theme classification using English Bible text
via Hugging Face models. The resulting labels are later transferred to the Spanish corpus.
"""


import pandas as pd
from transformers import pipeline
from pathlib import Path
from typing import List
from datetime import datetime
import argparse
import torch
from time import perf_counter

# ========================
# === CONFIG DEFAULTS ====
# ========================

# Default configuration values for the pipeline
DEFAULT_BIBLE = "bible_kjv"
DEFAULT_MODEL_EMOTION = "j-hartmann/emotion-english-distilroberta-base"
DEFAULT_MODEL_THEME = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
DEFAULT_THEME_LABELS = ["love", "faith", "hope", "forgiveness", "fear"]
DEFAULT_THRESHOLD = 0.7
DEFAULT_BATCH_SIZE = 32

# ========================
# === MODEL LOADER =======
# ========================

# Function to load a Hugging Face model pipeline
def load_classifier(model_name: str, device: int = 0, task: str = "text-classification"):
    device_id = device if torch.cuda.is_available() else -1  # Use GPU if available, otherwise CPU
    print(f"📦 Loading model: {model_name} (task: {task}) on {'cuda' if device_id >= 0 else 'cpu'}")
    return pipeline(task, model=model_name, device=device_id, top_k=None if task == "text-classification" else None)

# ========================
# === EMOTION STAGE ======
# ========================

# Classify emotions for a batch of texts
def classify_emotion_batch(texts: List[str], classifier) -> List[str]:
    outputs = classifier(texts)  # Run the classifier on the batch
    return [max(o, key=lambda x: x["score"])["label"] for o in outputs]  # Extract the label with the highest score

# Process emotion classification for a single file
def process_emotion(file: Path, output_dir: Path, classifier, batch_size: int) -> Path:
    df = pd.read_csv(file)  # Load the input CSV file
    if "text" not in df.columns:  # Check if the required column exists
        print(f"⚠️ Skipped (no 'text' column): {file.name}")
        return None

    texts = df["text"].astype(str).tolist()  # Convert the text column to a list of strings
    emotions = []
    for i in range(0, len(texts), batch_size):  # Process texts in batches
        emotions.extend(classify_emotion_batch(texts[i:i + batch_size], classifier))

    df["emotion"] = emotions  # Add the emotion labels to the DataFrame
    output_dir.mkdir(parents=True, exist_ok=True)  # Ensure the output directory exists
    out_path = output_dir / file.name.replace("_cleaned.csv", "_emotion.csv")  # Define the output file path
    df.to_csv(out_path, index=False)  # Save the updated DataFrame to a CSV file
    return out_path

# ========================
# === THEME STAGE =========
# ========================

# Classify themes for a batch of texts
def classify_theme_batch(texts: List[str], classifier, labels: List[str], threshold: float) -> List[str]:
    results = classifier(texts, candidate_labels=labels, multi_label=True)  # Run zero-shot classification
    if not isinstance(results, list): results = [results]  # Ensure results are a list
    return [
        ";".join([l for l, s in zip(r["labels"], r["scores"]) if s >= threshold])  # Filter labels by threshold
        for r in results
    ]

# Process theme classification for a single file
def process_theme(file: Path, output_dir: Path, classifier, labels: List[str], threshold: float, batch_size: int) -> Path:
    df = pd.read_csv(file)  # Load the input CSV file
    if "text" not in df.columns or "emotion" not in df.columns:  # Check if required columns exist
        print(f"⚠️ Skipped (missing 'text' or 'emotion'): {file.name}")
        return None

    texts = df["text"].astype(str).tolist()  # Convert the text column to a list of strings
    themes = []
    for i in range(0, len(texts), batch_size):  # Process texts in batches
        themes.extend(classify_theme_batch(texts[i:i + batch_size], classifier, labels, threshold))

    df["theme"] = themes  # Add the theme labels to the DataFrame
    output_dir.mkdir(parents=True, exist_ok=True)  # Ensure the output directory exists
    out_path = output_dir / file.name.replace("_emotion.csv", "_emotion_theme.csv")  # Define the output file path
    df.to_csv(out_path, index=False)  # Save the updated DataFrame to a CSV file
    return out_path

# ========================
# === PIPELINE RUNNER =====
# ========================

# Main function to run the entire pipeline
def run_pipeline(bible: str, emotion_model: str, theme_model: str, theme_labels: List[str], threshold: float,
                 device: int, skip_emotion: bool, skip_theme: bool, dry_run: Path = None):

    # Define paths for input, output, and logs
    base = Path("data")
    processed_dir = base / "processed" / bible
    labeled_base = base / "labeled" / bible
    emotion_dir = labeled_base / "emotion"
    theme_dir = labeled_base / "emotion_theme"
    log_dir = Path("logs") / "labeling_logs"
    log_dir.mkdir(parents=True, exist_ok=True)  # Ensure the log directory exists

    # Get the list of input files
    files = [dry_run] if dry_run else list(processed_dir.glob("*.csv"))
    if not files:  # Exit if no files are found
        print("❌ No input files found.")
        return

    # Initialize timing and logging
    start_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    global_start = perf_counter()
    total_emotion_time = 0.0
    total_theme_time = 0.0
    time_log = log_dir / f"pipeline_timings_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    # Load models if not skipping stages
    if not skip_emotion:
        classifier_emotion = load_classifier(emotion_model, device, task="text-classification")
    if not skip_theme:
        classifier_theme = load_classifier(theme_model, device, task="zero-shot-classification")

    print(f"\n🚀 Starting pipeline: {start_time_str}")
    print(f"\n🔄 Emotion Labeling:")

    # Process emotion labeling
    for idx, file in enumerate(files, start=1):
        print(f"[{idx}/{len(files)}] {file.name}", end=" — ")
        start = perf_counter()

        emotion_file = emotion_dir / file.name.replace("_cleaned.csv", "_emotion.csv")
        if skip_emotion:  # Skip if flagged
            print("⏩ Skipped (flagged)")
            continue
        if emotion_file.exists():  # Skip if already processed
            print("⏩ Skipped (already labeled)")
            continue

        emotion_file = process_emotion(file, emotion_dir, classifier_emotion, DEFAULT_BATCH_SIZE)
        if not emotion_file:  # Skip if processing failed
            print("❌ Failed")
            continue

        duration = perf_counter() - start  # Calculate processing time
        total_emotion_time += duration
        print(f"✅ Done in {duration:.2f}s")

        # Log the timing information
        with open(time_log, "a", encoding="utf-8") as f:
            f.write(f"{file.name}, emotion, {duration:.2f}s, {datetime.now().isoformat()}\n")

    print(f"\n🔄 Theme Labeling:")
    theme_files = list(emotion_dir.glob("*_emotion.csv"))  # Get files with emotion labels

    # Process theme labeling
    for idx, file in enumerate(theme_files, start=1):
        print(f"[{idx}/{len(theme_files)}] {file.name}", end=" — ")

        result_path = theme_dir / file.name.replace("_emotion.csv", "_emotion_theme.csv")
        if skip_theme:  # Skip if flagged
            print("⏩ Skipped (flagged)")
            continue
        if result_path.exists():  # Skip if already processed
            print("⏩ Skipped (already labeled)")
            continue

        start = perf_counter()
        result = process_theme(file, theme_dir, classifier_theme, theme_labels, threshold, DEFAULT_BATCH_SIZE)

        if result:  # Log success
            duration = perf_counter() - start
            total_theme_time += duration
            print(f"✅ Done in {duration:.2f}s")
            with open(time_log, "a", encoding="utf-8") as f:
                f.write(f"{file.name}, theme, {duration:.2f}s, {datetime.now().isoformat()}\n")
        else:  # Log failure
            print("❌ Failed")

    # Summarize and log the pipeline execution
    total_runtime = perf_counter() - global_start
    summary_lines = [
        f"\n📦 Pipeline started: {start_time_str}",
        f"📁 Files processed: {len(files)}",
        f"⚡ Total emotion time: {total_emotion_time:.2f}s",
        f"🎯 Total theme time: {total_theme_time:.2f}s",
        f"⏱️ Total runtime: {total_runtime:.2f}s"
    ]

    with open(time_log, "a", encoding="utf-8") as f:
        f.write("\n" + "\n".join(summary_lines) + "\n")

    print("\n🧾 Summary log saved to:", time_log)

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

    args = parser.parse_args()

    # Run the pipeline with the provided arguments
    run_pipeline(
        bible=args.bible,
        emotion_model=args.emotion_model,
        theme_model=args.theme_model,
        theme_labels=args.labels,
        threshold=args.threshold,
        device=args.device,
        skip_emotion=args.skip_emotion,
        skip_theme=args.skip_theme,
        dry_run=args.dry_run
    )
