"""
Apply emotion classification to cleaned Bible verses (bible_kjv) using the j-hartmann model.

Adds the dominant emotion label to the 'emotion' column.
Saves new files as *_labelled.csv and generates a summary log.
"""

# ========================
# === IMPORTS ============
# ========================

import pandas as pd
from transformers import pipeline
from pathlib import Path
from typing import Dict
from datetime import datetime

# ========================
# === CONFIG =============
# ========================

MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"
INPUT_DIR = Path("data/processed/bible_kjv")
OUTPUT_DIR = Path("data/annotated/bible_kjv")
LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / f"emotion_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
EMOTION_COL = "emotion"

# ========================
# === FUNCTIONS ==========
# ========================

def load_classifier(model_name: str):
    """Load HuggingFace emotion classification pipeline."""
    print(f"📦 Loading model: {model_name}")
    return pipeline("text-classification", model=model_name, top_k=None)

def classify_emotion(text: str, classifier) -> str:
    """Return the dominant emotion label for a given text."""
    try:
        scores = classifier(text)[0]
        best = max(scores, key=lambda x: x["score"])
        return best["label"]
    except Exception as e:
        print(f"❌ Error classifying: {text[:30]}... – {e}")
        return ""

def rename_to_labelled(file: Path) -> Path:
    """Rename output file with _labelled.csv suffix."""
    base = file.stem.replace("_cleaned", "")
    return file.with_name(base + "_labelled.csv")

def process_file(path: Path, classifier, log_lines: list) -> None:
    """Apply emotion classification to a single CSV file and log distribution."""
    print(f"🔍 Processing: {path.name}")
    df = pd.read_csv(path)

    if "text" not in df.columns:
        print(f"⚠️ No 'text' column in {path.name}")
        return

    df[EMOTION_COL] = df["text"].apply(lambda t: classify_emotion(str(t), classifier))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    new_filename = rename_to_labelled(path)
    output_path = OUTPUT_DIR / new_filename.name
    df.to_csv(output_path, index=False)
    print(f"✅ Saved: {output_path.name}")

    # Log emotion distribution
    counts = df[EMOTION_COL].value_counts().to_dict()
    summary = f"{output_path.name} — " + ", ".join(f"{k}: {v}" for k, v in counts.items())
    log_lines.append(summary)

def run_emotion_labeling():
    """Run emotion labeling and log summary for all processed Bible files."""
    classifier = load_classifier(MODEL_NAME)
    files = list(INPUT_DIR.glob("*.csv"))
    if not files:
        print("📁 No CSV files found in input directory.")
        return

    log_lines = []
    for file in files:
        process_file(file, classifier, log_lines)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    LOG_FILE.write_text("\n".join(log_lines), encoding="utf-8")
    print(f"📝 Emotion summary saved to {LOG_FILE}")

# ========================
# === MAIN ===============
# ========================

if __name__ == "__main__":
    run_emotion_labeling()
